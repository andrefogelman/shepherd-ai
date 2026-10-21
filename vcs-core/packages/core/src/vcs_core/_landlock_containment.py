"""Linux Landlock containment backend — the native syscall-deny tier (no container).

The Linux member of the `ContainmentBackend` family (`_containment.py`): unprivileged
Landlock LSM (kernel 5.13+) restricts filesystem writes to a writable root via raw syscalls
(444/445/446 + ``PR_SET_NO_NEW_PRIVS``). Unlike Seatbelt's external ``sandbox-exec -p <profile>``,
Landlock is **self-restriction-before-exec**: `launch` runs a confine-then-exec runner (this
module's ``__main__``) that applies the ruleset to itself and then ``execvp``s the command. The
backend's "profile" is therefore a backend-private string — the writable root (empty = none).

Lifted from the proven spike ``spikes/sandbox-jail/linux_landlock_fuse.py`` (8/8 in the
vcs-core-test container, ABI 7). Internal runtime surface — not part of the frozen consumer SPI.
"""

from __future__ import annotations

import ctypes
import json
import os
import pwd
import shutil
import stat
import subprocess
import sys
import warnings
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vcs_core._containment import JailNotEstablished

if TYPE_CHECKING:
    from collections.abc import Sequence

# Landlock raw syscalls (same numbers on x86_64 and aarch64); glibc lacks wrappers pre-2.36.
_NR_CREATE_RULESET, _NR_ADD_RULE, _NR_RESTRICT_SELF = 444, 445, 446
_PR_SET_NO_NEW_PRIVS = 38
_LL_CREATE_RULESET_VERSION = 1
_LL_RULE_PATH_BENEATH = 1
# v1 FS write-access bits (exclude v2 REFER / v3 TRUNCATE so a v1 kernel accepts the mask).
# Bit i = (1 << i): 1=write_file, 4=remove_dir, 5=remove_file, 6..12 = make_{char,dir,reg,sock,fifo,block,sym}.
# Reads/exec (bits 0,2,3) are intentionally NOT governed.
_WRITE_ACCESS = (
    (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12)
)

#: Shells discard output by writing to /dev/null (`cmd >/dev/null`, `2>/dev/null`).
#: Seatbelt allows writes under /dev; here every path outside the roots is denied, so
#: each such redirect failed with EACCES and the command exited 1 before it ran. A
#: confined Claude worker hit this on 26 of 173 Bash calls. Grant write_file on that one
#: character device, nothing under it and nothing beside it (/dev/shm stays denied).
_DEV_NULL = "/dev/null"
_WRITE_FILE = 1 << 1

#: Landlock (below ABI 6) does not govern connecting to a socket by path. The user's
#: runtime directory holds the session bus and systemd's private socket, so a confined
#: command could run `systemd-run --user -- cmd`: the service manager starts cmd as its own
#: child, outside this process tree, and the jail never applies to it. tmux and screen
#: servers do the same for `tmux new-window cmd` and `screen -X`. Where such a socket
#: directory exists and bubblewrap works, bubblewrap mounts an empty tmpfs over it before
#: the runner confines itself. Where bubblewrap is missing or cannot start, the launch
#: still runs, with that door open, and `exposure` says so.
_SERVICE_MANAGER_SOCKETS = ("bus", "systemd/private")

#: A trial bubblewrap run that takes longer than this counts as one that cannot start:
#: it runs before the first launch, and a hang there would hold the caller indefinitely.
_TRIAL_TIMEOUT_S = 10

#: Commands that wrap every launch from outside the jail (see `set_outer_prefix`).
_OUTER_PREFIX: list[str] = []

#: Whether bubblewrap could hide each tuple of directories, keyed by that tuple. Decided
#: once per process: a trial run costs a namespace setup on every launch otherwise.
_HIDING_WORKS: dict[tuple[str, ...], str | None] = {}


class JailExposureWarning(RuntimeWarning):
    """Raised per launch while the jail cannot hide the command servers' sockets.

    The reason is decided once per process, so every warning of a process repeats the
    same text; a caller that already showed `exposure()` can filter this category.
    """


_RUNNER_SENTINEL = "__landlock_confine_exec__"
_CONFINE_FAILED_RC = 3  # runner exit code when confinement could not be established (touch never uses 3)

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]  # ABI v1: single field


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # packed: u64 + s32 = 12 bytes (a classic Landlock gotcha)
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _syscall(*args: Any) -> int:
    conv = [a if not isinstance(a, int) else ctypes.c_long(a) for a in args]
    return int(_LIBC.syscall(*conv))


def landlock_abi() -> int:
    """Landlock ABI version: >=1 supported; -38 (ENOSYS) no kernel support; -1 (EPERM) blocked."""
    rv = _syscall(_NR_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint(_LL_CREATE_RULESET_VERSION))
    return int(rv) if rv >= 0 else -ctypes.get_errno()


def landlock_confine(writable_dirs: Sequence[str]) -> None:
    """Restrict this thread (and its children) so writes are allowed only beneath the roots.

    Reads/exec are not governed. An empty ``writable_dirs`` confines with NO writable root
    (every write denied — the ReadOnly case); each root adds one PATH_BENEATH rule, so a
    proper subset of the workspace yields per-binding grants. Raises OSError on any
    setup failure so the caller can fail closed.
    """
    attr = _RulesetAttr(handled_access_fs=_WRITE_ACCESS)
    fd = _syscall(_NR_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    fd = int(fd)
    for writable_dir in writable_dirs:
        dirfd = os.open(writable_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            rule = _PathBeneathAttr(allowed_access=_WRITE_ACCESS, parent_fd=dirfd)
            if (
                _syscall(
                    _NR_ADD_RULE,
                    ctypes.c_int(fd),
                    ctypes.c_int(_LL_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint(0),
                )
                < 0
            ):
                raise OSError(ctypes.get_errno(), "landlock_add_rule")
        finally:
            os.close(dirfd)
    _allow_dev_null(fd)
    if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
    if _syscall(_NR_RESTRICT_SELF, ctypes.c_int(fd), ctypes.c_uint(0)) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")


def _allow_dev_null(ruleset_fd: int) -> None:
    """Add a write_file-only rule for /dev/null to the ruleset.

    Landlock accepts a PATH_BENEATH rule on a single file when it grants only file
    rights. Any failure (no /dev/null, an older kernel refusing the rule) leaves the
    ruleset as it was: the jail stays at least as strict as before, never looser.
    """
    try:
        # O_PATH (Linux): a handle on the device that never opens it.
        nullfd = os.open(_DEV_NULL, getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC)
    except OSError:
        return
    try:
        rule = _PathBeneathAttr(allowed_access=_WRITE_FILE, parent_fd=nullfd)
        _syscall(
            _NR_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LL_RULE_PATH_BENEATH),
            ctypes.byref(rule),
            ctypes.c_uint(0),
        )
    finally:
        os.close(nullfd)


def set_outer_prefix(argv: Sequence[str]) -> None:
    """Put ``argv`` in front of every launch in this process, outside the jail.

    For a wrapper that has to sit above the jail, such as a resource limit
    (``systemd-run --user --scope -p MemoryMax=... --``): inside the jail it would have to
    reach the service manager, which the jail hides. The prefix runs unconfined, so the
    caller owns what it puts there. An empty ``argv`` removes it.
    """
    _OUTER_PREFIX[:] = list(argv)


def outer_prefix() -> tuple[str, ...]:
    """The argv set by `set_outer_prefix` (empty when none)."""
    return tuple(_OUTER_PREFIX)


def _service_manager_dirs() -> tuple[str, ...]:
    """The user's runtime directories that hold a service-manager socket."""
    found: list[str] = []
    for candidate in (os.environ.get("XDG_RUNTIME_DIR", ""), f"/run/user/{os.getuid()}"):
        if not candidate:
            continue
        real = os.path.realpath(candidate)
        if real not in found and any((Path(real) / s).exists() for s in _SERVICE_MANAGER_SOCKETS):
            found.append(real)
    return tuple(found)


def _terminal_server_dirs() -> tuple[str, ...]:
    """This user's tmux and screen socket directories that exist."""
    uid = os.getuid()
    try:
        user = pwd.getpwuid(uid).pw_name
    except KeyError:
        user = ""
    candidates = [str(Path(os.environ.get("TMUX_TMPDIR") or "/tmp") / f"tmux-{uid}")]
    if os.environ.get("SCREENDIR"):
        candidates.append(os.environ["SCREENDIR"])
    if user:
        candidates += [f"/run/screen/S-{user}", f"/var/run/screen/S-{user}"]
    found: list[str] = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real not in found and Path(real).is_dir():
            found.append(real)
    return tuple(found)


def _hidden_dirs() -> tuple[str, ...]:
    """Every directory the jail hides: the service manager's, then tmux's and screen's."""
    found = list(_service_manager_dirs())
    for extra in _terminal_server_dirs():
        if extra not in found:
            found.append(extra)
    return tuple(found)


def _find_bwrap() -> str | None:
    return shutil.which("bwrap")


def _bwrap_argv(bwrap: str, dirs: Sequence[str], keep: Sequence[str] = ()) -> list[str]:
    """Bubblewrap argv that keeps the host view and mounts an empty tmpfs over ``dirs``.

    A path in ``keep`` that falls under a hidden directory (a workspace or writable root
    under the runtime directory, for one) is bound back on top of the tmpfs, so the
    command still finds it. Writes there stay governed by the Landlock ruleset.
    """
    argv = [bwrap, "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    for hidden in dirs:
        argv += ["--tmpfs", hidden]
    for path in dict.fromkeys(os.path.realpath(p) for p in keep):
        if any(Path(path).is_relative_to(hidden) for hidden in dirs):
            argv += ["--bind", path, path]
    return [*argv, "--die-with-parent", "--"]


#: Exits 1 when a hidden directory holds anything besides the entries leading to a path
#: bound back into it. argv[1] is JSON: [[directory, [allowed entry, ...]], ...].
_HIDDEN_DIRS_CHECK = (
    "import json, os, sys; sys.exit(1 if any(set(os.listdir(d)) - set(a) for d, a in json.loads(sys.argv[1])) else 0)"
)


def _hidden_dirs_spec(dirs: Sequence[str], keep: Sequence[str] = ()) -> str:
    """The `_HIDDEN_DIRS_CHECK` argument for ``dirs``.

    Each directory goes with the first component of every kept path under it: the only
    entries allowed to show inside the jail.
    """
    spec = []
    for hidden in dirs:
        allowed = sorted(
            {
                Path(os.path.realpath(p)).relative_to(hidden).parts[0]
                for p in keep
                if Path(os.path.realpath(p)).is_relative_to(hidden) and Path(os.path.realpath(p)) != Path(hidden)
            }
        )
        spec.append([hidden, allowed])
    return json.dumps(spec)


def _why_not_hidden(dirs: tuple[str, ...]) -> str | None:
    """None when bubblewrap hides ``dirs``; otherwise why it cannot."""
    if dirs not in _HIDING_WORKS:
        bwrap = _find_bwrap()
        if bwrap is None:
            reason: str | None = "bubblewrap (bwrap) is not installed"
        else:
            try:
                trial = subprocess.run(
                    [*_bwrap_argv(bwrap, dirs), sys.executable, "-c", _HIDDEN_DIRS_CHECK, _hidden_dirs_spec(dirs)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_TRIAL_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                reason = f"bubblewrap did not finish a trial run within {_TRIAL_TIMEOUT_S} s"
            else:
                detail = (trial.stderr or trial.stdout).strip()
                reason = (
                    None if trial.returncode == 0 else f"bubblewrap could not hide them ({detail or trial.returncode})"
                )
        _HIDING_WORKS[dirs] = reason
    return _HIDING_WORKS[dirs]


def _docker_sockets(hidden: Sequence[str]) -> tuple[str, ...]:
    """Docker daemon sockets this user can write to and the jail does not hide."""
    candidates = ["/var/run/docker.sock", "/run/docker.sock"]
    host = os.environ.get("DOCKER_HOST", "")
    if host.startswith("unix://"):
        candidates.insert(0, host[len("unix://") :])
    found: list[str] = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real in found or any(Path(real).is_relative_to(h) for h in hidden):
            continue
        with suppress(OSError):
            if stat.S_ISSOCK(Path(real).stat().st_mode) and os.access(real, os.W_OK):
                found.append(real)
    return tuple(found)


def exposure() -> str | None:
    """A message for the user when the jail leaves a command server reachable, else None.

    Reachable means a confined command can have a process started outside the jail. The
    message names the paths, the reason, and how to close each one.
    """
    parts: list[str] = []
    dirs = _hidden_dirs()
    reason = _why_not_hidden(dirs) if dirs else None
    if reason is not None:
        parts.append(
            f"the jail does not hide {', '.join(dirs)}, which hold sockets of servers that run commands for "
            f"this user (the service manager, tmux, screen), and {reason}. A confined command can start "
            "processes outside the jail through them (`systemd-run --user`, `tmux new-window`), so its writes "
            "are no longer confined. Install bubblewrap to close this (Debian/Ubuntu: "
            "`sudo apt install bubblewrap`; Fedora: `sudo dnf install bubblewrap`); where AppArmor restricts "
            "unprivileged user namespaces (Ubuntu 23.10 and later), bwrap also needs the profile that allows "
            "them. Running without this protection."
        )
    hidden = dirs if reason is None else ()
    for sock in _docker_sockets(hidden):
        parts.append(
            f"{sock} is writable by this user: a confined command can drive the Docker daemon, which runs as "
            "root, and escape the jail through it. The jail does not hide it; remove the user from the docker "
            "group, or run where the socket is not writable, to close this."
        )
    return " Also: ".join(parts) if parts else None


def service_manager_exposure() -> str | None:
    """Same as `exposure` (the 0.3.6 name)."""
    return exposure()


def _hiding_argv(keep: Sequence[str] = ()) -> list[str]:
    """The bubblewrap prefix for this launch, or [] when nothing needs or can be hidden."""
    dirs = _hidden_dirs()
    if not dirs:
        return []
    bwrap = _find_bwrap()
    if bwrap is None or _why_not_hidden(dirs) is not None:
        warnings.warn(exposure() or "", JailExposureWarning, stacklevel=3)
        return []
    return _bwrap_argv(bwrap, dirs, keep)


class LandlockContainmentBackend:
    """Linux Landlock jail (unprivileged LSM). Native syscall-deny, no container."""

    name = "landlock"
    enforcement_tier = "native-syscall-deny"

    def available(self) -> tuple[bool, str]:
        if sys.platform != "linux":
            return (False, "Landlock is Linux-only")
        abi = landlock_abi()
        if abi >= 1:
            return (True, f"Landlock ABI {abi}")
        if abi == -38:
            return (False, "Landlock unsupported by this kernel (ENOSYS)")
        return (False, f"Landlock unavailable (errno {-abi}; e.g. blocked by seccomp)")

    def profile_for(self, writable_roots: Sequence[str], *, allow_network: bool) -> str:
        """Backend-private profile: a JSON array of the canonicalized writable roots.

        Empty array = ReadOnly (no writable root -> every write denied); one entry ==
        realpath(WORKDIR) = Permissive; a proper subset = per-binding grants. Landlock is
        filesystem-only, so ``allow_network`` is not enforced here (network confinement is
        the Seatbelt/egress-broker axis) — accepted for Protocol parity.
        """
        del allow_network  # Landlock governs filesystem writes only; network is a separate axis.
        return json.dumps([os.path.realpath(str(root)) for root in writable_roots])

    def launch(
        self,
        profile: str,
        working_root: Any,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        outer: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run command under a Landlock confine-then-exec runner (this module's ``__main__``).

        Order, outermost first: the outer prefix (unless ``outer`` is False), bubblewrap
        hiding the command servers' sockets (where one exists and bubblewrap works), the
        runner, the command. The probe passes ``outer=False``: its launches only touch a
        file, and a wrapper failing there would read as a denied write.
        """
        cwd = os.path.realpath(str(working_root))
        return subprocess.run(
            [
                *(_OUTER_PREFIX if outer else ()),
                *_hiding_argv((cwd, *json.loads(profile or "[]"))),
                sys.executable,
                "-m",
                "vcs_core._landlock_containment",
                _RUNNER_SENTINEL,
                profile,
                *command,
            ],
            cwd=cwd,
            env=env if env is not None else dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )

    def _write_result(self, profile: str, working_root: Any, target: Path) -> int:
        """Confined ``touch target``; return the runner rc: 0=allowed, 3=confine-failed, else denied.

        Cleanup is parent-side (unconfined), so it is reliable regardless of the outcome.
        """
        with suppress(FileNotFoundError):
            target.unlink()
        rc = self.launch(profile, working_root, ["/usr/bin/touch", "--", str(target)], outer=False).returncode
        with suppress(FileNotFoundError):
            target.unlink()
        return rc

    def probe(self, profile: str, working_root: Any, *, writable_roots: Sequence[str]) -> None:
        """Fail-closed (§6): prove the jail is BOTH live AND grant-conformant before the body.

        Deny-closed and per-root, mirroring the Seatbelt probe: (0) where bubblewrap hides
        the command servers' sockets, they are not reachable inside the jail; (1) an out-of-WORKSPACE write
        is denied; (2) a write beneath each declared writable root is allowed; (3) if WORKDIR
        is not itself a writable root, an in-WORKDIR write outside every root is denied.

        Exit-code-aware (unlike a presence-only check): a confine FAILURE (rc==3) is treated
        as "no jail" rather than "write denied", so a broken Landlock can never pass as a
        working jail.
        """
        wd = Path(os.path.realpath(str(working_root)))
        roots = [Path(os.path.realpath(str(root))) for root in writable_roots]

        def _denied(target: Path) -> bool:
            rc = self._write_result(profile, working_root, target)
            if rc == _CONFINE_FAILED_RC:
                raise JailNotEstablished("fail-closed: Landlock confinement could not be established")
            return rc != 0

        # (0) where hiding is in effect, check it inside the real jail. Runs first: a
        # bubblewrap that failed to start exits non-zero, which step (1) would read as a
        # denied write. Where hiding is not possible, `exposure` reports it.
        hidden = _hidden_dirs()
        if hidden and _why_not_hidden(hidden) is None:
            kept = (str(wd), *json.loads(profile or "[]"))
            done = self.launch(
                profile,
                working_root,
                [sys.executable, "-c", _HIDDEN_DIRS_CHECK, _hidden_dirs_spec(hidden, kept)],
                outer=False,
            )
            if done.returncode == _CONFINE_FAILED_RC:
                raise JailNotEstablished("fail-closed: Landlock confinement could not be established")
            if done.returncode != 0:
                detail = (done.stderr or done.stdout).strip()
                raise JailNotEstablished(
                    f"fail-closed: a command server's socket is reachable inside the jail ({detail})"
                )

        # (1) liveness: a write OUTSIDE the workspace must be denied.
        if not _denied(wd.parent / ".jail-probe"):
            raise JailNotEstablished("fail-closed: out-of-WORKDIR write was NOT denied — no jail established")

        # (2) per-root: each declared writable root must accept writes.
        for root in roots:
            if _denied(root / ".jail-probe-canary"):
                raise JailNotEstablished(
                    f"fail-closed: writable root {root} DENIES writes (profile too strict) — "
                    "a legit body's writes would spuriously fail"
                )

        # (3) deny-closed: an in-WORKDIR path outside every writable root must be denied.
        wd_is_writable = any(wd == root or wd.is_relative_to(root) for root in roots)
        if not wd_is_writable and not _denied(wd / ".jail-probe-denied"):
            raise JailNotEstablished(
                "fail-closed: an in-WORKDIR path outside every writable root was PERMITTED — would silently escalate"
            )


def _run_confined() -> int:
    """Runner entry: argv = [SENTINEL, roots_json, *command]. Confine self, then exec command.

    ``roots_json`` is the backend-private profile: a JSON array of writable roots (``[]`` =
    ReadOnly). JSON keeps the whole set in one argv element and is safe for arbitrary paths.
    """
    writable_roots = json.loads(sys.argv[2]) if sys.argv[2] else []
    command = sys.argv[3:]
    try:
        landlock_confine(writable_roots)
    except OSError:
        return _CONFINE_FAILED_RC  # fail-closed: never run the body if confinement failed
    if not command:
        return 0
    os.execvp(command[0], command)  # noqa: S606  intentional shell-free exec (the confined runner)
    return 127  # type: ignore[unreachable]  # os.execvp replaces the process or raises; never returns


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == _RUNNER_SENTINEL:
        raise SystemExit(_run_confined())
