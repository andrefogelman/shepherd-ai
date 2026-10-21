"""macOS Seatbelt containment backend — the native syscall-deny tier (no container).

`SeatbeltContainmentBackend` is the macOS member of the `ContainmentBackend` family
(`_containment.py`): `/usr/bin/sandbox-exec` with a `may=`-lowered SBPL profile. Moved
move-not-build from the skeleton staging (skeleton/src/skeleton/containment.py); the
lowering is verbatim from spikes/sandbox-jail/macos_may_lowering.py (15/15).

Internal runtime surface — not part of the frozen consumer SPI.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vcs_core._containment import JailNotEstablished

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_DEV = '(allow file-write* (subpath "/dev"))'  # /dev/null etc. — not the workspace


def _own_or_absent(candidate: str, is_kind: Callable[[int], bool]) -> str | None:
    """``candidate`` resolved, unless an entry of another kind or owner sits there.

    Seatbelt matches resolved paths, so the parent is resolved. The last component is
    followed only when it is a link this user made, and then the target must be this
    user's too: another user can plant a link at /tmp/tmux-<uid>. An absent entry keeps
    its rule, since a server started after the launch creates it there.
    """
    path = Path(os.path.realpath(Path(candidate).parent)) / Path(candidate).name
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) and st.st_uid == os.getuid():
            path = Path(os.path.realpath(path))
            st = path.lstat()
    except FileNotFoundError:
        return str(path)
    except OSError:
        return None
    return str(path) if is_kind(st.st_mode) and st.st_uid == os.getuid() else None


def _own_paths(candidates: Sequence[str], is_kind: Callable[[int], bool]) -> tuple[str, ...]:
    found = (_own_or_absent(c, is_kind) for c in candidates if c)
    return tuple(dict.fromkeys(p for p in found if p is not None))


def _skipped(candidates: Sequence[str], is_kind: Callable[[int], bool]) -> list[str]:
    """The candidates that exist but `_own_or_absent` refuses: no rule covers them."""
    return list(dict.fromkeys(c for c in candidates if c and os.path.lexists(c) and _own_or_absent(c, is_kind) is None))


def _terminal_server_dirs() -> tuple[str, ...]:
    """Where this user's tmux and screen servers keep their sockets, existing or not.

    `(allow default)` lets a confined command connect to any Unix socket, so `tmux
    new-window cmd` would have the user's tmux server start cmd outside the sandbox. A
    subpath deny also covers a directory created after the launch, so nothing is made here.
    """
    return _own_paths(_terminal_server_candidates(), stat.S_ISDIR)


def _terminal_server_candidates() -> list[str]:
    uid = os.getuid()
    candidates = [f"/tmp/tmux-{uid}"]
    if os.environ.get("TMUX_TMPDIR"):
        candidates.append(str(Path(os.environ["TMUX_TMPDIR"]) / f"tmux-{uid}"))
    if os.environ.get("SCREENDIR"):
        candidates.append(os.environ["SCREENDIR"])
    if os.environ.get("TMPDIR"):
        candidates.append(str(Path(os.environ["TMPDIR"]) / ".screen"))
    return candidates


def _named_sockets() -> tuple[str, ...]:
    """The session bus and the tmux server the environment names.

    Denied one by one, not by directory: that directory can be a shared one such as /tmp.
    """
    return _own_paths(_named_socket_candidates(), stat.S_ISSOCK)


def _named_socket_candidates() -> list[str]:
    candidates: list[str] = []
    for address in os.environ.get("DBUS_SESSION_BUS_ADDRESS", "").split(";"):
        if address.startswith("unix:"):
            candidates += [f[len("path=") :] for f in address[len("unix:") :].split(",") if f.startswith("path=")]
    if os.environ.get("TMUX"):
        candidates.append(os.environ["TMUX"].split(",")[0])
    return candidates


def _docker_sockets() -> tuple[str, ...]:
    """Docker daemon sockets this user can write to (Docker Desktop, Colima, OrbStack)."""
    home = Path.home()
    candidates = [
        "/var/run/docker.sock",
        str(home / ".docker/run/docker.sock"),
        str(home / ".colima/default/docker.sock"),
        str(home / ".orbstack/run/docker.sock"),
    ]
    host = os.environ.get("DOCKER_HOST", "")
    if host.startswith("unix://"):
        candidates.append(host[len("unix://") :])
    found: list[str] = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real not in found and Path(real).exists() and os.access(real, os.W_OK):
            found.append(real)
    return tuple(found)


def _sbpl_string(value: str) -> str:
    """Escape a path for a double-quoted SBPL string literal.

    SBPL string literals are double-quoted with C-style escapes; backslash and double-quote
    are the characters that can terminate or derail the literal, so escape those (backslash
    first). Security-shaped code must not depend on raw interpolation — a workspace path
    containing a double-quote would otherwise malform the profile.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def lower_to_seatbelt(
    writable_roots: Sequence[str],
    *,
    allow_network: bool,
    canonicalize: bool = True,
    deny_sockets: Sequence[str] = (),
    deny_socket_files: Sequence[str] = (),
) -> str:
    """`may=` -> Seatbelt SBPL, deny-closed over an arbitrary set of writable roots.

    The profile denies all file-writes, then re-allows writes beneath each canonicalized
    writable root (plus /dev). Zero roots = ReadOnly (no managed writable root at all); one
    root == realpath(WORKDIR) = the whole-workspace Permissive floor; a proper subset of the
    workspace = per-binding grants (``ReadWrite backend/`` beside ``ReadOnly docs/``).
    Anything outside every root is refused at the syscall — the deny-closed guarantee the
    per-binding soundness argument rests on.

    THE config-coupling seam: each writable root MUST be a realpath (Seatbelt does not resolve
    subpath symlinks, so an un-canonicalized root wrongly denies in-root writes via
    /var->/private/var). `canonicalize=False` exists only to demonstrate that gotcha in tests.
    Validated end-to-end in spikes/sandbox-jail/macos_may_lowering.py (15/15).

    Each directory in ``deny_sockets`` gets a `network-outbound` deny, which refuses a
    connect to any Unix socket beneath it; each path in ``deny_socket_files`` gets one for
    that socket alone. Unix sockets as a whole stay open: name
    resolution goes through mDNSResponder's socket.
    """
    roots = [os.path.realpath(root) if canonicalize else str(root) for root in writable_roots]
    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    if not allow_network:
        # No outbound network (no exfiltration). `allow default` above otherwise permits it.
        lines.append("(deny network-outbound)")
    lines.extend(f'(allow file-write* (subpath "{_sbpl_string(root)}"))' for root in roots)
    lines.append(_DEV)
    lines.extend(f'(deny network-outbound (subpath "{_sbpl_string(d)}"))' for d in deny_sockets)
    lines.extend(f'(deny network-outbound (literal "{_sbpl_string(f)}"))' for f in deny_socket_files)
    return "\n".join(lines) + "\n"


class SeatbeltContainmentBackend:
    """macOS Seatbelt jail (`sandbox-exec`). Native syscall-deny, no container."""

    name = "seatbelt"
    enforcement_tier = "native-syscall-deny"

    def available(self) -> tuple[bool, str]:
        if sys.platform != "darwin":
            return (False, "Seatbelt is macOS-only")
        if not Path("/usr/bin/sandbox-exec").exists():  # type: ignore[unreachable]  # darwin-only; unreachable under the linux mypy pin
            return (False, "/usr/bin/sandbox-exec not found")
        return (True, "sandbox-exec present")

    def profile_for(self, writable_roots: Sequence[str], *, allow_network: bool) -> str:
        return lower_to_seatbelt(
            writable_roots,
            allow_network=allow_network,
            deny_sockets=_terminal_server_dirs(),
            deny_socket_files=_named_sockets(),
        )

    def exposure(self) -> str | None:
        """What a confined command can still reach to run code outside the sandbox, or None."""
        parts = [
            f"{sock} is writable by this user: a confined command can drive the Docker daemon and escape the "
            "sandbox through it. The sandbox does not block it; stop the daemon, or run where the socket is "
            "not writable, to close this."
            for sock in _docker_sockets()
        ]
        host = os.environ.get("DOCKER_HOST", "")
        if host.startswith(("tcp://", "ssh://", "http://", "https://")):
            parts.append(
                f"DOCKER_HOST points at a Docker daemon over the network ({host.split('://')[0]}): a confined "
                "command that reaches that daemon can escape the sandbox through it. Unset DOCKER_HOST, or put "
                "the daemon where this user cannot reach it, to close this."
            )
        skipped = [
            *_skipped(_terminal_server_candidates(), stat.S_ISDIR),
            *_skipped(_named_socket_candidates(), stat.S_ISSOCK),
        ]
        if skipped:
            parts.append(
                f"{', '.join(skipped)} exist but are not a directory or socket this user owns (another user's, "
                "or a link to one), so the sandbox has no rule for them: a server listening there stays "
                "reachable. Remove them, or point the variable that names them elsewhere, to close this."
            )
        return " Also: ".join(parts) if parts else None

    def _attempt_write(self, profile: str, working_root: Any, target: Path) -> bool:
        """Attempt one create of `target` under `profile` (cwd=WORKDIR); True iff it landed.

        Uses `touch` with the path as a *literal argv element* — no shell, no SBPL/shell
        interpolation — so a workspace path containing quotes can neither malform the command
        nor inject. (`target` is an absolute realpath and `--` ends option parsing, so it
        cannot be read as an option.) Cleanup is parent-side (unsandboxed), reliable regardless
        of the profile.
        """
        with suppress(FileNotFoundError):
            target.unlink()
        self.launch(profile, working_root, ["/usr/bin/touch", "--", str(target)])
        landed = target.exists()
        with suppress(FileNotFoundError):
            target.unlink()
        return landed

    def probe(self, profile: str, working_root: Any, *, writable_roots: Sequence[str]) -> None:
        """Probe the jail fail-closed (§6): prove it is BOTH live AND grant-conformant.

        Before the body runs, verify confinement — not merely that *some* jail exists.
        Liveness alone is unsound: a profile that denies the parent dir but PERMITS a write
        outside the granted roots passes a liveness-only check yet violates the grant. So the
        probe is deny-closed and per-root:

          1. liveness   — an out-of-WORKSPACE write (parent dir) MUST be denied → a jail exists;
          2. per-root   — a write beneath EACH declared writable root MUST be allowed, else the
                          profile is too strict and a legit body's writes would spuriously fail;
          3. deny-closed — if WORKDIR itself is not covered by a writable root, an in-WORKDIR
                          write outside every root MUST be denied (the ReadOnly / unbound-path
                          case: a profile mis-lowered to a broader writable root is caught here).
        """
        wd = Path(os.path.realpath(str(working_root)))
        roots = [Path(os.path.realpath(str(root))) for root in writable_roots]

        # (1) liveness: a write OUTSIDE the workspace must be denied.
        if self._attempt_write(profile, working_root, wd.parent / ".jail-probe"):
            raise JailNotEstablished("fail-closed: out-of-WORKDIR probe write was NOT denied — no jail established")

        # (2) per-root: each declared writable root must actually accept writes.
        for root in roots:
            if not self._attempt_write(profile, working_root, root / ".jail-probe-canary"):
                raise JailNotEstablished(
                    f"fail-closed: writable root {root} DENIES writes (profile too strict) — "
                    "a legit body's writes would spuriously fail"
                )

        # (3) deny-closed: an in-WORKDIR path outside every writable root must be denied.
        wd_is_writable = any(wd == root or wd.is_relative_to(root) for root in roots)
        if not wd_is_writable and self._attempt_write(profile, working_root, wd / ".jail-probe-denied"):
            raise JailNotEstablished(
                "fail-closed: an in-WORKDIR path outside every writable root was PERMITTED "
                "(profile mis-lowered to a broader writable root) — would silently escalate"
            )

    def launch(
        self, profile: str, working_root: Any, command: list[str], *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """The launch (containment-and-carriers.md §5): `sandbox-exec -p <profile> <command>`."""
        return subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", profile, *command],
            cwd=os.path.realpath(str(working_root)),
            env=env if env is not None else dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
