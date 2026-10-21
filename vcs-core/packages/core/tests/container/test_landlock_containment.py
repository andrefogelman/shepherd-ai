# under-test: vcs_core._landlock_containment
"""B2b — Linux Landlock ContainmentBackend (real, container-verified).

Run via ``make test_container`` (privileged Podman, kernel >=5.13). These assert the real
Landlock syscall denial and the exit-code-aware fail-closed probe. Marked ``container`` +
Linux-gated: the default container seccomp profile can block the Landlock syscalls (444-446),
so they need the privileged image; on macOS they skip.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from pathlib import Path

import pytest
from vcs_core import _landlock_containment as landlock
from vcs_core._containment import ContainmentBackend, JailNotEstablished
from vcs_core._landlock_containment import LandlockContainmentBackend, landlock_abi

pytestmark = [
    pytest.mark.container,
    pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only"),
]


def test_landlock_available_and_protocol_conformant() -> None:
    backend = LandlockContainmentBackend()
    assert isinstance(backend, ContainmentBackend)
    ok, why = backend.available()
    assert ok, why
    assert backend.name == "landlock"
    assert backend.enforcement_tier == "native-syscall-deny"
    assert landlock_abi() >= 1


def test_landlock_profile_for_readonly_vs_permissive(tmp_path) -> None:
    backend = LandlockContainmentBackend()
    wd = os.path.realpath(str(tmp_path))
    assert backend.profile_for((str(tmp_path),), allow_network=True) == json.dumps([wd])
    assert backend.profile_for((), allow_network=False) == json.dumps([])  # no writable root
    # multi-root: one entry per granted root (Landlock is FS-only; allow_network is ignored).
    backend_dir = os.path.realpath(str(tmp_path / "backend"))
    assert backend.profile_for((str(tmp_path / "backend"),), allow_network=False) == json.dumps([backend_dir])


def test_landlock_permissive_allows_in_workdir_denies_outside(tmp_path) -> None:
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)

    inside = tmp_path / "ok.txt"
    assert backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(inside)]).returncode == 0
    assert inside.exists()  # in-WORKDIR write allowed

    outside = tmp_path.parent / "escape.txt"
    outside.unlink(missing_ok=True)
    assert backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(outside)]).returncode != 0
    assert not outside.exists()  # out-of-WORKDIR write denied at the syscall


def test_landlock_readonly_denies_in_workdir(tmp_path) -> None:
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((), allow_network=False)
    target = tmp_path / "nope.txt"
    assert backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(target)]).returncode != 0
    assert not target.exists()


def test_landlock_per_binding_writable_root_is_deny_closed(tmp_path) -> None:
    """v0.2: a single ReadWrite root (``backend/``) beside an ungranted sibling
    (``docs/``). A write under the granted root lands; the sibling and any unbound
    in-workspace path are refused at the syscall — the deny-closed guarantee."""
    backend = LandlockContainmentBackend()
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    profile = backend.profile_for((str(backend_dir),), allow_network=False)

    ok = backend_dir / "candidate.txt"
    assert backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(ok)]).returncode == 0
    assert ok.exists()

    denied = docs_dir / "nope.txt"
    assert backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(denied)]).returncode != 0
    assert not denied.exists()

    unbound = tmp_path / "stray.txt"
    assert backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(unbound)]).returncode != 0
    assert not unbound.exists()

    backend.probe(profile, tmp_path, writable_roots=(str(backend_dir),))


def test_landlock_probe_passes_for_conformant_profiles(tmp_path) -> None:
    backend = LandlockContainmentBackend()
    backend.probe(backend.profile_for((str(tmp_path),), allow_network=True), tmp_path, writable_roots=(str(tmp_path),))
    backend.probe(backend.profile_for((), allow_network=False), tmp_path, writable_roots=())


def test_landlock_probe_catches_readonly_mislowered_to_writable(tmp_path) -> None:
    backend = LandlockContainmentBackend()
    # BUG SIM: a ReadOnly run handed a Permissive-style writable-root profile must be caught
    # by the deny-closed canary (in-WORKDIR write would otherwise be permitted).
    mislowered = backend.profile_for((str(tmp_path),), allow_network=True)
    with pytest.raises(JailNotEstablished):
        backend.probe(mislowered, tmp_path, writable_roots=())


def test_landlock_allows_dev_null_and_nothing_else_under_dev(tmp_path) -> None:
    """Shells discard output into /dev/null; denying it made every `cmd >/dev/null` exit 1.
    Only that device is granted: /dev/shm, writable to anyone outside the jail, stays denied."""
    backend = LandlockContainmentBackend()
    permissive = backend.profile_for((str(tmp_path),), allow_network=True)
    readonly = backend.profile_for((), allow_network=False)
    for profile in (permissive, readonly):
        done = backend.launch(profile, tmp_path, ["/bin/sh", "-c", "echo hi > /dev/null && echo ok 2>/dev/null"])
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "ok"
    if Path("/dev/shm").is_dir():
        target = Path(f"/dev/shm/landlock-probe-{os.getpid()}")
        try:
            assert backend.launch(permissive, tmp_path, ["/usr/bin/touch", "--", str(target)]).returncode != 0
            assert not target.exists()
        finally:
            target.unlink(missing_ok=True)


def _user_bus_dir() -> str | None:
    """The user's runtime directory when it holds a socket of the service manager."""
    for candidate in (os.environ.get("XDG_RUNTIME_DIR", ""), f"/run/user/{os.getuid()}"):
        if candidate and any((Path(candidate) / s).exists() for s in ("bus", "systemd/private")):
            return candidate
    return None


def test_landlock_hides_the_user_service_manager_sockets(tmp_path) -> None:
    """Landlock does not govern connecting to a socket by path. Inside the jail the user's
    runtime directory, where the session bus and systemd's private socket live, is empty."""
    rundir = _user_bus_dir()
    if rundir is None:
        pytest.skip("no user service-manager socket on this host")
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    done = backend.launch(
        profile, tmp_path, [sys.executable, "-c", "import os, sys; print(sorted(os.listdir(sys.argv[1])))", rundir]
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_landlock_blocks_a_write_through_systemd_run(tmp_path) -> None:
    """`systemd-run --user` asks the user's service manager to start the command, so the
    command is a child of systemd, outside the jail, and its write outside the roots lands."""
    rundir = _user_bus_dir()
    if rundir is None or shutil.which("systemd-run") is None:
        pytest.skip("no user service manager reachable on this host")
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    outside = tmp_path.parent / f"bus-escape-{os.getpid()}.txt"
    outside.unlink(missing_ok=True)
    try:
        backend.launch(
            profile,
            tmp_path,
            # The variable is set here because a confined command can set it itself.
            [
                "/usr/bin/env",
                f"XDG_RUNTIME_DIR={rundir}",
                "systemd-run",
                "--user",
                "--quiet",
                "--wait",
                "--collect",
                "--",
                "/usr/bin/touch",
                "--",
                str(outside),
            ],
        )
        assert not outside.exists()
    finally:
        outside.unlink(missing_ok=True)


@pytest.fixture
def fresh_hiding(monkeypatch):
    """Each test decides bubblewrap's fate afresh and leaves no outer prefix behind."""
    monkeypatch.setattr(landlock, "_HIDING_WORKS", {})
    monkeypatch.setattr(landlock, "_SOCKETS_UNDER", {})
    monkeypatch.setattr(landlock, "_KEPT_WALKED", set())
    yield
    landlock.set_outer_prefix([])


def test_landlock_runs_with_a_warning_when_bwrap_is_missing(tmp_path, monkeypatch, fresh_hiding) -> None:
    """Without bubblewrap the command still runs; the user is told the door is open and how to close it."""
    if _user_bus_dir() is None:
        pytest.skip("no user service-manager socket on this host")
    monkeypatch.setattr(landlock, "_find_bwrap", lambda: None)
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    backend.probe(profile, tmp_path, writable_roots=(str(tmp_path),))
    with pytest.warns(RuntimeWarning, match="Install bubblewrap"):
        done = backend.launch(profile, tmp_path, ["/bin/sh", "-c", "echo ran"])
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ran"
    message = landlock.service_manager_exposure()
    assert message is not None
    assert "not installed" in message
    assert "sudo apt install bubblewrap" in message


def test_landlock_runs_with_a_warning_when_bwrap_cannot_start(tmp_path, monkeypatch, fresh_hiding) -> None:
    """A bubblewrap that fails (e.g. user namespaces denied) degrades the same way, with its error."""
    if _user_bus_dir() is None:
        pytest.skip("no user service-manager socket on this host")
    monkeypatch.setattr(landlock, "_find_bwrap", lambda: "/bin/false")
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    backend.probe(profile, tmp_path, writable_roots=(str(tmp_path),))
    with pytest.warns(RuntimeWarning, match="could not hide"):
        done = backend.launch(profile, tmp_path, ["/bin/sh", "-c", "echo ran"])
    assert done.returncode == 0, done.stderr


def test_landlock_reports_no_exposure_when_hidden_or_absent(monkeypatch, fresh_hiding) -> None:
    monkeypatch.setattr(landlock, "_docker_sockets", lambda hidden: ())
    monkeypatch.setattr(landlock, "_x_displays", lambda: ())
    monkeypatch.setattr(landlock, "_loopback_ssh", lambda: False)
    monkeypatch.setattr(landlock, "_skipped", lambda candidates, is_kind: [])
    if _user_bus_dir() is not None and shutil.which("bwrap") is not None:
        assert landlock.exposure() is None
    monkeypatch.setattr(landlock, "_hidden_dirs", lambda: ())
    monkeypatch.setattr(landlock, "_find_bwrap", lambda: None)
    assert landlock.exposure() is None
    assert landlock.service_manager_exposure() is None
    assert landlock._hiding_argv() == []


def _needs_bwrap() -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed")


def test_landlock_hides_the_tmux_socket_directory(tmp_path, monkeypatch, fresh_hiding) -> None:
    """`tmux new-window cmd` runs cmd under the tmux server, outside the jail, like systemd-run."""
    _needs_bwrap()
    tmux_tmp = tmp_path / "tmuxtmp"
    tmux_dir = tmux_tmp / f"tmux-{os.getuid()}"
    tmux_dir.mkdir(parents=True)
    (tmux_dir / "default").write_text("")
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    work = tmp_path / "work"
    work.mkdir()
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(work),), allow_network=True)
    done = backend.launch(
        profile, work, [sys.executable, "-c", "import os, sys; print(os.listdir(sys.argv[1]))", str(tmux_dir)]
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_landlock_keeps_a_workspace_that_sits_under_a_hidden_directory(tmp_path, monkeypatch, fresh_hiding) -> None:
    """A writable root under a hidden directory is bound back over the tmpfs: its writes
    land on the host instead of vanishing into the mask, and siblings stay hidden."""
    _needs_bwrap()
    tmux_tmp = tmp_path / "tmuxtmp"
    hidden = tmux_tmp / f"tmux-{os.getuid()}"
    work = hidden / "work"
    work.mkdir(parents=True)
    (hidden / "default").write_text("")
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(work),), allow_network=True)
    backend.probe(profile, work, writable_roots=(str(work),))
    done = backend.launch(
        profile,
        work,
        ["/bin/sh", "-c", 'touch -- "$1/made" && ls -A -- "$2"', "sh", str(work), str(hidden)],
    )
    assert done.returncode == 0, done.stderr
    assert (work / "made").exists()
    assert done.stdout.split() == ["work"]  # the socket beside it is not visible


def test_landlock_warns_with_its_own_category(tmp_path, monkeypatch, fresh_hiding) -> None:
    """Callers that already showed `exposure()` filter the per-launch repeats by category, not by text."""
    if _user_bus_dir() is None:
        pytest.skip("no user service-manager socket on this host")
    monkeypatch.setattr(landlock, "_find_bwrap", lambda: None)
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    with pytest.warns(landlock.JailExposureWarning):
        backend.launch(profile, tmp_path, ["/bin/true"])
    assert issubclass(landlock.JailExposureWarning, RuntimeWarning)


def test_landlock_bwrap_trial_that_hangs_counts_as_failed(tmp_path, monkeypatch, fresh_hiding) -> None:
    """A trial bubblewrap that never returns must not hold the caller; it degrades like a failed one."""
    if _user_bus_dir() is None:
        pytest.skip("no user service-manager socket on this host")
    hang = tmp_path / "bwrap"
    hang.write_text("#!/bin/sh\nexec sleep 30\n")
    hang.chmod(0o755)
    monkeypatch.setattr(landlock, "_find_bwrap", lambda: str(hang))
    monkeypatch.setattr(landlock, "_TRIAL_TIMEOUT_S", 1)
    message = landlock.exposure()
    assert message is not None
    assert "did not finish a trial run within 1 s" in message


def test_landlock_reports_a_writable_docker_socket(tmp_path, monkeypatch, fresh_hiding) -> None:
    """The Docker daemon runs as root; a socket the user can write to is a way out the jail cannot hide."""
    sock_path = tmp_path / "docker.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(sock_path))
        monkeypatch.setenv("DOCKER_HOST", f"unix://{sock_path}")
        monkeypatch.setattr(landlock, "_hidden_dirs", lambda: ())
        message = landlock.exposure()
        assert message is not None
        assert str(sock_path) in message
        assert "Docker daemon" in message
    finally:
        server.close()


def test_landlock_probe_runs_without_the_outer_prefix(tmp_path, fresh_hiding) -> None:
    """The probe only touches files; a wrapper failing there would read as a denied write."""
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    landlock.set_outer_prefix(["/bin/false"])
    backend.probe(profile, tmp_path, writable_roots=(str(tmp_path),))
    assert backend.launch(profile, tmp_path, ["/bin/true"]).returncode != 0  # the body still gets it


def test_landlock_outer_prefix_runs_outside_the_jail(tmp_path, fresh_hiding) -> None:
    """The prefix wraps the jail: it can write where the confined command cannot."""
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(tmp_path),), allow_network=True)
    outside = tmp_path.parent / f"prefix-mark-{os.getpid()}.txt"
    outside.unlink(missing_ok=True)
    landlock.set_outer_prefix(["/bin/sh", "-c", 'touch -- "$0" && exec "$@"', str(outside)])
    assert landlock.outer_prefix() == ("/bin/sh", "-c", 'touch -- "$0" && exec "$@"', str(outside))
    try:
        done = backend.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(tmp_path.parent / "denied.txt")])
        assert outside.exists()  # the prefix ran unconfined
        assert done.returncode != 0  # the command after it is still jailed
        assert not (tmp_path.parent / "denied.txt").exists()
    finally:
        outside.unlink(missing_ok=True)
    landlock.set_outer_prefix([])
    assert landlock.outer_prefix() == ()


def test_landlock_hides_the_runtime_dir_without_a_bus(tmp_path, monkeypatch) -> None:
    """Rootless Docker and Podman listen in the runtime directory too, and a service manager
    may start after the launch: the directory is hidden whether or not a bus is there."""
    rundir = tmp_path / "run"
    rundir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(rundir))
    assert os.path.realpath(rundir) in landlock._service_manager_dirs()


def test_landlock_masks_a_session_bus_named_elsewhere_and_not_its_directory(
    tmp_path, monkeypatch, fresh_hiding
) -> None:
    """Where no systemd runs, the bus is `unix:path=/tmp/dbus-XXXX`: hiding its directory
    would empty /tmp for the command, so only the socket gets /dev/null over it."""
    _needs_bwrap()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "neighbour").write_text("x")
    bus = shared / "dbus-test"
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", f"unix:guid=abc,path={bus};tcp:host=localhost")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(bus))
        server.listen(1)
        assert os.path.realpath(shared) not in landlock._hidden_dirs()
        assert landlock._named_sockets() == (os.path.realpath(bus),)
        work = tmp_path / "work"
        work.mkdir()
        backend = LandlockContainmentBackend()
        profile = backend.profile_for((str(work),), allow_network=True)
        backend.probe(profile, work, writable_roots=(str(work),))
        done = backend.launch(
            profile,
            work,
            [
                sys.executable,
                "-c",
                "import os, stat, sys; d = sys.argv[1]\n"
                "print(sorted(os.listdir(d)), stat.S_ISSOCK(os.lstat(os.path.join(d, 'dbus-test')).st_mode))",
                str(shared),
            ],
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "['dbus-test', 'neighbour'] False"
    finally:
        server.close()


def test_landlock_hides_both_tmux_directories_and_masks_the_socket_tmux_names(tmp_path, monkeypatch) -> None:
    """tmux uses $TMUX_TMPDIR when set and /tmp otherwise; the caller's server may use
    either. The socket $TMUX names is masked alone, since its directory may be shared."""
    tmux_tmp = tmp_path / "tmuxtmp"
    tmux_tmp.mkdir()
    current = tmp_path / "current"
    current.mkdir()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(current / "default"))
        monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
        monkeypatch.setenv("TMUX", f"{current}/default,1234,0")
        candidates = landlock._terminal_server_candidates()
        assert f"/tmp/tmux-{os.getuid()}" in candidates
        assert str(tmux_tmp / f"tmux-{os.getuid()}") in candidates
        assert str(current) not in candidates
        assert os.path.realpath(current / "default") in landlock._named_sockets()
    finally:
        server.close()


def test_landlock_creates_an_absent_tmux_directory_private(tmp_path, monkeypatch) -> None:
    """A tmux server started during the launch puts its socket under the mask, not beside it."""
    tmux_tmp = tmp_path / "tmuxtmp"
    tmux_tmp.mkdir()
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    made = tmux_tmp / f"tmux-{os.getuid()}"
    assert str(made) in landlock._terminal_server_dirs()
    assert made.is_dir()
    assert made.stat().st_mode & 0o777 == 0o700


def test_landlock_leaves_a_parent_it_cannot_write(tmp_path) -> None:
    landlock._claim_socket_dirs([str(tmp_path / "absent-parent" / "tmux-1")])
    assert not (tmp_path / "absent-parent").exists()


def _as_another_user(monkeypatch) -> int:
    """Make the backend see a uid other than the one that owns this test's files, so the
    files stand in for another user's (chown needs root)."""
    other = os.getuid() + 1
    monkeypatch.setattr(landlock.os, "getuid", lambda: other)
    monkeypatch.setattr(landlock, "_claim_socket_dirs", lambda candidates: None)
    return other


def test_landlock_does_not_follow_another_users_link_at_the_tmux_directory(tmp_path, monkeypatch) -> None:
    """/tmp is shared: another user can put a link at /tmp/tmux-<uid>. Following it would
    hide whatever it points at, so neither the link nor its target is masked."""
    tmux_tmp = tmp_path / "tmuxtmp"
    tmux_tmp.mkdir()
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    # control: this user's own directory there is hidden
    real = tmux_tmp / f"tmux-{os.getuid()}"
    real.mkdir()
    assert str(real) in landlock._terminal_server_dirs()
    other = _as_another_user(monkeypatch)
    target = tmp_path / "somewhere"
    target.mkdir()
    (tmux_tmp / f"tmux-{other}").symlink_to(target)
    dirs = landlock._terminal_server_dirs()
    assert str(tmux_tmp / f"tmux-{other}") not in dirs
    assert str(target) not in dirs
    assert list(target.iterdir()) == []


def test_landlock_follows_this_users_own_link_to_its_own_directory(tmp_path, monkeypatch) -> None:
    """A runtime directory linked elsewhere by this user (some container and WSL setups)
    is still hidden, at the link's target."""
    target = tmp_path / "real-runtime"
    target.mkdir()
    link = tmp_path / "runtime"
    link.symlink_to(target)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(link))
    assert str(target) in landlock._service_manager_dirs()


def test_landlock_skips_a_directory_another_user_owns(tmp_path, monkeypatch) -> None:
    """tmux and screen refuse a socket directory they do not own; so does the mask."""
    mine = tmp_path / "mine"
    mine.mkdir()
    assert landlock._own_dirs([str(mine)]) == (str(mine),)
    _as_another_user(monkeypatch)
    assert landlock._own_dirs([str(mine)]) == ()


def test_landlock_reports_what_it_leaves_unmasked(tmp_path, monkeypatch, fresh_hiding) -> None:
    """A skipped candidate is no longer silent: exposure() names it."""
    tmux_tmp = tmp_path / "tmuxtmp"
    tmux_tmp.mkdir()
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr(landlock, "_docker_sockets", lambda hidden: ())
    monkeypatch.setattr(landlock, "_x_displays", lambda: ())
    monkeypatch.setattr(landlock, "_loopback_ssh", lambda: False)
    monkeypatch.setattr(landlock, "_why_not_hidden", lambda dirs: None)
    other = _as_another_user(monkeypatch)
    planted = tmux_tmp / f"tmux-{other}"
    assert planted.name not in (landlock.exposure() or "")
    planted.symlink_to(tmp_path)
    assert str(planted) in (landlock.exposure() or "")


def test_landlock_does_not_follow_another_users_link_named_as_the_tmux_socket(tmp_path, monkeypatch) -> None:
    current = tmp_path / "current"
    current.mkdir()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(tmp_path / "real"))
        (current / "default").symlink_to(tmp_path / "real")
        monkeypatch.setenv("TMUX", f"{current}/default,1234,0")
        # control: this user's own link to its own socket is followed
        assert landlock._named_sockets() == (str(tmp_path / "real"),)
        _as_another_user(monkeypatch)
        assert landlock._named_sockets() == ()
    finally:
        server.close()


def test_landlock_walks_each_kept_path_once_while_it_holds(tmp_path, monkeypatch) -> None:
    """A workspace under a hidden directory can be a large tree; launches after the first
    reuse the walk until an entry right under it changes."""
    monkeypatch.setattr(landlock, "_SOCKETS_UNDER", {})
    walks: list[str] = []
    real_walk = os.walk

    def counted(path, *args, **kwargs):
        walks.append(str(path))
        return real_walk(path, *args, **kwargs)

    monkeypatch.setattr(landlock.os, "walk", counted)
    first_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    second_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        first_sock.bind(str(tmp_path / "s"))
        first = landlock._sockets_under(str(tmp_path))
        again = landlock._sockets_under(str(tmp_path))
        assert first == again == (str(tmp_path / "s"),)
        assert walks == [str(tmp_path)]
        second_sock.bind(str(tmp_path / "t"))
        os.utime(tmp_path, ns=(1, 1))  # the mtime must move even within one clock tick
        assert set(landlock._sockets_under(str(tmp_path))) == {str(tmp_path / "s"), str(tmp_path / "t")}
        assert walks == [str(tmp_path), str(tmp_path)]
    finally:
        first_sock.close()
        second_sock.close()


def test_landlock_drops_a_listed_socket_that_is_gone(tmp_path, monkeypatch, fresh_hiding) -> None:
    """bubblewrap creates a file where a mount destination is missing, so a socket removed
    since the walk must not stay in the argv: it would leave an empty file in the workspace."""
    kept = tmp_path / "kept"
    (kept / "sub").mkdir(parents=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(kept / "sub" / "s"))
    server.close()
    assert landlock._sockets_under(str(kept)) == (str(kept / "sub" / "s"),)
    (kept / "sub" / "s").unlink()
    (kept / "sub").rmdir()
    stamp = kept.stat().st_mtime_ns
    monkeypatch.setitem(landlock._SOCKETS_UNDER, str(kept), (stamp, (str(kept / "sub" / "s"),)))
    assert landlock._sockets_under(str(kept)) == ()
    argv = landlock._bwrap_argv("bwrap", (str(tmp_path),), (str(kept),))
    assert str(kept / "sub" / "s") not in argv


def test_landlock_probe_walks_kept_paths_the_listing_missed(tmp_path, monkeypatch, fresh_hiding) -> None:
    """The first probe walks the kept paths inside the jail itself, so a socket the walk
    outside did not list is caught and the run degrades with the warning."""
    _needs_bwrap()
    tmux_tmp = tmp_path / "tmuxtmp"
    hidden = tmux_tmp / f"tmux-{os.getuid()}"
    work = hidden / "work"
    work.mkdir(parents=True)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    monkeypatch.setattr(landlock, "_docker_sockets", lambda hidden: ())
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(work / "s"))
        monkeypatch.setitem(landlock._SOCKETS_UNDER, str(work), (work.stat().st_mtime_ns, ()))
        backend = LandlockContainmentBackend()
        profile = backend.profile_for((str(work),), allow_network=True)
        with pytest.warns(landlock.JailExposureWarning, match="is still a socket"):
            backend.probe(profile, work, writable_roots=(str(work),))
    finally:
        server.close()


def test_landlock_probe_walks_kept_paths_once_per_process(tmp_path, monkeypatch, fresh_hiding) -> None:
    tmux_tmp = tmp_path / "tmuxtmp"
    hidden = tmux_tmp / f"tmux-{os.getuid()}"
    work = hidden / "work"
    work.mkdir(parents=True)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    assert json.loads(landlock._hidden_dirs_spec((str(hidden),), (str(work),), walk_kept=True))["kept"] == [str(work)]
    assert json.loads(landlock._hidden_dirs_spec((str(hidden),), (str(work),)))["kept"] == []
    _needs_bwrap()
    seen: list[list[str]] = []
    backend = LandlockContainmentBackend()
    real_launch = backend.launch

    def recording(profile, root, command, **kwargs):
        if command[1:2] == ["-c"] and command[2] == landlock._HIDDEN_DIRS_CHECK:
            seen.append(json.loads(command[3])["kept"])
        return real_launch(profile, root, command, **kwargs)

    monkeypatch.setattr(backend, "launch", recording)
    profile = backend.profile_for((str(work),), allow_network=True)
    backend.probe(profile, work, writable_roots=(str(work),))
    backend.probe(profile, work, writable_roots=(str(work),))
    assert seen == [[str(work)], []]


def test_landlock_jail_has_its_own_pid_namespace(tmp_path, monkeypatch, fresh_hiding) -> None:
    """With a setuid bwrap, /proc/<pid>/root of a process outside would show the hidden
    directories; in the jail's own pid namespace those processes are not there."""
    _needs_bwrap()
    tmux_tmp = tmp_path / "tmuxtmp"
    (tmux_tmp / f"tmux-{os.getuid()}").mkdir(parents=True)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    work = tmp_path / "work"
    work.mkdir()
    assert "--unshare-pid" in landlock._bwrap_argv("bwrap", (str(tmux_tmp),))
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(work),), allow_network=True)
    done = backend.launch(
        profile,
        work,
        [sys.executable, "-c", "import os, sys; print(os.path.exists('/proc/' + sys.argv[1]))", str(os.getpid())],
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_landlock_reports_a_reachable_x_display(tmp_path, monkeypatch, fresh_hiding) -> None:
    monkeypatch.setattr(landlock, "_X11_DIR", str(tmp_path / "x11"))
    monkeypatch.delenv("DISPLAY", raising=False)
    assert landlock._x_displays() == ()
    (tmp_path / "x11").mkdir()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(tmp_path / "x11" / "X0"))
        assert landlock._x_displays() == (str(tmp_path / "x11" / "X0"),)
    finally:
        server.close()
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    assert "DISPLAY=localhost:10.0" in landlock._x_displays()
    monkeypatch.setattr(landlock, "_docker_sockets", lambda hidden: ())
    monkeypatch.setattr(landlock, "_loopback_ssh", lambda: False)
    assert "keystrokes" in (landlock.exposure() or "")


def test_landlock_reports_ssh_to_loopback_only_with_a_key_that_logs_in(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(landlock.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(landlock.socket, "create_connection", lambda address, timeout: socket.socket())
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_ed25519").write_text("")
    assert landlock._loopback_ssh() is False  # no authorized_keys: the key does not log in here
    (ssh / "authorized_keys").write_text("")
    assert landlock._loopback_ssh() is True

    def refused(address, timeout):
        raise ConnectionRefusedError

    monkeypatch.setattr(landlock.socket, "create_connection", refused)
    assert landlock._loopback_ssh() is False  # no sshd on loopback


def test_landlock_masks_a_socket_inside_a_bound_back_path(tmp_path, monkeypatch, fresh_hiding) -> None:
    """A kept path under a hidden directory comes back with its contents; a socket in it
    gets /dev/null over it, so the bus under a kept `systemd` cannot be reached."""
    _needs_bwrap()
    tmux_tmp = tmp_path / "tmuxtmp"
    hidden = tmux_tmp / f"tmux-{os.getuid()}"
    work = hidden / "work"
    work.mkdir(parents=True)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock_path = work / "s"
    try:
        server.bind(str(sock_path))
        server.listen(1)
        backend = LandlockContainmentBackend()
        profile = backend.profile_for((str(work),), allow_network=True)
        backend.probe(profile, work, writable_roots=(str(work),))
        done = backend.launch(
            profile,
            work,
            [
                sys.executable,
                "-c",
                "import os, socket, stat, sys\n"
                "print(stat.S_ISSOCK(os.lstat(sys.argv[1]).st_mode))\n"
                "c = socket.socket(socket.AF_UNIX)\n"
                "try:\n    c.connect(sys.argv[1]); print('connected')\n"
                "except OSError as e:\n    print('refused', e.errno)\n",
                str(sock_path),
            ],
        )
        assert done.returncode == 0, done.stderr
        lines = done.stdout.split("\n")
        assert lines[0] == "False"
        assert lines[1].startswith("refused")
    finally:
        server.close()


def test_landlock_probe_degrades_when_the_hiding_check_fails(tmp_path, monkeypatch, fresh_hiding) -> None:
    """A check that finds something still reachable warns and lets the run go on without bubblewrap."""
    _needs_bwrap()
    tmux_tmp = tmp_path / "tmuxtmp"
    (tmux_tmp / f"tmux-{os.getuid()}").mkdir(parents=True)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmux_tmp))
    monkeypatch.setattr(landlock, "_docker_sockets", lambda hidden: ())
    work = tmp_path / "work"
    work.mkdir()
    dirs = landlock._hidden_dirs()
    assert landlock._why_not_hidden(dirs) is None  # bubblewrap works here
    monkeypatch.setattr(landlock, "_HIDDEN_DIRS_CHECK", "import sys; print('socket left'); sys.exit(1)")
    backend = LandlockContainmentBackend()
    profile = backend.profile_for((str(work),), allow_network=True)
    with pytest.warns(landlock.JailExposureWarning, match="socket left"):
        backend.probe(profile, work, writable_roots=(str(work),))
    with pytest.warns(landlock.JailExposureWarning):
        assert landlock._hiding_argv((str(work),)) == []
    done = backend.launch(profile, work, ["/bin/sh", "-c", "echo ran"])
    assert done.stdout.strip() == "ran"


def test_landlock_reports_docker_host_over_the_network(monkeypatch, fresh_hiding) -> None:
    monkeypatch.setattr(landlock, "_hidden_dirs", lambda: ())
    monkeypatch.setattr(landlock, "_docker_sockets", lambda hidden: ())
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")
    message = landlock.exposure()
    assert message is not None
    assert "over the network (tcp)" in message
    monkeypatch.setenv("DOCKER_HOST", "ssh://user@host")
    assert "over the network (ssh)" in (landlock.exposure() or "")
