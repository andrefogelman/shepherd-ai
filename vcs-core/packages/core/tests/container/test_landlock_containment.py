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
    if _user_bus_dir() is not None and shutil.which("bwrap") is not None:
        assert landlock.service_manager_exposure() is None
    monkeypatch.setattr(landlock, "_service_manager_dirs", lambda: ())
    monkeypatch.setattr(landlock, "_find_bwrap", lambda: None)
    assert landlock.service_manager_exposure() is None
    assert landlock._hiding_argv() == []


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
