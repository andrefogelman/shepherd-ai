# under-test: vcs_core._jail_exposure
"""What both jails report and cannot close: Docker daemons and ssh logins over sockets.

Platform-independent: nothing here confines anything. The ssh login runs a stub `ssh`
that prints what the real one prints, so no host's sshd is touched.
"""

from __future__ import annotations

import json
import socket
import stat
import sys
import threading
from pathlib import Path

import pytest
from vcs_core import _jail_exposure as exposure


@pytest.fixture(autouse=True)
def live_environment(monkeypatch):
    """Each test reads the environment it sets, not one a resolve_programs call kept earlier."""
    monkeypatch.setattr(exposure, "_ENVIRON", None, raising=False)


def test_dbus_addresses_split_paths_from_abstract_names_and_decode_them() -> None:
    value = "unix:path=/run/user/1000/bus;unix:abstract=/tmp/dbus-X,guid=1;tcp:host=h;unix:path=/a%20b%7ec"
    assert exposure.dbus_addresses(value) == (["/run/user/1000/bus", "/a b~c"], ["/tmp/dbus-X"])
    assert exposure.dbus_addresses("") == ([], [])


def _context(config: Path, name: str, host: str) -> None:
    meta = config / "contexts" / "meta" / name
    meta.mkdir(parents=True)
    (meta / "meta.json").write_text(json.dumps({"Name": name, "Endpoints": {"docker": {"Host": host}}}))


def test_docker_contexts_and_per_user_sockets_are_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "cfg"))
    monkeypatch.setenv("DOCKER_HOST", "unix:///from/env.sock")
    monkeypatch.setenv("HOME", str(tmp_path))
    _context(tmp_path / "cfg", "desktop-linux", "unix:///home/u/.docker/desktop/docker.sock")
    _context(tmp_path / "cfg", "remote", "ssh://u@far")
    (tmp_path / "cfg" / "contexts" / "meta" / "broken").mkdir()
    (tmp_path / "cfg" / "contexts" / "meta" / "broken" / "meta.json").write_text("{not json")
    candidates = exposure.docker_socket_candidates()
    assert candidates[:2] == ["/from/env.sock", "/home/u/.docker/desktop/docker.sock"]
    assert str(tmp_path / ".colima/default/docker.sock") in candidates
    assert not any("far" in c for c in candidates)


def test_docker_daemon_on_loopback_tcp_is_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "cfg"))
    _context(tmp_path / "cfg", "local", "tcp://127.0.0.1:2376")
    _context(tmp_path / "cfg", "far", "tcp://10.1.2.3:2376")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    monkeypatch.setattr(exposure, "_DOCKER_PLAIN_PORT", server.getsockname()[1])
    monkeypatch.setattr(exposure, "_LOOPBACK", ("127.0.0.1",))

    def answer() -> None:
        conn, _ = server.accept()
        with conn:
            conn.recv(512)
            conn.sendall(b"HTTP/1.0 200 OK\r\nApi-Version: 1.52\r\n\r\nOK")

    thread = threading.Thread(target=answer)
    thread.start()
    try:
        found = exposure.docker_network_endpoints()
    finally:
        thread.join(timeout=5)
        server.close()
    assert found == ["tcp://127.0.0.1:2376", f"tcp://127.0.0.1:{exposure._DOCKER_PLAIN_PORT}"]


def test_a_port_that_is_not_docker_is_not_reported() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def answer() -> None:
        conn, _ = server.accept()
        with conn:
            conn.recv(512)
            conn.sendall(b"HTTP/1.0 200 OK\r\nServer: other\r\n\r\n")

    thread = threading.Thread(target=answer)
    thread.start()
    try:
        assert exposure._answers_docker_api("127.0.0.1", port) is False
    finally:
        thread.join(timeout=5)
        server.close()


def test_sshd_endpoints_follow_port_listen_address_and_include(tmp_path) -> None:
    included = tmp_path / "d"
    included.mkdir()
    (included / "10.conf").write_text("Port 2222\nListenAddress 192.0.2.7\nListenAddress [::1]:2200\n")
    config = tmp_path / "sshd_config"
    config.write_text(
        f"# comment\nInclude {included}/*.conf\nListenAddress 0.0.0.0:2022\nMatch User x\nPort 9999\nListenAddress 10.9.9.9\n"
    )
    assert exposure.sshd_endpoints(str(config)) == [
        ("192.0.2.7", 2222),
        ("::1", 2200),
        ("127.0.0.1", 2022),
        ("::1", 2022),
    ]


def test_sshd_endpoints_default_to_loopback_port_22(tmp_path) -> None:
    assert exposure.sshd_endpoints(str(tmp_path / "absent")) == [("127.0.0.1", 22), ("::1", 22)]
    config = tmp_path / "sshd_config"
    config.write_text("Port 2222\nPasswordAuthentication no\n")
    assert exposure.sshd_endpoints(str(config)) == [("127.0.0.1", 2222), ("::1", 2222)]


def _stub_ssh(tmp_path: Path, stderr: str, then: str) -> tuple[str, Path]:
    """An `ssh` that records its argv, prints like the real one, then exits or waits (`-N`)."""
    log = tmp_path / "argv.json"
    stub = tmp_path / "ssh"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"sys.stderr.write({stderr!r}); sys.stderr.flush()\n"
        f"{then}\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return str(stub), log


@pytest.fixture
def ssh_host(tmp_path, monkeypatch):
    """One reachable endpoint and one readable host key; the login result is the stub's."""
    monkeypatch.setattr(exposure, "_SSH_LOGS_IN", None)
    monkeypatch.setattr(exposure, "_SSH_TIMEOUT_S", 5)
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "ssh_host_ed25519_key.pub").write_text("ssh-ed25519 AAAAhost root@h\n")
    monkeypatch.setattr(exposure, "_SSH_HOST_KEYS", str(keys / "ssh_host_*_key.pub"))
    monkeypatch.setattr(exposure, "sshd_endpoints", lambda: [("127.0.0.1", 2222)])
    monkeypatch.setattr(exposure.socket, "create_connection", lambda address, timeout: socket.socket())
    return tmp_path


def test_ssh_login_counts_when_ssh_says_it_authenticated(ssh_host, monkeypatch) -> None:
    stub, log = _stub_ssh(
        ssh_host, 'Authenticated to 127.0.0.1 ([127.0.0.1]:2222) using "publickey".\n', "time.sleep(60)"
    )
    monkeypatch.setattr(exposure.shutil, "which", lambda name: stub if name == "ssh" else None)
    assert exposure.ssh_logs_in() is True
    argv = json.loads(log.read_text().splitlines()[0])
    assert argv[0] == "-N"  # no session: nothing runs on the far side
    for option in (
        "BatchMode=yes",
        "StrictHostKeyChecking=yes",
        "HostKeyAlias=jail-exposure-loopback",
        "ControlMaster=no",
        "ControlPath=none",
        "ProxyCommand=none",
        "RemoteCommand=none",
        "PermitLocalCommand=no",
        "ForwardAgent=no",
        "ForwardX11=no",
        "ClearAllForwardings=yes",
    ):
        assert option in argv
    assert argv[-5:-2] == ["-p", "2222", "-l"]
    assert argv[-1] == "127.0.0.1"
    known = argv[argv.index("-o", argv.index("StrictHostKeyChecking=yes")) + 1]
    assert known.startswith("UserKnownHostsFile=")
    assert not Path(known.split("=", 1)[1]).exists()  # the pinned keys file is gone after the check
    assert exposure.ssh_logs_in() is True
    assert len(log.read_text().splitlines()) == 1  # once per process


def test_ssh_login_refused_is_not_reported(ssh_host, monkeypatch) -> None:
    stub, _log = _stub_ssh(ssh_host, "u@127.0.0.1: Permission denied (publickey,password).\n", "sys.exit(255)")
    monkeypatch.setattr(exposure.shutil, "which", lambda name: stub if name == "ssh" else None)
    assert exposure.ssh_logs_in() is False


def test_ssh_login_that_never_answers_is_stopped(ssh_host, monkeypatch) -> None:
    stub, _log = _stub_ssh(ssh_host, "", "time.sleep(60)")
    monkeypatch.setattr(exposure.shutil, "which", lambda name: stub if name == "ssh" else None)
    monkeypatch.setattr(exposure, "_SSH_TIMEOUT_S", 1)
    assert exposure.ssh_logs_in() is False


def test_ssh_login_is_not_tried_without_host_keys_or_a_listener(ssh_host, monkeypatch) -> None:
    stub, log = _stub_ssh(ssh_host, "Authenticated to x\n", "time.sleep(60)")
    monkeypatch.setattr(exposure.shutil, "which", lambda name: stub if name == "ssh" else None)
    monkeypatch.setattr(exposure, "_SSH_HOST_KEYS", str(ssh_host / "none" / "*.pub"))
    assert exposure.ssh_logs_in() is False  # cannot tell this host's sshd from another listener

    def refused(address, timeout):
        raise ConnectionRefusedError

    monkeypatch.setattr(exposure, "_SSH_LOGS_IN", None)
    monkeypatch.setattr(exposure, "_SSH_HOST_KEYS", str(ssh_host / "keys" / "ssh_host_*_key.pub"))
    monkeypatch.setattr(exposure.socket, "create_connection", refused)
    assert exposure.ssh_logs_in() is False
    assert not log.exists()


def test_ssh_login_is_not_tried_for_a_uid_without_an_account(ssh_host, monkeypatch) -> None:
    stub, log = _stub_ssh(ssh_host, "Authenticated to x\n", "time.sleep(60)")
    monkeypatch.setattr(exposure.shutil, "which", lambda name: stub if name == "ssh" else None)

    def unknown(uid):
        raise KeyError(uid)

    monkeypatch.setattr(exposure.pwd, "getpwuid", unknown)
    assert exposure.ssh_logs_in() is False
    assert not log.exists()


def test_ssh_message_names_the_jail() -> None:
    assert "inside the sandbox" in exposure.ssh_message("sandbox", "whenever the run allows network access")


def test_programs_resolved_first_ignore_a_path_set_later(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exposure, "_PROGRAMS", {})
    real = tmp_path / "real"
    planted = tmp_path / "planted"
    for directory in (real, planted):
        directory.mkdir()
        for name in ("bwrap", "ssh", "security"):
            tool = directory / name
            tool.write_text("#!/bin/sh\n")
            tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(real))
    assert exposure.program("ssh") == str(real / "ssh")  # nothing resolved yet: looked up now
    monkeypatch.setenv("PATH", str(planted))
    assert exposure.program("ssh") == str(planted / "ssh")
    monkeypatch.setenv("PATH", str(real))
    assert exposure.resolve_programs() == {
        "bwrap": str(real / "bwrap"),
        "ssh": str(real / "ssh"),
        "security": str(real / "security"),
    }
    monkeypatch.setenv("PATH", str(planted))  # a repository's PATH, set after
    assert exposure.program("bwrap") == str(real / "bwrap")
    assert exposure.program("ssh") == str(real / "ssh")
    assert exposure.program("security") == str(real / "security")
    assert exposure.resolve_programs()["bwrap"] == str(real / "bwrap")  # kept, not looked up again


def test_resolve_programs_keeps_the_environment_the_checks_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exposure, "_PROGRAMS", {})
    home = tmp_path / "home"
    planted = tmp_path / "planted"
    _context(home / ".docker", "desktop", f"unix://{home}/docker.sock")
    _context(planted, "planted", f"unix://{planted}/docker.sock")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user-docker.sock")
    exposure.resolve_programs()
    # a repository's settings, put in this process's environment after the call
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("DOCKER_CONFIG", str(planted))
    monkeypatch.delenv("DOCKER_HOST")
    assert exposure.home() == home
    assert exposure.environ()["DOCKER_HOST"] == "unix:///run/user-docker.sock"
    assert exposure.docker_context_hosts() == [f"unix://{home}/docker.sock"]
    candidates = exposure.docker_socket_candidates()
    assert "/run/user-docker.sock" in candidates
    assert str(home / "docker.sock") in candidates
    assert not any(c.startswith(str(planted)) for c in candidates)
    monkeypatch.setattr(exposure, "_ENVIRON", None)  # not resolved first: the live values
    assert exposure.docker_context_hosts() == [f"unix://{planted}/docker.sock"]
