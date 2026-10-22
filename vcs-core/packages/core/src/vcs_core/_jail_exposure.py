"""Ways out of a jail that run over a socket: daemons and logins this user can reach.

Shared by the Landlock and Seatbelt backends. Neither governs connecting to a socket
(Landlock below ABI 6 does not; the Seatbelt profile leaves outbound connects open for
name resolution), so a daemon that runs commands for this user and accepts its
connection is a way out of both. The helpers here find the ones that can be found
without root, for the backends to report; nothing here blocks a launch.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import selectors
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Where per-user Docker daemons put their socket: Docker Desktop (macOS, then Linux),
#: Colima (profile `default`, older releases), OrbStack, Rancher Desktop.
_USER_DOCKER_SOCKETS = (
    ".docker/run/docker.sock",
    ".docker/desktop/docker.sock",
    ".colima/default/docker.sock",
    ".colima/docker.sock",
    ".orbstack/run/docker.sock",
    ".rd/docker.sock",
)

#: The Docker API port without TLS. A daemon there answers any local process, no credential.
_DOCKER_PLAIN_PORT = 2375

_LOOPBACK = ("127.0.0.1", "::1")

_SSHD_CONFIG = "/etc/ssh/sshd_config"
_SSH_HOST_KEYS = "/etc/ssh/ssh_host_*_key.pub"
#: The name the pinned host keys are filed under, whatever address or port is tried.
_HOST_KEY_ALIAS = "jail-exposure-loopback"
#: ssh prints the first at LogLevel VERBOSE once authentication succeeds (OpenSSH 7.x and
#: later); the second is the debug1 line of older releases.
_AUTHENTICATED = re.compile(rb"Authenticated to |Authentication succeeded")
_SSH_TIMEOUT_S = 10

# The login is tried once per process: exposure is computed again on launches where
# hiding fails, and each try is a line in the host's auth log.
_SSH_LOGS_IN: bool | None = None

#: The programs the jail runs by name, and their paths once `resolve_programs` found them.
_PROGRAM_NAMES = ("bwrap", "ssh", "security")
_PROGRAMS: dict[str, str | None] = {}

#: The environment as `resolve_programs` found it; until then the checks read the live one.
_ENVIRON: dict[str, str] | None = None


def resolve_programs() -> dict[str, str | None]:
    """Find bubblewrap, ssh and security on the current PATH now, and use those paths for the process.

    A caller that is about to put someone else's environment in this process's (a
    repository's settings for the jailed command, say) calls this first, so the PATH it
    sets chooses the tools of the jailed command and not the binaries that hide sockets
    from the jail or check what it leaves in reach. The environment itself is kept too:
    which sockets to hide and what to report are read from it (`environ`), so the caller
    may then unset the command servers' addresses for the jailed command without the jail
    losing track of them. Without this call each use reads the live values when it runs.
    """
    global _ENVIRON
    if _ENVIRON is None:
        _ENVIRON = dict(os.environ)
    for name in _PROGRAM_NAMES:
        if name not in _PROGRAMS:
            _PROGRAMS[name] = shutil.which(name, path=_ENVIRON.get("PATH"))
    return dict(_PROGRAMS)


def environ() -> Mapping[str, str]:
    """The environment `resolve_programs` kept, else the live one."""
    return _ENVIRON if _ENVIRON is not None else os.environ


def home() -> Path:
    """The user's home in `environ`, as `Path.home` reads it from the live one."""
    value = environ().get("HOME")
    return Path(value) if value else Path.home()


def program(name: str) -> str | None:
    """The path `resolve_programs` found for ``name``, else a lookup on the current PATH."""
    if name in _PROGRAMS:
        return _PROGRAMS[name]
    return shutil.which(name)


def dbus_addresses(value: str) -> tuple[list[str], list[str]]:
    r"""The ``path=`` sockets and the ``abstract=`` names in a D-Bus address list.

    Values are percent-decoded: the D-Bus spec escapes every byte outside
    ``[-0-9A-Za-z_/.\\*]``, so a path with a space or a ``~`` arrives as ``%20`` or ``%7e``.
    """
    paths: list[str] = []
    abstract: list[str] = []
    for address in value.split(";"):
        if not address.startswith("unix:"):
            continue
        for field in address[len("unix:") :].split(","):
            key, _, raw = field.partition("=")
            if key == "path":
                paths.append(unquote(raw))
            elif key == "abstract":
                abstract.append(unquote(raw))
    return paths, abstract


def _docker_config_dir() -> Path:
    return Path(environ().get("DOCKER_CONFIG") or home() / ".docker")


def docker_context_hosts() -> list[str]:
    """The endpoint of every Docker context of this user, as a Host URL.

    `docker --context NAME` reaches any of them, so all count, not only the current one.
    Docker Desktop for Linux points its context at a socket in the user's home, with no
    DOCKER_HOST set.
    """
    hosts: list[str] = []
    for meta in sorted((_docker_config_dir() / "contexts" / "meta").glob("*/meta.json")):
        try:
            data = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        endpoints = data.get("Endpoints") if isinstance(data, dict) else None
        docker = endpoints.get("docker") if isinstance(endpoints, dict) else None
        host = docker.get("Host") if isinstance(docker, dict) else None
        if isinstance(host, str) and host:
            hosts.append(host)
    return hosts


def docker_socket_candidates() -> list[str]:
    """Docker sockets a command of this user may reach, besides the system-wide ones.

    DOCKER_HOST, every context's socket, and where per-user daemons put theirs.
    """
    candidates = [
        host[len("unix://") :]
        for host in (environ().get("DOCKER_HOST", ""), *docker_context_hosts())
        if host.startswith("unix://")
    ]
    candidates += [str(home() / relative) for relative in _USER_DOCKER_SOCKETS]
    return candidates


def _is_loopback(host: str) -> bool:
    return host in ("localhost", "::1") or host.startswith("127.")


def docker_network_endpoints() -> list[str]:
    """Docker daemons on this host reached over TCP.

    A context whose endpoint is on loopback, and a daemon answering the API without TLS
    on loopback port 2375.
    """
    found: list[str] = []
    for host in docker_context_hosts():
        scheme, _, rest = host.partition("://")
        name = rest.rsplit("@", 1)[-1].split("/", 1)[0]
        name = name[1:].split("]", 1)[0] if name.startswith("[") else name.rsplit(":", 1)[0]
        if scheme in ("tcp", "http", "https") and _is_loopback(name) and host not in found:
            found.append(host)
    for address in _LOOPBACK:
        url = f"tcp://[{address}]:{_DOCKER_PLAIN_PORT}" if ":" in address else f"tcp://{address}:{_DOCKER_PLAIN_PORT}"
        if url not in found and _answers_docker_api(address, _DOCKER_PLAIN_PORT):
            found.append(url)
    return found


def _answers_docker_api(address: str, port: int) -> bool:
    try:
        conn = socket.create_connection((address, port), timeout=0.2)
    except OSError:
        return False
    with conn:
        try:
            conn.settimeout(1)
            conn.sendall(b"GET /_ping HTTP/1.0\r\n\r\n")
            reply = conn.recv(512)
        except OSError:
            return False
    return b"Api-Version" in reply


def _sshd_config_lines(path: str | Path, depth: int = 0) -> list[list[str]]:
    """The global keyword lines of an sshd config, Include expanded, up to the first Match."""
    out: list[list[str]] = []
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        words = line.split("#", 1)[0].replace("=", " ").split()
        if not words:
            continue
        keyword = words[0].lower()
        if keyword == "match":
            out.append(["match"])
            break
        if keyword == "include" and depth < 8:
            for pattern in words[1:]:
                full = Path(pattern) if Path(pattern).is_absolute() else Path(_SSHD_CONFIG).parent / pattern
                for included in sorted(_glob(full)):
                    nested = _sshd_config_lines(included, depth + 1)
                    out += nested
                    if nested and nested[-1] == ["match"]:
                        return out
            continue
        out.append([keyword, *words[1:]])
    return out


def sshd_endpoints(config: str = _SSHD_CONFIG) -> list[tuple[str, int]]:
    """Where this host's sshd listens, as far as its config says.

    Every ListenAddress on every Port: port 22 when none is set, loopback for a wildcard
    or no address. sshd_config is world-readable on the common systems; where it is not,
    loopback port 22.
    """
    lines = _sshd_config_lines(config)
    ports = [int(w[1]) for w in lines if w[0] == "port" and len(w) > 1 and w[1].isdigit()] or [22]
    endpoints: list[tuple[str, int]] = []
    for words in lines:
        if words[0] != "listenaddress" or len(words) < 2:
            continue
        spec = words[1]
        listen: list[tuple[str, int]]
        if spec.startswith("["):
            address, _, tail = spec[1:].partition("]")
            given = tail[1:] if tail.startswith(":") else ""
            listen = [(address, int(given))] if given.isdigit() else [(address, p) for p in ports]
        elif spec.count(":") == 1:
            address, given = spec.split(":")
            listen = [(address, int(given))] if given.isdigit() else []
        else:
            listen = [(spec, p) for p in ports]
        for address, port in listen:
            # a wildcard listens on loopback too, which is where a command on this host connects
            hosts = _LOOPBACK if address in ("0.0.0.0", "::", "*") else (address,)  # noqa: S104 - parsed, not bound
            endpoints += [(h, port) for h in hosts if (h, port) not in endpoints]
    if not endpoints:
        endpoints = [(h, p) for p in ports for h in _LOOPBACK]
    return endpoints


def _glob(pattern: Path) -> list[Path]:
    """The files matching an absolute ``pattern`` whose wildcards are in its last component."""
    return list(pattern.parent.glob(pattern.name))


def _host_keys() -> list[str]:
    keys: list[str] = []
    for path in sorted(_glob(Path(_SSH_HOST_KEYS))):
        with suppress(OSError):
            fields = path.read_text().split()
            if len(fields) >= 2:
                keys.append(f"{fields[0]} {fields[1]}")
    return keys


def _ssh_argv(ssh: str, known_hosts: str, user: str, host: str, port: int) -> list[str]:
    """Log in and open no session (`-N`): nothing runs on the far side, no shell rc either.

    Every option the user's ssh config could use to run something, forward something or
    leave a master connection behind is set here first, and ssh keeps the first value it
    gets. Only the real sshd of this host passes: its keys are the only known hosts.
    """
    options = {
        "BatchMode": "yes",
        "ConnectTimeout": "3",
        "LogLevel": "VERBOSE",
        "StrictHostKeyChecking": "yes",
        "UserKnownHostsFile": known_hosts,
        "GlobalKnownHostsFile": "/dev/null",
        "HostKeyAlias": _HOST_KEY_ALIAS,
        "UpdateHostKeys": "no",
        "ControlMaster": "no",
        "ControlPath": "none",
        "ProxyCommand": "none",
        "RemoteCommand": "none",
        "PermitLocalCommand": "no",
        "ForwardAgent": "no",
        "ForwardX11": "no",
        "ClearAllForwardings": "yes",
        "RequestTTY": "no",
    }
    argv = [ssh, "-N"]
    for key, value in options.items():
        argv += ["-o", f"{key}={value}"]
    return [*argv, "-p", str(port), "-l", user, host]


def _authenticates(argv: Sequence[str]) -> bool:
    """Run ssh until it says it authenticated, exits, or the timeout passes; then stop it."""
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=dict(environ())
        )
    except OSError:
        return False
    assert proc.stderr is not None
    seen = b""
    deadline = time.monotonic() + _SSH_TIMEOUT_S
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stderr, selectors.EVENT_READ)
            while (left := deadline - time.monotonic()) > 0 and selector.select(left):
                chunk = os.read(proc.stderr.fileno(), 4096)
                if not chunk:
                    break
                seen += chunk
                if _AUTHENTICATED.search(seen):
                    return True
        return False
    finally:
        proc.kill()
        proc.wait()
        proc.stderr.close()


def ssh_logs_in() -> bool:
    """True when this user logs in to this host's own sshd without a prompt.

    Measured by logging in: which keys sshd accepts can depend on configuration only root
    reads (AuthorizedKeysFile, Match, AllowUsers). Tried on each address and port sshd
    listens on, against this host's keys only; without readable host keys nothing is tried.
    """
    global _SSH_LOGS_IN
    if _SSH_LOGS_IN is None:
        _SSH_LOGS_IN = _try_ssh_logins()
    return _SSH_LOGS_IN


def _try_ssh_logins() -> bool:
    ssh = program("ssh")
    keys = _host_keys()
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return False  # a uid with no passwd entry (a container's) has no account sshd would take
    if ssh is None or not keys:
        return False
    reachable = []
    for host, port in sshd_endpoints():
        try:
            socket.create_connection((host, port), timeout=0.2).close()
        except OSError:
            continue
        reachable.append((host, port))
    if not reachable:
        return False
    with tempfile.NamedTemporaryFile("w", prefix="jail-known-hosts-", suffix=".txt") as known:
        known.write("".join(f"{_HOST_KEY_ALIAS} {key}\n" for key in keys))
        known.flush()
        return any(_authenticates(_ssh_argv(ssh, known.name, user, host, port)) for host, port in reachable)


def ssh_message(jail: str, when: str) -> str:
    """The report for `ssh_logs_in`, worded for ``jail`` ("jail", "sandbox")."""
    return (
        f"this user logged in to this host's sshd without a prompt: `ssh localhost cmd` from inside the {jail} "
        f"runs cmd outside it {when}. Run where sshd does not accept this user's key to close this."
    )
