"""Shell-level tests for ``thorn-sandbox-entrypoint``.

The entrypoint is a thin POSIX shell trampoline that runs as root at
container start, installs the gateway-mounted broker MITM CA into
the system trust store, then ``setpriv``-drops to the operator's
uid/gid before execing the toolhost daemon.  These tests drive the
script under ``/bin/sh`` with stub ``update-ca-certificates`` and
``setpriv`` on ``$PATH`` so the behaviour is observable without a
real container boot or privilege escalation.

The tests pin three properties the Dockerfile + ``ContainerDaemonHost``
rely on:

1. the script parses cleanly under ``sh -n`` (no bashisms),
2. when the broker CA is mounted it is copied to the Debian trust
   directory and ``update-ca-certificates --fresh`` is invoked, and
3. the final step exec-replaces the script process with the tail
   command, delegating uid/gid/bounding-set drop to ``setpriv``.
"""

from __future__ import annotations

import os
import stat
import subprocess
from importlib.resources import as_file, files
from pathlib import Path

import pytest


ENTRYPOINT_RESOURCE = (
    "thorn.sandbox._resources",
    "thorn-sandbox-entrypoint",
)


@pytest.fixture
def entrypoint_path() -> Path:
    """Path to the wheel-shipped entrypoint script.

    Uses :func:`importlib.resources.files` so editable and built-wheel
    installs both resolve to a real on-disk file the shell can exec.
    """
    traversable = files(ENTRYPOINT_RESOURCE[0]).joinpath(
        ENTRYPOINT_RESOURCE[1],
    )
    with as_file(traversable) as on_disk:
        yield Path(on_disk).resolve()


def _make_stub(
    path: Path,
    *,
    record_file: Path,
    label: str,
) -> None:
    """Drop an executable stub at *path* that logs its argv to *record_file*.

    The stub is a tiny ``/bin/sh`` script that appends a single
    label-prefixed line per invocation (argv space-joined), which
    lets tests assert both "was this called?" and "with what
    args?" without needing a real ``setpriv``/``update-ca-certificates``.
    """
    path.write_text(
        "#!/bin/sh\n"
        f'printf "{label}: %s\\n" "$*" >> "{record_file}"\n'
        "exit 0\n",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestShellParse:
    def test_script_parses_under_sh_n(
        self,
        entrypoint_path: Path,
    ) -> None:
        """``sh -n`` should accept the entrypoint with no diagnostics."""
        result = subprocess.run(
            ["/bin/sh", "-n", str(entrypoint_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"sh -n rejected the entrypoint: stderr={result.stderr!r}"
        )

    def test_script_is_executable(
        self,
        entrypoint_path: Path,
    ) -> None:
        """The resource must ship with +x so the Dockerfile ``COPY``
        preserves the executable bit and the runtime can exec it
        directly (no ``sh /usr/local/bin/thorn-sandbox-entrypoint``
        wrapper needed)."""
        mode = entrypoint_path.stat().st_mode
        assert mode & stat.S_IXUSR, f"script not executable: mode={oct(mode)}"


class TestCaInstall:
    def _run_entrypoint(
        self,
        entrypoint_path: Path,
        tmp_path: Path,
        *,
        ca_content: str | None,
        uid: str | None = "4242",
        gid: str | None = "4343",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        """Run the entrypoint with stubbed ``setpriv`` / ``update-ca-certificates``.

        The script expects fixed paths:
        ``/etc/thorn/onecli-ca.pem`` and
        ``/usr/local/share/ca-certificates/thorn-broker.crt``.  We
        cannot rewrite those without rewriting the script, so this
        helper builds a per-test shim that re-executes the script
        with the two paths remapped into ``tmp_path`` via a thin
        sed.  That keeps the real script under test while avoiding
        the need for root or a real container.
        """
        record = tmp_path / "calls.log"
        record.write_text("")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        _make_stub(
            bin_dir / "setpriv", record_file=record, label="setpriv",
        )
        _make_stub(
            bin_dir / "update-ca-certificates",
            record_file=record,
            label="update-ca-certificates",
        )

        src_path = tmp_path / "broker-ca-mount.pem"
        dst_dir = tmp_path / "trust"
        dst_dir.mkdir()
        dst_path = dst_dir / "thorn-broker.crt"

        if ca_content is not None:
            src_path.write_text(ca_content)

        # Rewrite the two hard-coded paths so the test can observe
        # the copy without needing root.  Keeping the script's
        # text as the source of truth (rather than forking a test-
        # only copy) means future changes to the trampoline remain
        # covered automatically.
        script_body = entrypoint_path.read_text()
        script_body = script_body.replace(
            "BROKER_CA_SRC=/etc/thorn/onecli-ca.pem",
            f"BROKER_CA_SRC={src_path}",
        )
        script_body = script_body.replace(
            "BROKER_CA_DST=/usr/local/share/ca-certificates/thorn-broker.crt",
            f"BROKER_CA_DST={dst_path}",
        )
        patched = tmp_path / "entrypoint-patched.sh"
        patched.write_text(script_body)
        patched.chmod(0o755)

        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            # ``install`` is part of coreutils and reads no other env
            # we care about, so propagating a minimal environment
            # keeps the test deterministic.
            "LANG": "C",
        }
        if uid is not None:
            env["THORN_SANDBOX_UID"] = uid
        if gid is not None:
            env["THORN_SANDBOX_GID"] = gid

        result = subprocess.run(
            [str(patched), "/bin/echo", "hello-from-daemon"],
            capture_output=True,
            text=True,
            env=env,
        )
        return result, record

    def test_ca_install_runs_when_mount_present(
        self,
        entrypoint_path: Path,
        tmp_path: Path,
    ) -> None:
        result, record = self._run_entrypoint(
            entrypoint_path, tmp_path,
            ca_content="-----BEGIN CERTIFICATE-----\nX\n",
        )
        assert result.returncode == 0, (
            f"entrypoint failed: stderr={result.stderr!r}"
        )
        log = record.read_text()
        assert "update-ca-certificates: --fresh" in log
        # And the CA ended up at the (patched) destination path.
        dst = tmp_path / "trust" / "thorn-broker.crt"
        assert dst.is_file()
        assert "BEGIN CERTIFICATE" in dst.read_text()

    def test_ca_install_skipped_when_mount_absent(
        self,
        entrypoint_path: Path,
        tmp_path: Path,
    ) -> None:
        """No broker mount means we're in a non-broker deployment;
        the script must still exec the tail command without
        touching the trust store."""
        result, record = self._run_entrypoint(
            entrypoint_path, tmp_path, ca_content=None,
        )
        assert result.returncode == 0
        log = record.read_text()
        assert "update-ca-certificates" not in log
        dst = tmp_path / "trust" / "thorn-broker.crt"
        assert not dst.exists()


class TestPrivilegeDrop:
    def test_exec_delegates_to_setpriv_with_configured_uid_gid(
        self,
        entrypoint_path: Path,
        tmp_path: Path,
    ) -> None:
        record = tmp_path / "calls.log"
        record.write_text("")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        _make_stub(
            bin_dir / "setpriv", record_file=record, label="setpriv",
        )
        _make_stub(
            bin_dir / "update-ca-certificates",
            record_file=record,
            label="update-ca-certificates",
        )

        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "LANG": "C",
            "THORN_SANDBOX_UID": "4242",
            "THORN_SANDBOX_GID": "4343",
        }

        # Run the real script (no broker mount at the hard-coded
        # path, so it skips CA install) and assert the ``setpriv``
        # stub got invoked with the expected argv tail.
        result = subprocess.run(
            [str(entrypoint_path), "/usr/bin/env", "python3", "-c", "pass"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        log = record.read_text()
        setpriv_lines = [l for l in log.splitlines() if l.startswith("setpriv:")]
        assert len(setpriv_lines) == 1, f"unexpected log: {log!r}"
        argv = setpriv_lines[0][len("setpriv: "):]
        assert "--reuid=4242" in argv
        assert "--regid=4343" in argv
        assert "--clear-groups" in argv
        assert "--inh-caps=-all" in argv
        assert "--bounding-set=-all" in argv
        assert "/usr/bin/env python3 -c pass" in argv

    def test_refuses_when_uid_env_unset(
        self,
        entrypoint_path: Path,
        tmp_path: Path,
    ) -> None:
        """Running the script without ``THORN_SANDBOX_UID`` set is a
        configuration error: the gateway is expected to inject it,
        so a missing value means the daemon would run as root
        forever.  Fail fast rather than silently retain privilege."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        record = tmp_path / "calls.log"
        record.write_text("")
        _make_stub(
            bin_dir / "setpriv", record_file=record, label="setpriv",
        )
        _make_stub(
            bin_dir / "update-ca-certificates",
            record_file=record,
            label="update-ca-certificates",
        )

        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "LANG": "C",
            # THORN_SANDBOX_UID deliberately omitted.
            "THORN_SANDBOX_GID": "4343",
        }
        result = subprocess.run(
            [str(entrypoint_path), "/bin/true"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 64, (
            f"expected exit 64 (sysexits EX_USAGE-ish); got "
            f"{result.returncode}, stderr={result.stderr!r}"
        )
        assert "THORN_SANDBOX_UID" in result.stderr
        # setpriv must NOT have been reached.
        assert "setpriv:" not in record.read_text()
