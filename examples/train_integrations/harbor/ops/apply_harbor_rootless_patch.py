#!/usr/bin/env python3
"""Apply or verify the Harbor rootless Docker upload patch in site-packages."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

UPLOAD_PATCH_MARKER = "Equivalent to tar --owner=0 --group=0 --numeric-owner."
CLEANUP_PATCH_MARKER = "Preserve prebuilt image cache across Harbor trial cleanup."
REMOTE_PATCH_MARKERS = (
    'async def _stream_tar_to_container(',
    '"docker",\n            "compose",',
    '"--owner=0"',
    '"--group=0"',
    '"--numeric-owner"',
)

STOCK_IMPORTS = """import asyncio
import asyncio.subprocess
import os
import shlex
from pathlib import Path
"""

PATCHED_IMPORTS = """import asyncio
import asyncio.subprocess
import io
import os
import shlex
import tarfile
from pathlib import Path, PurePosixPath
"""

STOCK_UPLOAD_BLOCK = """    async def upload_file(self, source_path: Path | str, target_path: str):
        await self._run_docker_compose_command(
            [
                "cp",
                str(source_path),
                f"main:{target_path}",
            ],
            check=True,
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        await self._run_docker_compose_command(
            [
                "cp",
                f"{source_dir}/.",
                f"main:{target_dir}",
            ],
            check=True,
        )
"""

PATCHED_UPLOAD_BLOCK = """    @staticmethod
    def _normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
        # Equivalent to tar --owner=0 --group=0 --numeric-owner.
        tarinfo.uid = 0
        tarinfo.gid = 0
        tarinfo.uname = "root"
        tarinfo.gname = "root"
        return tarinfo

    @classmethod
    def _build_tar_from_dir(cls, source_dir: Path | str) -> bytes:
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for path in sorted(source_dir.rglob("*")):
                archive.add(
                    path,
                    arcname=path.relative_to(source_dir).as_posix(),
                    recursive=False,
                    filter=cls._normalize_tarinfo,
                )
        return buffer.getvalue()

    @classmethod
    def _build_tar_from_file(
        cls, source_path: Path | str, target_name: str
    ) -> bytes:
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            archive.add(
                source_path,
                arcname=target_name,
                recursive=False,
                filter=cls._normalize_tarinfo,
            )
        return buffer.getvalue()

    async def _get_main_container_id(self) -> str:
        result = await self._run_docker_compose_command(["ps", "-q", "main"])
        container_id = (result.stdout or "").strip()
        if not container_id:
            raise RuntimeError(
                f"Could not resolve the main container id for {self.environment_name}"
            )
        return container_id.splitlines()[-1]

    async def _run_docker_exec_command(
        self,
        container_id: str,
        command: list[str],
        stdin_bytes: bytes | None = None,
    ) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            container_id,
            *command,
            stdin=(
                asyncio.subprocess.PIPE
                if stdin_bytes is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await process.communicate(stdin_bytes)
        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None

        result = ExecResult(
            stdout=stdout,
            stderr=None,
            return_code=process.returncode or 0,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Docker exec failed for environment {self.environment_name}. "
                f"Command: {' '.join(command)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}."
            )
        return result

    async def _stream_tar_to_directory(
        self, tar_bytes: bytes, target_dir: str
    ) -> None:
        container_id = await self._get_main_container_id()
        await self._run_docker_exec_command(
            container_id, ["mkdir", "-p", target_dir]
        )
        await self._run_docker_exec_command(
            container_id,
            ["tar", "-xf", "-", "-C", target_dir],
            stdin_bytes=tar_bytes,
        )

    async def upload_file(self, source_path: Path | str, target_path: str):
        target = PurePosixPath(target_path)
        target_dir = str(target.parent)
        target_name = target.name
        if not target_name:
            raise ValueError(f"Invalid upload target path: {target_path}")

        tar_bytes = self._build_tar_from_file(source_path, target_name)
        await self._stream_tar_to_directory(tar_bytes, target_dir)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        tar_bytes = self._build_tar_from_dir(source_dir)
        await self._stream_tar_to_directory(tar_bytes, target_dir)
"""

STOCK_STOP_BLOCK = """    async def stop(self, delete: bool):
        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `--delete` option are set. "
                "keep_containers takes precedence."
            )
        if self._keep_containers:
            try:
                await self._run_docker_compose_command(["stop"])
            except RuntimeError as e:
                self.logger.warning(f"Docker compose stop failed: {e}")
        elif delete:
            try:
                await self._run_docker_compose_command(
                    ["down", "--rmi", "all", "--volumes", "--remove-orphans"]
                )
            except RuntimeError as e:
                self.logger.warning(f"Docker compose down failed: {e}")

            # await self._cleanup_build_cache()
        else:
            try:
                await self._run_docker_compose_command(["down"])
            except RuntimeError as e:
                self.logger.warning(f"Docker compose down failed: {e}")
"""

PATCHED_STOP_BLOCK = """    async def stop(self, delete: bool):
        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `--delete` option are set. "
                "keep_containers takes precedence."
            )
        if self._keep_containers:
            try:
                await self._run_docker_compose_command(["stop"])
            except RuntimeError as e:
                self.logger.warning(f"Docker compose stop failed: {e}")
        elif delete:
            try:
                cleanup_command = ["down", "--volumes", "--remove-orphans"]
                # Preserve prebuilt image cache across Harbor trial cleanup.
                if not self._use_prebuilt:
                    cleanup_command[1:1] = ["--rmi", "all"]
                await self._run_docker_compose_command(cleanup_command)
            except RuntimeError as e:
                self.logger.warning(f"Docker compose down failed: {e}")

            # await self._cleanup_build_cache()
        else:
            try:
                await self._run_docker_compose_command(["down"])
            except RuntimeError as e:
                self.logger.warning(f"Docker compose down failed: {e}")
"""


def resolve_target(explicit_target: str | None) -> Path:
    if explicit_target:
        return Path(explicit_target).resolve()
    docker_mod = importlib.import_module("harbor.environments.docker.docker")
    return Path(docker_mod.__file__).resolve()


def has_upload_patch_present(text: str) -> bool:
    if UPLOAD_PATCH_MARKER in text:
        return True
    return all(marker in text for marker in REMOTE_PATCH_MARKERS)


def has_cleanup_patch_present(text: str) -> bool:
    if CLEANUP_PATCH_MARKER in text:
        return True
    return 'cleanup_command = ["down", "--volumes", "--remove-orphans"]' in text


def is_patch_present(text: str) -> bool:
    return has_upload_patch_present(text) and has_cleanup_patch_present(text)


def patch_text(text: str) -> str:
    if not has_upload_patch_present(text):
        if STOCK_IMPORTS not in text:
            raise RuntimeError("Unexpected Harbor docker.py imports block; aborting patch.")
        if STOCK_UPLOAD_BLOCK not in text:
            raise RuntimeError("Unexpected Harbor docker.py upload block; aborting patch.")
        text = text.replace(STOCK_IMPORTS, PATCHED_IMPORTS, 1)
        text = text.replace(STOCK_UPLOAD_BLOCK, PATCHED_UPLOAD_BLOCK, 1)

    if not has_cleanup_patch_present(text):
        if STOCK_STOP_BLOCK not in text:
            raise RuntimeError("Unexpected Harbor docker.py stop block; aborting patch.")
        text = text.replace(STOCK_STOP_BLOCK, PATCHED_STOP_BLOCK, 1)

    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None, help="Explicit harbor/environments/docker/docker.py path")
    parser.add_argument("--check", action="store_true", help="Only verify whether the patch is present")
    parser.add_argument("--backup", action="store_true", help="Write a .bak copy before patching")
    args = parser.parse_args()

    target = resolve_target(args.target)
    text = target.read_text()
    patched = is_patch_present(text)

    if args.check:
        if not patched:
            raise SystemExit(f"PATCH_MISSING {target}")
        print(f"PATCH_PRESENT {target}")
        return 0

    if patched:
        print(f"PATCH_PRESENT {target}")
        return 0

    new_text = patch_text(text)
    if args.backup:
        backup_path = target.with_suffix(target.suffix + ".bak")
        backup_path.write_text(text)
        print(f"BACKUP_WRITTEN {backup_path}")
    target.write_text(new_text)
    print(f"PATCH_APPLIED {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
