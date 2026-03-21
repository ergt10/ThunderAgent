#!/usr/bin/env python3
"""Apply or verify the Harbor site-packages patches needed by the runtime-full workflow."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import apply_harbor_rootless_patch as rootless_patch


ASSET_ROOT = Path(__file__).resolve().parent / "harbor_site_packages"
TEMPLATE_ASSET = (
    ASSET_ROOT / "agents" / "installed" / "install-mini-swe-agent.sh.j2"
)
COMPOSE_ASSET = (
    ASSET_ROOT
    / "environments"
    / "docker"
    / "docker-compose-base.yaml"
)


def _resolve_targets() -> dict[str, Path]:
    base_mod = importlib.import_module("harbor.agents.installed.base")
    docker_mod = importlib.import_module("harbor.environments.docker.docker")

    base_dir = Path(base_mod.__file__).resolve().parent
    docker_dir = Path(docker_mod.__file__).resolve().parent

    return {
        "docker.py": Path(docker_mod.__file__).resolve(),
        "install-mini-swe-agent.sh.j2": base_dir / "install-mini-swe-agent.sh.j2",
        "docker-compose-base.yaml": docker_dir / "docker-compose-base.yaml",
    }


def _asset_map() -> dict[str, Path]:
    return {
        "install-mini-swe-agent.sh.j2": TEMPLATE_ASSET,
        "docker-compose-base.yaml": COMPOSE_ASSET,
    }


def _ensure_backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(path.read_text())
        print(f"BACKUP_WRITTEN {backup_path}")


def _sync_text_asset(*, target: Path, asset: Path, check: bool, backup: bool) -> None:
    expected = asset.read_text()
    current = target.read_text()
    if current == expected:
        print(f"PATCH_PRESENT {target}")
        return

    if check:
        raise SystemExit(f"PATCH_MISSING {target}")

    if backup:
        _ensure_backup(target)
    target.write_text(expected)
    print(f"PATCH_APPLIED {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify whether all patches are present")
    parser.add_argument("--backup", action="store_true", help="Write .bak files before patching")
    args = parser.parse_args()

    targets = _resolve_targets()

    docker_target = targets["docker.py"]
    docker_text = docker_target.read_text()
    docker_patched = rootless_patch.is_patch_present(docker_text)
    if args.check:
        if not docker_patched:
            raise SystemExit(f"PATCH_MISSING {docker_target}")
        print(f"PATCH_PRESENT {docker_target}")
    elif docker_patched:
        print(f"PATCH_PRESENT {docker_target}")
    else:
        new_text = rootless_patch.patch_text(docker_text)
        if args.backup:
            _ensure_backup(docker_target)
        docker_target.write_text(new_text)
        print(f"PATCH_APPLIED {docker_target}")

    for name, asset in _asset_map().items():
        _sync_text_asset(
            target=targets[name],
            asset=asset,
            check=args.check,
            backup=args.backup,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
