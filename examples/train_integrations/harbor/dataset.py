import json

from loguru import logger
from typing import List, Optional
from pathlib import Path


class HarborTaskDataset:
    """
    A dataset that loads Harbor task data from direct file/directory paths.
    Each dataset item is a path to a task directory.
    """

    def __init__(
        self,
        data_files: List[str],
        max_tasks: Optional[int] = None,
    ):
        """
        Initialize the HarborTaskDataset.

        Args:
            data_files: List of direct file/directory paths pointing to Harbor task data
            max_tasks: If set, limit the dataset to this many tasks (for smoke/pilot runs)
        """
        self.data_files = data_files

        # Load all data files
        self.task_paths = self._load_data_files()

        if max_tasks is not None and max_tasks < len(self.task_paths):
            logger.info(f"HarborTaskDataset limiting to {max_tasks} tasks (out of {len(self.task_paths)} available)")
            self.task_paths = self.task_paths[:max_tasks]

        logger.info(f"HarborTaskDataset initialized with {len(self.task_paths)} task paths")

    @staticmethod
    def _canonicalize_task_path(task_path: Path) -> Path:
        return task_path.expanduser().resolve()

    @classmethod
    def _make_uid(cls, task_path: Path) -> str:
        return str(cls._canonicalize_task_path(task_path))

    def _load_manifest_task_paths(self, manifest_path: Path) -> List[Path]:
        """Resolve task directories listed in a curated-subset MANIFEST.json."""
        try:
            data = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.warning(f"Failed to parse Harbor manifest {manifest_path}: {exc}")
            return []

        tasks = data.get("tasks")
        if not isinstance(tasks, dict):
            logger.warning(f"Harbor manifest missing task mapping: {manifest_path}")
            return []

        data_root = manifest_path.parent.parent
        resolved_paths = []
        for bucket_name, bucket_tasks in tasks.items():
            if not isinstance(bucket_tasks, list):
                logger.warning(f"Skipping malformed Harbor manifest bucket {bucket_name} in {manifest_path}")
                continue
            for rel_task_path in bucket_tasks:
                task_path = (data_root / rel_task_path).resolve()
                if not self._is_valid_task_directory(task_path):
                    logger.warning(f"Skipping invalid Harbor manifest task path: {task_path}")
                    continue
                resolved_paths.append(task_path)

        logger.info(f"Resolved {len(resolved_paths)} Harbor tasks from manifest {manifest_path}")
        return resolved_paths

    def _load_data_files(self) -> List[Path]:
        """Load all data files from direct paths and return list of task paths."""
        task_paths = []
        seen_uids = set()

        for data_source in self.data_files:
            source_path = Path(data_source)

            if not source_path.exists():
                logger.warning(f"Path does not exist: {data_source}")
                continue

            logger.info(f"Loading data from: {data_source}")

            # If the path is a directory, find all valid task subdirectories
            if source_path.is_dir():
                # Look for task subdirectories and validate them
                all_dirs = sorted(d for d in source_path.iterdir() if d.is_dir())
                valid_task_dirs = [d for d in all_dirs if self._is_valid_task_directory(d)]

                if valid_task_dirs:
                    for task_dir in valid_task_dirs:
                        uid = self._make_uid(task_dir)
                        if uid in seen_uids:
                            logger.warning(f"Skipping duplicate Harbor task path: {task_dir}")
                            continue
                        task_paths.append(self._canonicalize_task_path(task_dir))
                        seen_uids.add(uid)
                    logger.info(
                        f"Found {len(valid_task_dirs)} valid task directories out of {len(all_dirs)} total directories"
                    )
                elif self._is_valid_task_directory(source_path):
                    # If no subdirectories but the main directory is valid, treat it as a task
                    uid = self._make_uid(source_path)
                    if uid in seen_uids:
                        logger.warning(f"Skipping duplicate Harbor task path: {source_path}")
                        continue
                    task_paths.append(self._canonicalize_task_path(source_path))
                    seen_uids.add(uid)
                    logger.info("Using main directory as valid task")
                elif (source_path / "MANIFEST.json").is_file():
                    manifest_task_dirs = self._load_manifest_task_paths(source_path / "MANIFEST.json")
                    for task_dir in manifest_task_dirs:
                        uid = self._make_uid(task_dir)
                        if uid in seen_uids:
                            logger.warning(f"Skipping duplicate Harbor task path from manifest: {task_dir}")
                            continue
                        task_paths.append(self._canonicalize_task_path(task_dir))
                        seen_uids.add(uid)
                else:
                    logger.warning(f"No valid task directories found in {source_path}")
            else:
                # If it's a file, treat it as a single task (files can't be valid task directories)
                logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")

        return task_paths

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        """Check if a directory is a valid task directory (has instruction.md file)."""
        if not task_path.is_dir():
            return False

        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict:
        """Get a task path by index as a dictionary with 'prompt', 'env_class', and 'env_extras' keys."""
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        return {
            "prompt": str(self.task_paths[index]),
            "env_class": None,
            "env_extras": {"data_source": str(self.task_paths[index])},
            "uid": self._make_uid(self.task_paths[index]),
        }

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.task_paths)

    def __iter__(self):
        """Iterate over all task paths as dictionaries."""
        for task_path in self.task_paths:
            yield {
                "prompt": str(task_path),
                "env_class": None,
                "env_extras": {"data_source": str(task_path)},
                "uid": self._make_uid(task_path),
            }

    def get_task_paths(self) -> List[Path]:
        """Return all task paths as a list."""
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        """Collate function for batching task dictionaries."""
        return item_list
