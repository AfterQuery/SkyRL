from pathlib import Path
from typing import List, Optional

from loguru import logger


class ToolathlonTaskDataset:
    """Dataset of Toolathlon task names.

    Each data file is a newline-separated list of task names (e.g. ``find-alita-paper``),
    or a directory containing task subdirectories (e.g. ``<toolathlon_repo>/tasks/finalpool``).
    Each dataset item's ``prompt`` is the task name; the generator resolves it to
    ``<tasks_domain>/<task_name>`` when invoking the Toolathlon runner.
    """

    def __init__(
        self,
        data_files: List[str],
        toolathlon_repo_path: str,
        tasks_domain: str = "finalpool",
    ):
        """
        Args:
            data_files: Paths to task-list text files (one task name per line, ``#`` comments
                allowed) or directories of task subdirectories.
            toolathlon_repo_path: Root of the Toolathlon repository, used to validate that
                each task exists.
            tasks_domain: Subdirectory of ``<repo>/tasks`` holding the tasks.
        """
        self.toolathlon_repo_path = Path(toolathlon_repo_path)
        self.tasks_domain = tasks_domain
        self.task_names = self._load_task_names(data_files)
        logger.info(f"ToolathlonTaskDataset initialized with {len(self.task_names)} tasks")

    def _load_task_names(self, data_files: List[str]) -> List[str]:
        names: List[str] = []
        for data_source in data_files:
            source_path = Path(data_source)
            if not source_path.exists():
                logger.warning(f"Path does not exist: {data_source}")
                continue
            if source_path.is_dir():
                names.extend(sorted(d.name for d in source_path.iterdir() if self._is_valid_task(d)))
            else:
                for line in source_path.read_text().splitlines():
                    name = line.strip()
                    if not name or name.startswith("#"):
                        continue
                    # Accept both "task-name" and "domain/task-name" entries.
                    name = name.split("/")[-1]
                    task_dir = self.toolathlon_repo_path / "tasks" / self.tasks_domain / name
                    if self._is_valid_task(task_dir):
                        names.append(name)
                    else:
                        logger.warning(f"Skipping invalid task (no task_config.json): {task_dir}")
        return names

    @staticmethod
    def _is_valid_task(task_dir: Path) -> bool:
        return task_dir.is_dir() and (task_dir / "task_config.json").is_file()

    def __getitem__(self, index: int) -> dict:
        if index >= len(self.task_names):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_names)}")
        return self._make_item(index)

    def __len__(self) -> int:
        return len(self.task_names)

    def __iter__(self):
        for index in range(len(self.task_names)):
            yield self._make_item(index)

    def _make_item(self, index: int) -> dict:
        task_name = self.task_names[index]
        return {
            "prompt": task_name,
            "env_class": None,
            "env_extras": {"task_name": task_name, "tasks_domain": self.tasks_domain},
            "uid": str(index),
        }

    def get_task_names(self) -> List[str]:
        return self.task_names.copy()

    def collate_fn(self, item_list):
        return item_list
