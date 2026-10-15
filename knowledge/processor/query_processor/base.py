"""Base class for query workflow nodes."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from knowledge.processor.query_processor.config import QueryConfig, get_config
from knowledge.processor.query_processor.exceptions import QueryProcessError
from knowledge.utils.task_util import add_done_task, add_node_duration, add_running_task

T = TypeVar("T")


class BaseNode(ABC):
    """Base node with logging, task tracking, and consistent error wrapping."""

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id", "") if isinstance(state, dict) else ""
        try:
            self.logger.info(f"--- {self.name} start ---")
            if task_id:
                add_running_task(task_id, self.name)

            start_time = time.time()
            result = self.process(state)
            end_time = time.time()

            if task_id:
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, end_time - start_time)

            self.logger.info(f"--- {self.name} done ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} failed: {e}")
            raise QueryProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
