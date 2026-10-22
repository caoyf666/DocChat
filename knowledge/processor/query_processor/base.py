"""Base class for query workflow nodes."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from knowledge.processor.query_processor.config import QueryConfig, get_config
from knowledge.processor.query_processor.exceptions import QueryProcessError
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import (
    add_done_task,
    add_node_duration,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
)

T = TypeVar("T")


class BaseNode(ABC):
    """Base node with logging, task tracking, and consistent error wrapping."""

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id", "") if isinstance(state, dict) else ""
        is_stream = state.get("is_stream", False) if isinstance(state, dict) else False
        try:
            self.logger.info(f"--- {self.name} start ---")
            if task_id:
                add_running_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            start_time = time.time()
            result = self.process(state)
            end_time = time.time()

            if task_id:
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, end_time - start_time)
                if is_stream:
                    self._push_progress(task_id)

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

    def _push_progress(self, task_id: str):
        push_sse_event(
            task_id=task_id,
            event=SSEEvent.PROGRESS,
            data={
                "status": get_task_status(task_id),
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            },
        )


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
