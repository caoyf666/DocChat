import logging
import uuid
from typing import Any, Dict, List

from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.mongo_history_util import clear_history, get_recent_messages
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_task_result,
    update_task_status,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:
    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id() -> str:
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)

        query_init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream,
        }

        try:
            query_app.invoke(query_init_state)
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"运行查询流程出现异常: {e}")
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)

    def get_task_result(self, task_id: str) -> str:
        return get_task_result(task_id=task_id, key="answer")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(record.get("_id", "")),
                "session_id": record.get("session_id", ""),
                "role": record.get("role", ""),
                "text": record.get("text", ""),
                "rewritten_query": record.get("rewritten_query", ""),
                "item_names": record.get("item_names", []),
                "ts": record.get("ts"),
            }
            for record in records
        ]

    def clear_history(self, session_id: str) -> int:
        return clear_history(session_id)
