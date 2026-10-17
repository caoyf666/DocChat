import logging
from datetime import datetime
from typing import Any, Dict, List

from bson import ObjectId
from pymongo import DESCENDING
from pymongo.collection import Collection

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: List[str] = None,
    message_id: str = None,
) -> str:
    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names or [],
        "ts": ts,
    }

    collection = _get_collection()
    if message_id:
        collection.update_one({"_id": ObjectId(message_id)}, {"$set": document})
        return message_id

    result = collection.insert_one(document)
    return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    if not session_id:
        return []

    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    if not session_id:
        return 0

    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0
