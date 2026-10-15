"""Query workflow state definitions."""

import copy
from typing import List, TypedDict


class QueryGraphState(TypedDict, total=False):
    session_id: str
    task_id: str
    message_id: str
    original_query: str
    answer: str
    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool


DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    return copy.deepcopy(DEFAULT_STATE)
