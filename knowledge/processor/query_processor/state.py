"""Query workflow state definitions."""

import copy
from typing import List, TypedDict


class QueryGraphState(TypedDict, total=False):
    session_id: str
    task_id: str
    message_id: str
    original_query: str
    embedding_chunks: list
    hyde_embedding_chunks: list
    rrf_chunks: list
    web_search_docs: list
    reranked_docs: list
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
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
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
