"""Configuration for query workflow nodes."""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryConfig:
    """Query workflow configuration loaded from environment variables."""

    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.75"))
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.45"))
    )
    item_name_score_gap: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SCORE_GAP", "0.08"))
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "3"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )
    item_name_search_limit: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_SEARCH_LIMIT", "5"))
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
