"""Milvus hybrid search helpers."""

import logging
from typing import Any, Dict, List, Tuple

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

logger = logging.getLogger(__name__)


def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    expr_params=None,
    limit=5,
) -> List[AnnSearchRequest]:
    """Create dense and sparse AnnSearchRequest objects for Milvus hybrid search."""
    if dense_vector is None or sparse_vector is None:
        raise ValueError("dense_vector and sparse_vector cannot be None")

    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        expr_params=expr_params,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        expr_params=expr_params,
        limit=limit,
    )
    return [dense_req, sparse_req]


def execute_hybrid_search_query(
    milvus_client: MilvusClient,
    collection_name: str,
    search_requests,
    ranker_weights=(0.5, 0.5),
    norm_score=True,
    limit=5,
    output_fields=None,
    search_params=None,
):
    """Execute Milvus hybrid search with weighted reranking."""
    if milvus_client is None:
        raise ValueError("milvus_client cannot be None")
    if not search_requests:
        raise ValueError("search_requests cannot be empty")

    if output_fields is None:
        output_fields = ["item_name"]

    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        result = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )
        total_hits = sum(len(hits) for hits in result) if result else 0
        logger.info(
            "Milvus hybrid search completed: queries=%s hits=%s",
            len(result) if result else 0,
            total_hits,
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Milvus hybrid search failed (collection={collection_name}): {e}") from e


def item_names_filter(item_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "item_name in {item_names}", {"item_names": item_names}


def _item_names_filter(item_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Backward-compatible alias used by query nodes."""
    return item_names_filter(item_names)
