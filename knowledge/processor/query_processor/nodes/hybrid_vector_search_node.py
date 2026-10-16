from typing import Any, Dict, List, Tuple, Union

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    _item_names_filter,
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)


class HybridVectorSearch(BaseNode):
    """Search chunks directly with the rewritten query using dense+sparse vectors."""

    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        rewritten_query, item_names = self._validate_state(state)

        if not self.config.chunks_collection:
            self.logger.error("CHUNKS_COLLECTION is empty")
            return state

        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 client unavailable: {e}")
            return state

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus client unavailable: {e}")
            return state

        try:
            embed_query_vector = generate_bge_m3_hybrid_vectors(
                model=bge_m3_client,
                embedding_documents=[rewritten_query],
            )
        except Exception as e:
            self.logger.error(f"query embedding failed for {rewritten_query}: {e}")
            return state

        try:
            expr, expr_params = _item_names_filter(item_names)
            hybrid_search_req = create_hybrid_search_requests(
                dense_vector=embed_query_vector["dense"][0],
                sparse_vector=embed_query_vector["sparse"][0],
                expr=expr,
                expr_params=expr_params,
                limit=self.config.embedding_search_limit,
            )
            hybrid_search_res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_req,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=self.config.embedding_search_limit,
                output_fields=["chunk_id", "content", "item_name", "title"],
            )

            if not hybrid_search_res or not hybrid_search_res[0]:
                return state

            return {"embedding_chunks": hybrid_search_res[0]}
        except Exception as e:
            self.logger.error(f"hybrid vector search failed for {rewritten_query}: {e}")
            return state

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names


if __name__ == "__main__":
    import json

    node = HybridVectorSearch()
    result = node.process(
        {
            "rewritten_query": "万用表如何测量电阻",
            "item_names": ["RS-12 数字万用表"],
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
