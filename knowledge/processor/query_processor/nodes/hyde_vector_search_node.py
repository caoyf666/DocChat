from typing import Any, Dict, List, Optional, Tuple, Union

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    _item_names_filter,
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)


class HyDeVectorSearchNode(BaseNode):
    """Generate a hypothetical document, then search chunks with its hybrid embedding."""

    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        rewritten_query, item_names = self._validate_state(state)

        if not self.config.chunks_collection:
            self.logger.error("CHUNKS_COLLECTION is empty")
            return state

        hy_document = self._generate_hy_document(rewritten_query, item_names)
        if hy_document is None:
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
            embed_hy_vector = generate_bge_m3_hybrid_vectors(
                bge_m3_client,
                [f"{rewritten_query}\n{hy_document}"],
            )
        except Exception as e:
            self.logger.error(f"HyDE document embedding failed: {e}")
            return state

        try:
            expr, expr_params = _item_names_filter(item_names)
            hybrid_search_req = create_hybrid_search_requests(
                dense_vector=embed_hy_vector["dense"][0],
                sparse_vector=embed_hy_vector["sparse"][0],
                expr=expr,
                expr_params=expr_params,
                limit=self.config.hyde_search_limit,
            )
            hybrid_search_res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_req,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=self.config.hyde_search_limit,
                output_fields=["chunk_id", "content", "item_name", "title"],
            )

            if not hybrid_search_res or not hybrid_search_res[0]:
                return state

            return {"hyde_embedding_chunks": hybrid_search_res[0]}
        except Exception as e:
            self.logger.error(f"HyDE vector search failed for {rewritten_query}: {e}")
            return state

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    def _generate_hy_document(self, rewritten_query: str, item_names: List[str]) -> Optional[str]:
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"LLM client unavailable: {e}")
            return None

        system_prompt = f"您是一位{item_names}方面的技术文档专家，擅长编写技术文档、操作手册和规格说明。"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names,
            rewritten_query=rewritten_query,
        )

        try:
            llm_response = llm_client.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
        except Exception as e:
            self.logger.error(f"failed to generate HyDE document for {item_names}: {e}")
            return None

        content = getattr(llm_response, "content", "")
        if not content or not content.strip():
            return None

        return content.strip()


if __name__ == "__main__":
    import json

    node = HyDeVectorSearchNode()
    result = node.process(
        {
            "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
            "item_names": ["RS-12 数字万用表"],
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
