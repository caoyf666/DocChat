from typing import Any, Dict, List, Tuple

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class RrfMergeNode(BaseNode):
    """Merge local retrieval branches with Reciprocal Rank Fusion."""

    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        embedding_chunks = state.get("embedding_chunks") or []
        hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

        search_result_weight = {
            "embedding_search_chunks": (self._validate_search_result(embedding_chunks), 1.0),
            "hyde_embedding_search_chunks": (self._validate_search_result(hyde_embedding_chunks), 1.0),
        }

        merged_rrf_results = self._merge_rrf_docs(
            list(search_result_weight.values()),
            self.config.rrf_k,
            self.config.rrf_max_results,
        )
        state["rrf_chunks"] = [rrf_res for rrf_res, _ in merged_rrf_results]
        return state

    def _merge_rrf_docs(
        self,
        rrf_inputs: List[Tuple[List[Dict[str, Any]], float]],
        rrf_k: int,
        rrf_max_results: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        chunk_score = {}
        chunk_data = {}

        for search_result, weight in rrf_inputs:
            for rank, entity in enumerate(search_result, 1):
                chunk_id = entity.get("chunk_id")
                if not chunk_id:
                    continue

                chunk_score[chunk_id] = chunk_score.get(chunk_id, 0.0) + weight / (rrf_k + rank)
                chunk_data.setdefault(chunk_id, entity)

        final_rrf_result = sorted(
            [(chunk_data.get(chunk_id), score) for chunk_id, score in chunk_score.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return final_rrf_result[:rrf_max_results] if rrf_max_results else final_rrf_result

    def _validate_search_result(self, search_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not search_result:
            return []

        search_results = []
        for res in search_result:
            if not res or not isinstance(res, dict):
                continue

            entity = res.get("entity")
            if not entity or not isinstance(entity, dict):
                continue

            search_results.append(entity)

        return search_results


if __name__ == "__main__":
    import json

    node = RrfMergeNode()
    result = node.process(
        {
            "embedding_chunks": [
                {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
                {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            ],
            "hyde_embedding_chunks": [
                {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
                {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            ],
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
