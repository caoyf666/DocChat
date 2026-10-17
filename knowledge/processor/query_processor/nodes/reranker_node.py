import math
from typing import Any, Dict, List

from FlagEmbedding import FlagReranker

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    """Rerank fused local results and web snippets."""

    name = "reranker_node"

    @staticmethod
    def _sigmoid(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = state.get("rewritten_query") or state.get("original_query") or ""
        rerank_outputs = self._collect_rerank_inputs(state)
        refine_docs = self._refine_rank(user_query, rerank_outputs)
        state["reranked_docs"] = self._cliff_cutoff(
            refine_docs,
            self.config.rerank_min_top_k,
            self.config.rerank_max_top_k,
        )
        return state

    def _collect_rerank_inputs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        final_docs = []

        rrf_chunks = state.get("rrf_chunks") or []
        for chunk in rrf_chunks:
            if not chunk or not isinstance(chunk, dict):
                continue

            content = chunk.get("content", "")
            if not content:
                continue

            final_docs.append(
                self._format_doc(
                    content=content,
                    chunk_id=chunk.get("chunk_id"),
                    title=chunk.get("title", ""),
                    source="local",
                )
            )

        web_search_docs = state.get("web_search_docs") or []
        for doc in web_search_docs:
            if not doc or not isinstance(doc, dict):
                continue

            content = doc.get("snippet", "")
            if not content:
                continue

            final_docs.append(
                self._format_doc(
                    content=content,
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    source="web",
                )
            )

        self.logger.info(f"Reranker input document count: {len(final_docs)}")
        return final_docs

    def _format_doc(
        self,
        content: str,
        chunk_id: int = None,
        title: str = "",
        url: str = "",
        source: str = "",
    ):
        return {
            "content": content,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "source": source,
        }

    def _refine_rank(self, user_query: str, rerank_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rerank_outputs:
            return []

        try:
            rerank_client: FlagReranker = AIClients.get_bge_m3_rerank_client()
        except (ConnectionError, EnvironmentError) as e:
            self.logger.error(f"BGE reranker unavailable: {e}")
            return [{**doc, "score": None} for doc in rerank_outputs]

        query_doc_pairs = [(user_query, doc.get("content", "")) for doc in rerank_outputs]

        try:
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_pairs)
            doc_score = [
                {**doc, "score": self._sigmoid(float(score))}
                for doc, score in zip(rerank_outputs, rerank_scores)
            ]
            return sorted(doc_score, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            self.logger.error(f"BGE reranker scoring failed: {e}")
            return [{**doc, "score": None} for doc in rerank_outputs]

    def _cliff_cutoff(
        self,
        refine_docs: List[Dict[str, Any]],
        rerank_min_top_k: int,
        rerank_max_top_k: int,
    ) -> List[Dict[str, Any]]:
        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        cut_off = upper_bound
        max_gap = 0

        for i in range(0, upper_bound - 1):
            current_score = refine_docs[i].get("score")
            next_score = refine_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            gap = current_score - next_score
            if gap >= 0.15 and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
                self.logger.info(f"Rerank cliff found at position {i + 1}")

        cut_off = max(cut_off, lower_bound)
        return refine_docs[:cut_off]


if __name__ == "__main__":
    import json

    node = RerankerNode()
    result = node.process(
        {
            "rewritten_query": "怎么测这块主板的短路问题？",
            "rrf_chunks": [
                {
                    "chunk_id": "local_1",
                    "title": "主板维修手册",
                    "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表蜂鸣档测量。",
                }
            ],
            "web_search_docs": [
                {
                    "url": "https://example.com/repair",
                    "title": "短路查修指南",
                    "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。",
                }
            ],
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
