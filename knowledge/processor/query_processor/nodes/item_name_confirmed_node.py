import json
import logging
import re
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.config import get_config
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

logger = logging.getLogger(__name__)


class _ItemNameExtractor:
    """Extract item names and rewrite the user query with an LLM."""

    def extract_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        llm_result = {"item_names": [], "rewritten_query": original_query}

        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f"LLM client unavailable: {e}")
            return llm_result

        system_prompt = "你是一位商品名提取专家，请从用户问题和历史对话中提取相关商品名并改写原始查询。"
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context.strip() if history_context else "暂无历史上下文",
            query=original_query,
        )

        try:
            llm_response = llm_client.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
        except Exception as e:
            logger.error(f"LLM item name extraction failed: {e}")
            return llm_result

        llm_response_content = llm_response.content
        if not llm_response_content:
            return llm_result

        try:
            parsed_result = self._clean_and_parse(llm_response_content)
        except JSONDecodeError:
            return llm_result

        llm_result["item_names"] = parsed_result.get("item_names", [])
        llm_result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
        return llm_result

    def _clean_and_parse(self, llm_response_content: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        try:
            llm_content_obj = json.loads(content)
        except JSONDecodeError as e:
            logger.error(f"LLM output is not valid JSON: {llm_response_content}; reason: {e}")
            raise

        raw_item_names = llm_content_obj.get("item_names")
        if not isinstance(raw_item_names, list):
            item_names = []
        else:
            item_names = [
                item_name.strip()
                for item_name in raw_item_names
                if isinstance(item_name, str) and item_name.strip()
            ]

        raw_rewritten_query = llm_content_obj.get("rewritten_query")
        rewritten_query = raw_rewritten_query.strip() if isinstance(raw_rewritten_query, str) else ""

        return {"item_names": item_names, "rewritten_query": rewritten_query}


class _ItemNameAligner:
    """Align LLM extracted names to item names stored in Milvus."""

    def __init__(self):
        self._config = get_config()

    def search_and_align(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        search_result = self._search_vector(item_names)
        if not search_result:
            return [], []

        confirmed, options = self._align(search_result)
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
        return confirmed, options

    def _search_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        final_search_result = []

        if not self._config.item_name_collection:
            logger.error("ITEM_NAME_COLLECTION is empty")
            return final_search_result

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"Milvus client unavailable: {e}")
            return final_search_result

        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"BGE-M3 client unavailable: {e}")
            return final_search_result

        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(
                model=bge_m3_client,
                embedding_documents=item_names,
            )
        except Exception as e:
            logger.error(f"failed to embed item names {item_names}: {e}")
            return final_search_result

        ranker_weights = (self._config.item_name_dense_weight, self._config.item_name_sparse_weight)
        for index, item_name in enumerate(item_names):
            try:
                hybrid_requests = create_hybrid_search_requests(
                    hybrid_vector_result["dense"][index],
                    hybrid_vector_result["sparse"][index],
                    limit=self._config.item_name_search_limit,
                )
                hybrid_search_result = execute_hybrid_search_query(
                    milvus_client=milvus_client,
                    collection_name=self._config.item_name_collection,
                    search_requests=hybrid_requests,
                    ranker_weights=ranker_weights,
                    norm_score=True,
                    limit=self._config.item_name_search_limit,
                    output_fields=["item_name"],
                )
            except Exception as e:
                logger.error(f"Milvus item-name search failed for {item_name}: {e}")
                matches = []
            else:
                matches = [
                    {
                        "score": item_search_res["distance"],
                        "item_name": item_search_res["entity"]["item_name"],
                    }
                    for item_search_res in (hybrid_search_result[0] if hybrid_search_result else [])
                ]

            final_search_result.append({"extracted_name": item_name, "matches": matches})

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        confirmed = []
        options = []

        for item_search_result in search_result:
            llm_extract_item_name = item_search_result.get("extracted_name")
            item_name_matches = item_search_result.get("matches") or []
            item_name_matches_sorted = sorted(
                item_name_matches,
                key=lambda x: x.get("score", 0),
                reverse=True,
            )

            high = [
                match
                for match in item_name_matches_sorted
                if match.get("score", 0) > self._config.item_name_high_confidence
            ]

            if high:
                exact = next(
                    (match for match in high if match.get("item_name") == llm_extract_item_name),
                    None,
                )
                if exact:
                    picked = exact.get("item_name")
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0].get("item_name")
                    if picked and picked not in confirmed:
                        confirmed.append(picked)
                else:
                    top_score = high[0].get("score", 0)
                    second_score = high[1].get("score", 0)
                    if top_score - second_score >= self._config.item_name_score_gap:
                        picked = high[0].get("item_name")
                        if picked and picked not in confirmed:
                            confirmed.append(picked)
                    else:
                        for match in high[: self._config.item_name_max_options]:
                            picked = match.get("item_name")
                            if picked and picked not in options and picked not in confirmed:
                                options.append(picked)
            else:
                mid = [
                    match
                    for match in item_name_matches_sorted
                    if match.get("score", 0) >= self._config.item_name_mid_confidence
                    and match.get("item_name") not in options
                    and match.get("item_name") not in confirmed
                ]
                for match in mid[: self._config.item_name_max_options]:
                    picked = match.get("item_name")
                    if picked and picked not in options and picked not in confirmed:
                        options.append(picked)

        return confirmed, options[: self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):
        item_name_score = {}
        for result in search_result:
            for match in result.get("matches", []):
                score = match.get("score", 0)
                item_name = match.get("item_name")
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        if not item_name_score:
            return confirmed

        max_score = max(item_name_score.values())
        return [
            name
            for name, score in item_name_score.items()
            if max_score - score <= self._config.item_name_score_gap
        ]


class ItemNameConfirmedNode(BaseNode):
    """Confirm product names mentioned in a query against the item-name collection."""

    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state.get("original_query", "")
        if not original_query:
            raise StateFieldError(
                node_name=self.name,
                field_name="original_query",
                expected_type=str,
            )

        history_context = self._format_history(state.get("history", []))
        llm_result = self._extractor.extract_item_name(original_query, history_context)

        item_names = llm_result.get("item_names", [])
        rewritten_query = llm_result.get("rewritten_query") or original_query

        if item_names:
            confirmed, options = self._aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []

        self._decide(confirmed, options, state, rewritten_query)
        return state

    def _format_history(self, history_context: list) -> str:
        formatted_history = []
        for history in history_context or []:
            if not isinstance(history, dict):
                continue
            role = history.get("role", "")
            text = history.get("text", "")
            if role or text:
                formatted_history.append(f"角色:{role},内容:{text}")
        return " ".join(formatted_history)

    def _decide(
        self,
        confirmed: List[str],
        options: List[str],
        state: QueryGraphState,
        rewritten_query: str,
    ):
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
            state["answer"] = ""
        elif options:
            state["item_names"] = []
            state["rewritten_query"] = rewritten_query
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state["item_names"] = []
            state["rewritten_query"] = rewritten_query
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == "__main__":
    from knowledge.processor.query_processor.base import setup_logging

    setup_logging()
    node = ItemNameConfirmedNode()
    result = node.process({"original_query": "RS-12数字万用表如何测量电压？"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
