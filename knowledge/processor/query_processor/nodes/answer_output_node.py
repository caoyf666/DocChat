from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.task_util import set_task_result


class AnswerOutPutNode(BaseNode):
    """Generate the final answer and persist chat history."""

    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        task_id = state.get("task_id", "")

        if state.get("answer"):
            self._push_exist_answer(task_id, state)
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(prompt, task_id, state)

        self.save_history(state)
        return state

    def _push_exist_answer(self, task_id: str, state: QueryGraphState):
        set_task_result(task_id=task_id, key="answer", value=state.get("answer", ""))

    def _build_prompt(self, state: QueryGraphState) -> str:
        max_context_chars = self.config.max_context_chars
        user_query = state.get("rewritten_query") or state.get("original_query") or ""
        item_names = state.get("item_names") or []

        retrieval_context = state.get("reranked_docs") or []
        formatted_context, remaining_chars = self._format_retrieval_context(
            retrieval_context,
            max_context_chars,
        )

        chat_history_context = state.get("history") or []
        formatted_history = self._format_chat_history(chat_history_context, remaining_chars)

        return ANSWER_PROMPT.format(
            context=formatted_context or "暂无检索到上下文",
            history=formatted_history or "暂无历史上下文",
            item_names=",".join(item_names),
            question=user_query,
        )

    def _format_retrieval_context(
        self,
        retrieval_context: List[Dict[str, Any]],
        max_context_chars: int,
    ) -> Tuple[str, int]:
        formatted_lines = []
        usage = 0

        for index, context in enumerate(retrieval_context, 1):
            content = context.get("content", "")
            if not content:
                continue

            metadata_content = [f"[文档:{index}]"]
            for meta_field, template in [
                ("chunk_id", "[chunk_id={}]"),
                ("title", "[title={}]"),
                ("source", "[source={}]"),
                ("url", "[url={}]"),
            ]:
                field_value = str(context.get(meta_field, "")).strip()
                if field_value:
                    metadata_content.append(template.format(field_value))

            doc_score = context.get("score")
            if doc_score is not None:
                metadata_content.append(f"[score={float(doc_score):.6f}]")

            formatted_line = " ".join(metadata_content) + "\n" + content
            sep_chars = 2 if formatted_lines else 0
            total_length = sep_chars + len(formatted_line)

            if usage + total_length > max_context_chars:
                break

            formatted_lines.append(formatted_line)
            usage += total_length

        return "\n\n".join(formatted_lines), max_context_chars - usage

    def _generate_answer(self, prompt: str, task_id: str, state: QueryGraphState):
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except (ConnectionError, EnvironmentError) as e:
            self.logger.error(f"LLM client unavailable: {e}")
            state["answer"] = "LLM暂无法回答"
            set_task_result(task_id=task_id, key="answer", value=state["answer"])
            return

        state["answer"] = self._invoke_llm(prompt, llm_client)
        set_task_result(task_id=task_id, key="answer", value=state["answer"])

    def _invoke_llm(self, prompt: str, llm_client: ChatOpenAI) -> str:
        try:
            llm_res = llm_client.invoke(prompt)
            llm_content = getattr(llm_res, "content", "") or ""
            if not llm_content:
                return "LLM暂无法回答"
            return llm_content
        except Exception as e:
            self.logger.error(f"LLM answer generation failed: {e}")
            return "LLM暂无法回答"

    def save_history(self, state: QueryGraphState):
        session_id = state.get("session_id")
        user_query = state.get("original_query")
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names") or []
        answer = state.get("answer")

        if not session_id or not user_query or not answer:
            return

        try:
            save_chat_message(
                session_id=session_id,
                role="user",
                text=user_query,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
        except Exception as e:
            self.logger.error(f"保存历史对话到MongoDB失败 原因:{e}")

    def _format_chat_history(self, chat_history_context: List[Dict[str, Any]], usage_chars: int) -> str:
        formatted_lines = []
        used_chars = 0
        role_map = {"user": "用户", "assistant": "助手"}

        for msg in chat_history_context:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"
            separator_usage = 1 if formatted_lines else 0
            total_usage = separator_usage + len(formatted_line)

            if used_chars + total_usage > usage_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines)


if __name__ == "__main__":
    import json

    node = AnswerOutPutNode()
    state = {
        "original_query": "万用表如何测电压？",
        "rewritten_query": "RS-12 数字万用表如何测量电压？",
        "item_names": ["RS-12 数字万用表"],
        "reranked_docs": [
            {
                "content": "将旋钮拨到直流电压档，黑表笔接COM，红表笔接VΩ端。",
                "chunk_id": "c1",
                "title": "测量电压",
                "source": "local",
                "score": 0.9,
            }
        ],
    }
    print(json.dumps(node._build_prompt(state), ensure_ascii=False, indent=2))
