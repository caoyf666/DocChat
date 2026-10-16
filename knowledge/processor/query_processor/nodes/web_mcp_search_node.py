import asyncio
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple, Union

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


class WebMcpSearchNode(BaseNode):
    """Search the web through a DashScope MCP server branch."""

    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        try:
            rewritten_query, _item_names = self._validate_state(state)
            web_search_results = asyncio.run(self._execute_mcp_server(rewritten_query))
        except Exception as e:
            self.logger.warning("Web MCP search failed, skipped this branch: %s", e)
            return state

        if not web_search_results:
            return state

        return {"web_search_docs": web_search_results}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    async def _execute_mcp_server(self, rewritten_query: str) -> List[Dict[str, Any]]:
        if not self.config.mcp_dashscope_base_url:
            self.logger.warning("MCP_DASHSCOPE_BASE_URL is empty")
            return []
        if not self.config.openai_api_key:
            self.logger.warning("OPENAI_API_KEY is empty")
            return []

        async with MCPServerStreamableHttp(
            name="web_search",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                "timeout": 60,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        ) as mcp_client:
            web_search_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": rewritten_query, "count": 3},
            )

        if not web_search_result.content:
            return []

        text_content = web_search_result.content[0]
        if not text_content or not text_content.text:
            return []

        try:
            text_content_obj = json.loads(text_content.text)
        except JSONDecodeError as e:
            self.logger.error(
                "Web search JSON parse failed: %s, content=%s, pos=%s",
                e.msg,
                e.doc,
                e.pos,
            )
            return []

        pages = text_content_obj.get("pages", [])
        if not pages:
            return []

        return [
            {
                "snippet": page.get("snippet", "").strip(),
                "title": page.get("title", "").strip(),
                "url": page.get("url", "").strip(),
            }
            for page in pages
        ]


if __name__ == "__main__":
    node = WebMcpSearchNode()
    print(
        node.process(
            {
                "rewritten_query": "RS-12 digital multimeter how to measure DC voltage?",
                "item_names": ["RS-12 digital multimeter"],
            }
        )
    )
