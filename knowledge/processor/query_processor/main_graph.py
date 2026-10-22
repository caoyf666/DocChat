"""Query workflow graph."""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutPutNode
from knowledge.processor.query_processor.nodes.hybrid_vector_search_node import HybridVectorSearch
from knowledge.processor.query_processor.nodes.hyde_vector_search_node import HyDeVectorSearchNode
from knowledge.processor.query_processor.nodes.item_name_confirmed_node import ItemNameConfirmedNode
from knowledge.processor.query_processor.nodes.reranker_node import RerankerNode
from knowledge.processor.query_processor.nodes.rrf_merge_node import RrfMergeNode
from knowledge.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode
from knowledge.processor.query_processor.state import QueryGraphState

load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    return bool(state.get("answer"))


def create_query_graph() -> CompiledStateGraph:
    workflow = StateGraph(QueryGraphState)  # type:ignore

    nodes = {
        "item_name_confirmed_node": ItemNameConfirmedNode(),
        "multi_search": lambda x: x,
        "hybrid_vector_search_node": HybridVectorSearch(),
        "hyde_vector_search_node": HyDeVectorSearchNode(),
        "web_mcp_search_node": WebMcpSearchNode(),
        "join": lambda x: {},
        "rrf_merge_node": RrfMergeNode(),
        "reranker_node": RerankerNode(),
        "answer_output_node": AnswerOutPutNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)  # type:ignore

    workflow.set_entry_point("item_name_confirmed_node")

    workflow.add_conditional_edges(
        "item_name_confirmed_node",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output_node",
        },
    )

    workflow.add_edge("multi_search", "hybrid_vector_search_node")
    workflow.add_edge("multi_search", "hyde_vector_search_node")
    workflow.add_edge("multi_search", "web_mcp_search_node")

    workflow.add_edge("hybrid_vector_search_node", "join")
    workflow.add_edge("hyde_vector_search_node", "join")
    workflow.add_edge("web_mcp_search_node", "join")

    workflow.add_edge("join", "rrf_merge_node")
    workflow.add_edge("rrf_merge_node", "reranker_node")
    workflow.add_edge("reranker_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


query_app = create_query_graph()
