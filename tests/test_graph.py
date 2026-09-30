import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graphs.graph_builder import GraphBuilder


@pytest.mark.asyncio
async def test_topic_graph_pauses_for_approval_then_resumes(mock_llm, mock_research_llm):
    graph_builder = GraphBuilder(mock_llm, mock_research_llm)
    graph = graph_builder.setup_graph(usecase="topic", checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    result = await graph.ainvoke({"topic": "Agentic AI"}, config=config)
    assert "__interrupt__" in result

    state = await graph.aget_state(config)
    assert state.next == ("human_approval",)

    final = await graph.ainvoke(Command(resume="approved"), config=config)
    assert final["approval_status"] == "approved"
