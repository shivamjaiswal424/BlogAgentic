from unittest.mock import MagicMock

from conftest import FakeResponse

from src.nodes.blog_node import BlogNode


def test_title_creation(mock_llm):
    mock_llm.invoke.return_value = FakeResponse("# My Fake Title")
    node = BlogNode(llm=mock_llm)
    result = node.title_creation({"topic": "Agentic AI"})
    assert result == {"blog": {"title": "# My Fake Title"}}


def test_title_creation_no_topic_returns_none(mock_llm):
    node = BlogNode(llm=mock_llm)
    assert node.title_creation({"topic": ""}) is None


def test_content_generation(mock_llm):
    mock_llm.invoke.return_value = FakeResponse("Some generated content.")
    node = BlogNode(llm=mock_llm)
    state = {"topic": "Agentic AI", "blog": {"title": "My Title"}, "research_notes": "notes"}
    result = node.content_generation(state)
    assert result == {"blog": {"title": "My Title", "content": "Some generated content."}}


def test_research_success(mock_research_llm):
    node = BlogNode(llm=MagicMock(), research_llm=mock_research_llm)
    result = node.research({"topic": "Agentic AI"})
    assert result["research_notes"] == "Fake research notes."
    assert result["sources"] == ["https://example.com/source"]


def test_research_falls_back_on_error(mock_llm):
    failing_research_llm = MagicMock()
    failing_research_llm.invoke.side_effect = Exception("search backend down")
    mock_llm.invoke.return_value = FakeResponse("Fallback research notes.")
    node = BlogNode(llm=mock_llm, research_llm=failing_research_llm)
    result = node.research({"topic": "Agentic AI"})
    assert result == {"research_notes": "Fallback research notes.", "sources": []}


def test_translation_parses_title_and_content(mock_llm):
    mock_llm.invoke.return_value = FakeResponse("TITLE: Titre Faux\nCONTENT:\nContenu faux ici.")
    node = BlogNode(llm=mock_llm)
    state = {"blog": {"title": "Fake Title", "content": "Fake content."}, "current_language": "french"}
    result = node.translation(state)
    assert result == {"blog": {"title": "Titre Faux", "content": "Contenu faux ici."}}


def test_translation_falls_back_to_original_on_bad_format(mock_llm):
    mock_llm.invoke.return_value = FakeResponse("the model ignored the requested format entirely")
    node = BlogNode(llm=mock_llm)
    state = {"blog": {"title": "Fake Title", "content": "Fake content."}, "current_language": "french"}
    result = node.translation(state)
    assert result == {"blog": state["blog"]}


def test_route_decision_hindi():
    node = BlogNode(llm=MagicMock())
    assert node.route_decision({"current_language": "hindi"}) == "hindi"


def test_route_decision_french():
    node = BlogNode(llm=MagicMock())
    assert node.route_decision({"current_language": "french"}) == "french"


def test_approval_decision_approved():
    node = BlogNode(llm=MagicMock())
    assert node.approval_decision({"approval_status": "approved"}) == "approved"


def test_approval_decision_rejected_by_default():
    node = BlogNode(llm=MagicMock())
    assert node.approval_decision({"approval_status": "anything else"}) == "rejected"
