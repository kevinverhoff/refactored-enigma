import pytest
from unittest.mock import MagicMock, patch
import tools


SAMPLE_CHUNK = {
    "text": "The circuit breaker credit limits property tax liability.",
    "score": 0.95,
    "title": "Circuit Breaker Guidance",
    "author": "Wood",
    "memo_date": "2025-03-14",
    "doc_type": "MEMO",
    "source_year": 2025,
    "source": "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf",
}


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Replace module-level pipeline with a mock before each test."""
    mock = MagicMock()
    mock.retrieve.return_value = [SAMPLE_CHUNK]
    mock.gemini.models.generate_content.return_value.text = "Mocked generation output."
    tools._pipeline = mock
    yield mock
    tools._pipeline = None


def test_tools_importable():
    for t in [tools.search, tools.answer, tools.summarize,
              tools.compare_years, tools.get_topics, tools.extract_quotes]:
        assert hasattr(t, "invoke"), f"{t} has no .invoke()"


def test_search_returns_formatted_results(reset_pipeline):
    result = tools.search.invoke({"query": "circuit breaker credits"})
    assert "Circuit Breaker Guidance" in result
    assert "Wood" in result
    assert "2025-03-14" in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result
    assert "0.95" in result


def test_search_passes_filters_to_retrieve(reset_pipeline):
    tools.search.invoke({"query": "budget", "year": 2024, "doc_type": "MEMO", "author": "Wood"})
    reset_pipeline.retrieve.assert_called_once_with(
        "budget", year=2024, doc_type="MEMO", author="Wood", n=10
    )


def test_search_no_results(reset_pipeline):
    reset_pipeline.retrieve.return_value = []
    result = tools.search.invoke({"query": "nonexistent topic"})
    assert "No documents found" in result


def test_answer_returns_answer_and_urls(reset_pipeline):
    from rag_pipeline import RAGResponse
    reset_pipeline.answer.return_value = RAGResponse(
        answer="The circuit breaker credit [1] limits tax liability.",
        sources=[SAMPLE_CHUNK],
        query="What is the circuit breaker credit?",
        model="gemini-2.5-flash",
        n_retrieved=1,
    )
    result = tools.answer.invoke({"query": "What is the circuit breaker credit?"})
    assert "circuit breaker credit" in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_answer_no_results(reset_pipeline):
    from rag_pipeline import RAGResponse
    reset_pipeline.answer.return_value = RAGResponse(
        answer="No relevant documents found for this query.",
        sources=[],
        query="something obscure",
        model="gemini-2.5-flash",
        n_retrieved=0,
    )
    result = tools.answer.invoke({"query": "something obscure"})
    assert "No relevant documents" in result


def test_get_topics_reads_cluster_csv(tmp_path, monkeypatch):
    csv_content = (
        "cluster_id,cluster_label,doc_count,doc_types,mean_char_count,top_terms,date_range\n"
        "-1,Unassigned,155,MEMO (139),6506,,2021-04-19 to 2026-05-27\n"
        '0,reassessment parcels,33,MEMO (33),954,"parcels, reassessment",2022-01-03 to 2026-06-02\n'
        '1,sales disclosure,8,MEMO (8),3203,"sales disclosure, fee",2022-12-30 to 2025-12-11\n'
    )
    csv_file = tmp_path / "cluster_summary.csv"
    csv_file.write_text(csv_content)
    monkeypatch.setattr(tools, "_CLUSTER_CSV", csv_file)

    result = tools.get_topics.invoke({})
    assert "reassessment parcels" in result
    assert "sales disclosure" in result
    assert "33 docs" in result
    assert "Unassigned" not in result


def test_get_topics_missing_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_CLUSTER_CSV", tmp_path / "nonexistent.csv")
    result = tools.get_topics.invoke({})
    assert "not available" in result.lower()


def test_summarize_calls_retrieve_with_n12(reset_pipeline):
    result = tools.summarize.invoke({"topic": "circuit breaker credits"})
    reset_pipeline.retrieve.assert_called_once_with(
        "circuit breaker credits", year=None, doc_type=None, author=None, n=12
    )
    assert "Mocked generation output." in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_summarize_no_results(reset_pipeline):
    reset_pipeline.retrieve.return_value = []
    result = tools.summarize.invoke({"topic": "nonexistent"})
    assert "No documents found" in result


def test_extract_quotes_returns_quoted_output(reset_pipeline):
    result = tools.extract_quotes.invoke({"topic": "circuit breaker"})
    reset_pipeline.retrieve.assert_called_once_with("circuit breaker", year=None, n=8)
    assert "Mocked generation output." in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_extract_quotes_no_results(reset_pipeline):
    reset_pipeline.retrieve.return_value = []
    result = tools.extract_quotes.invoke({"topic": "nonexistent"})
    assert "No documents found" in result


def test_compare_years_two_retrieval_calls(reset_pipeline):
    early_chunk = {**SAMPLE_CHUNK, "source_year": 2022, "memo_date": "2022-06-01",
                   "source": "https://www.in.gov/dlgf/files/22-early.pdf"}
    late_chunk = {**SAMPLE_CHUNK, "source_year": 2025, "memo_date": "2025-06-01",
                  "source": "https://www.in.gov/dlgf/files/25-late.pdf"}
    reset_pipeline.retrieve.side_effect = [[early_chunk], [late_chunk]]

    result = tools.compare_years.invoke({
        "topic": "circuit breaker", "year_start": 2022, "year_end": 2025
    })

    calls = reset_pipeline.retrieve.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["where"] == {"source_year": {"$lte": 2022}}
    assert calls[1].kwargs["where"] == {"source_year": {"$gte": 2025}}
    assert "Mocked generation output." in result
    assert "22-early.pdf" in result
    assert "25-late.pdf" in result


def test_compare_years_no_results(reset_pipeline):
    reset_pipeline.retrieve.side_effect = [[], []]
    result = tools.compare_years.invoke({
        "topic": "nonexistent", "year_start": 2022, "year_end": 2025
    })
    assert "No documents found" in result


def test_browse_cluster_with_query(reset_pipeline, tmp_path, monkeypatch):
    csv_content = (
        "cluster_id,cluster_label,doc_count,doc_types,mean_char_count,top_terms,date_range\n"
        '0,reassessment parcels,33,MEMO (33),954,"parcels, reassessment",2022-01-03 to 2026-06-02\n'
    )
    (tmp_path / "cluster_summary.csv").write_text(csv_content)
    monkeypatch.setattr(tools, "_CLUSTER_CSV", tmp_path / "cluster_summary.csv")

    result = tools.browse_cluster.invoke({"cluster_id": 0, "query": "reassessment"})

    reset_pipeline.retrieve.assert_called_once_with(
        "reassessment", where={"cluster_id": {"$eq": 0}}, n=10
    )
    assert "reassessment parcels" in result
    assert "Circuit Breaker Guidance" in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_browse_cluster_no_query_uses_top_terms(reset_pipeline, tmp_path, monkeypatch):
    csv_content = (
        "cluster_id,cluster_label,doc_count,doc_types,mean_char_count,top_terms,date_range\n"
        '0,reassessment parcels,33,MEMO (33),954,"parcels, reassessment",2022-01-03 to 2026-06-02\n'
    )
    (tmp_path / "cluster_summary.csv").write_text(csv_content)
    monkeypatch.setattr(tools, "_CLUSTER_CSV", tmp_path / "cluster_summary.csv")

    tools.browse_cluster.invoke({"cluster_id": 0})

    call_args = reset_pipeline.retrieve.call_args
    assert call_args.kwargs["where"] == {"cluster_id": {"$eq": 0}}
    assert "parcels" in call_args.args[0]


def test_browse_cluster_invalid_id(reset_pipeline, tmp_path, monkeypatch):
    csv_content = (
        "cluster_id,cluster_label,doc_count,doc_types,mean_char_count,top_terms,date_range\n"
        '0,reassessment parcels,33,MEMO (33),954,"parcels, reassessment",2022-01-03 to 2026-06-02\n'
    )
    (tmp_path / "cluster_summary.csv").write_text(csv_content)
    monkeypatch.setattr(tools, "_CLUSTER_CSV", tmp_path / "cluster_summary.csv")

    result = tools.browse_cluster.invoke({"cluster_id": 999})
    assert "No cluster with ID 999" in result


def test_browse_cluster_no_results(reset_pipeline, tmp_path, monkeypatch):
    csv_content = (
        "cluster_id,cluster_label,doc_count,doc_types,mean_char_count,top_terms,date_range\n"
        '0,reassessment parcels,33,MEMO (33),954,"parcels, reassessment",2022-01-03 to 2026-06-02\n'
    )
    (tmp_path / "cluster_summary.csv").write_text(csv_content)
    monkeypatch.setattr(tools, "_CLUSTER_CSV", tmp_path / "cluster_summary.csv")
    reset_pipeline.retrieve.return_value = []

    result = tools.browse_cluster.invoke({"cluster_id": 0, "query": "reassessment"})
    assert "No documents found in cluster 0" in result
