import pytest
from unittest.mock import MagicMock, patch

from rag_pipeline import RAGResponse, EMBED_MODEL, GENERATION_MODEL


def test_ragresponse_defaults():
    r = RAGResponse(answer="hello")
    assert r.answer == "hello"
    assert r.sources == []
    assert r.query == ""
    assert r.model == ""
    assert r.n_retrieved == 0


def test_ragresponse_all_fields():
    r = RAGResponse(
        answer="an answer",
        sources=[{"title": "doc"}],
        query="a question",
        model="gemini-2.0-flash",
        n_retrieved=1,
    )
    assert r.n_retrieved == 1
    assert r.sources[0]["title"] == "doc"


def test_constants_are_strings():
    assert isinstance(EMBED_MODEL, str)
    assert isinstance(GENERATION_MODEL, str)
    assert len(EMBED_MODEL) > 0
    assert len(GENERATION_MODEL) > 0
from rag_pipeline import RAGResponse, RAGPipeline, EMBED_MODEL, GENERATION_MODEL


@pytest.fixture
def pipeline():
    with patch("rag_pipeline.genai.Client"), \
         patch("rag_pipeline.chromadb.PersistentClient") as mock_chroma, \
         patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        yield RAGPipeline()


def test_build_where_no_params(pipeline):
    assert pipeline._build_where() is None


def test_build_where_year_only(pipeline):
    assert pipeline._build_where(year=2024) == {"source_year": {"$gte": 2024}}


def test_build_where_doc_type_only(pipeline):
    assert pipeline._build_where(doc_type="MEMO") == {"doc_type": {"$eq": "MEMO"}}


def test_build_where_author_only(pipeline):
    assert pipeline._build_where(author="Wood") == {"author": {"$eq": "Wood"}}


def test_build_where_multiple_params(pipeline):
    result = pipeline._build_where(year=2024, doc_type="MEMO")
    assert result == {"$and": [
        {"source_year": {"$gte": 2024}},
        {"doc_type": {"$eq": "MEMO"}},
    ]}


def test_build_where_all_three(pipeline):
    result = pipeline._build_where(year=2023, doc_type="TEMPLATE", author="Wood")
    assert result == {"$and": [
        {"source_year": {"$gte": 2023}},
        {"doc_type": {"$eq": "TEMPLATE"}},
        {"author": {"$eq": "Wood"}},
    ]}


def test_build_where_raw_overrides_convenience(pipeline):
    raw = {"cluster_id": {"$eq": 16}}
    result = pipeline._build_where(year=2024, doc_type="MEMO", where=raw)
    assert result == raw