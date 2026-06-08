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

def test_build_prompt_numbers_sources(pipeline):
    chunks = [
        {"title": "Memo A", "author": "Wood", "memo_date": "2024-01-01",
         "doc_type": "MEMO", "text": "Content of A"},
        {"title": "Memo B", "author": "Shackle", "memo_date": "2024-06-01",
         "doc_type": "MEMO", "text": "Content of B"},
    ]
    prompt = pipeline._build_prompt("What is homestead?", chunks)
    assert "[1] Memo A | Wood | 2024-01-01 | MEMO" in prompt
    assert "[2] Memo B | Shackle | 2024-06-01 | MEMO" in prompt
    assert "Content of A" in prompt
    assert "Content of B" in prompt
    assert "QUESTION: What is homestead?" in prompt


def test_build_prompt_missing_fields_fallback(pipeline):
    chunks = [{"title": "", "author": "", "memo_date": "", "doc_type": "", "text": "Body"}]
    prompt = pipeline._build_prompt("test?", chunks)
    assert "[1] Untitled | Unknown" in prompt
    assert "Body" in prompt


def test_build_prompt_sources_separated(pipeline):
    chunks = [
        {"title": "A", "author": "X", "memo_date": "2024-01-01",
         "doc_type": "MEMO", "text": "Text A"},
        {"title": "B", "author": "Y", "memo_date": "2024-02-01",
         "doc_type": "MEMO", "text": "Text B"},
    ]
    prompt = pipeline._build_prompt("q?", chunks)
    assert "---" in prompt

def test_embed_query_calls_gemini(pipeline):
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    pipeline.gemini.models.embed_content = MagicMock(
        return_value=MagicMock(embeddings=[mock_embedding])
    )
    result = pipeline._embed_query("homestead deduction")
    assert result == [0.1, 0.2, 0.3]
    pipeline.gemini.models.embed_content.assert_called_once()


def test_retrieve_returns_enriched_chunks(pipeline):
    pipeline._embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    pipeline.collection.query = MagicMock(return_value={
        "documents": [["chunk text"]],
        "metadatas": [[{
            "title": "Test Memo", "author": "Wood", "memo_date": "2024-01-01",
            "doc_type": "MEMO", "source_year": 2024,
            "source": "https://example.com/test.pdf",
        }]],
        "distances": [[0.25]],
    })
    chunks = pipeline.retrieve("homestead deduction")
    assert len(chunks) == 1
    assert chunks[0]["title"] == "Test Memo"
    assert chunks[0]["score"] == 0.75
    assert chunks[0]["text"] == "chunk text"


def test_retrieve_passes_where_filter(pipeline):
    pipeline._embed_query = MagicMock(return_value=[0.1])
    pipeline.collection.query = MagicMock(return_value={
        "documents": [[]], "metadatas": [[]], "distances": [[]]
    })
    pipeline.retrieve("query", year=2024)
    call_kwargs = pipeline.collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"source_year": {"$gte": 2024}}