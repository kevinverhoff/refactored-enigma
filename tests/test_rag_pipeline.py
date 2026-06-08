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