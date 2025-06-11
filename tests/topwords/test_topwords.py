# In tests/topwords/test_topwords.py

import pytest
from lexos.topwords import TextacyKeywords, ZTestTopwords
import string


# ---------------- Fixtures ----------------

@pytest.fixture
def simple_text():
    return "Lexos is a tool for text analysis. Lexos helps find keywords in documents."

@pytest.fixture
def repeated_text():
    return "Lexos Lexos Lexos Lexos Lexos."

@pytest.fixture
def stopwords_text():
    return "the and if but or so yet for nor"

@pytest.fixture
def punctuation_text():
    return "!!! ??? ... ,,, ;;; :::"

@pytest.fixture
def target_texts():
    return ["Lexos is a tool for text analysis.", "Lexos helps find keywords."]

@pytest.fixture
def background_texts():
    return ["This document is about something else.", "It does not mention Lexos."]

@pytest.fixture
def identical_texts():
    return ["Lexos is great.", "Lexos is great."]

# ---------------- TextacyKeywords Tests ----------------

def test_textacy_keywords_textrank(simple_text):
    """Test textrank method returns expected structure and content."""
    extractor = TextacyKeywords(text=simple_text, method="textrank", topn=5)
    result = extractor()
    assert "keywords" in result
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) <= 5
    for kw in result["keywords"]:
        assert "term" in kw and "score" in kw
    # Check that a known keyword appears
    terms = [kw["term"] for kw in result["keywords"]]
    # MODIFIED: Check for lemmatized and lowercased terms instead of originals.
    assert any(t in ["lexos", "keyword", "document", "text analysis"] for t in terms)

def test_textacy_keywords_sgrank(simple_text):
    """Test sgrank method returns expected structure and content."""
    extractor = TextacyKeywords(text=simple_text, method="sgrank", topn=5)
    result = extractor()
    assert "keywords" in result
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) <= 5
    for kw in result["keywords"]:
        assert "term" in kw and "score" in kw

def test_textacy_keywords_large_topn(simple_text):
    """Test with a very large topn value."""
    extractor = TextacyKeywords(text=simple_text, method="textrank", topn=100)
    result = extractor()
    assert len(result["keywords"]) <= 100

def test_textacy_keywords_only_stopwords(stopwords_text):
    """Test with input containing only stopwords."""
    extractor = TextacyKeywords(text=stopwords_text, method="textrank", topn=5)
    result = extractor()
    assert result["keywords"] == []

def test_textacy_keywords_only_punctuation(punctuation_text):
    """Test with input containing only punctuation."""
    extractor = TextacyKeywords(text=punctuation_text, method="textrank", topn=5)
    result = extractor()
    assert result["keywords"] == []

def test_textacy_keywords_repeated_words(repeated_text):
    """Test with input containing repeated words."""
    extractor = TextacyKeywords(text=repeated_text, method="textrank", topn=3)
    result = extractor()
    assert len(result["keywords"]) <= 3
    if result["keywords"]:
        assert all(kw["term"].lower() == "lexos" for kw in result["keywords"])

def test_textacy_keywords_invalid_method(simple_text):
    """Test that an invalid method raises an exception."""
    with pytest.raises(ValueError, match="Invalid keyword extraction method."):
        # Pydantic v2 raises error on initialization, not on call
        TextacyKeywords(text=simple_text, method="invalid", topn=5)

def test_textacy_keywords_invalid_topn(simple_text):
    """Test that negative or zero topn raises an exception."""
    with pytest.raises(ValueError):
        TextacyKeywords(text=simple_text, method="textrank", topn=0)
    with pytest.raises(ValueError):
        TextacyKeywords(text=simple_text, method="textrank", topn=-1)


# ---------------- ZTestTopwords Tests ----------------

def test_ztest_topwords_basic(target_texts, background_texts):
    """Test ZTestTopwords returns expected structure and content."""
    extractor = ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=5)
    result = extractor()
    assert "topwords" in result
    assert isinstance(result["topwords"], list)
    assert len(result["topwords"]) <= 5
    for tw in result["topwords"]:
        # MODIFIED: Changed "zscore" to "z_score" to match implementation.
        assert "term" in tw and "z_score" in tw
    # Check that "Lexos" is likely a topword
    terms = [tw["term"].lower() for tw in result["topwords"]]
    assert "lexos" in terms

def test_ztest_topwords_empty_input():
    """Test ZTestTopwords with empty input returns empty list."""
    extractor = ZTestTopwords(target_texts=[], background_texts=[], topn=5)
    result = extractor()
    assert "topwords" in result
    assert isinstance(result["topwords"], list)
    assert len(result["topwords"]) == 0

def test_ztest_topwords_large_topn(target_texts, background_texts):
    """Test ZTestTopwords with large topn value."""
    extractor = ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=100)
    result = extractor()
    assert len(result["topwords"]) <= 100

def test_ztest_topwords_only_stopwords(stopwords_text):
    """Test ZTestTopwords with only stopwords."""
    # With remove_stopwords=True (default), no tokens should be processed
    extractor = ZTestTopwords(target_texts=[stopwords_text], background_texts=[stopwords_text], topn=5)
    result = extractor()
    assert result["topwords"] == []

def test_ztest_topwords_repeated_words(repeated_text):
    """Test ZTestTopwords with repeated words in target."""
    extractor = ZTestTopwords(target_texts=[repeated_text], background_texts=["other words"], topn=3)
    result = extractor()
    assert len(result["topwords"]) <= 3
    if result["topwords"]:
        # The top word should be 'lexos'
        assert result["topwords"][0]["term"].lower() == "lexos"

def test_ztest_topwords_identical_target_background(identical_texts):
    """Test ZTestTopwords with identical target and background."""
    extractor = ZTestTopwords(target_texts=identical_texts, background_texts=identical_texts, topn=5)
    result = extractor()
    # No word should be more significant, so Z-scores will be 0 and filtered out.
    assert result["topwords"] == []

def test_ztest_topwords_invalid_topn(target_texts, background_texts):
    """Test ZTestTopwords with invalid topn values."""
    with pytest.raises(ValueError):
        ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=0)
    with pytest.raises(ValueError):
        ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=-1)