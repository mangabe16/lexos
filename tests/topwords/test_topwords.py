import pytest
from pydantic import ValidationError
from lexos.topwords import TextacyKeywords, ZTestTopwords
import spacy


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
    return ["This document is about general topics.", "It does not mention special tools."]

@pytest.fixture
def identical_texts():
    return ["Lexos is great.", "Lexos is great."]

@pytest.fixture
def nlp():
    return spacy.blank("en")

# ---------------- TextacyKeywords Tests ----------------

def test_textacy_keywords_textrank(simple_text, nlp):
    extractor = TextacyKeywords(text=simple_text, method="textrank", topn=5)
    extractor()
    result = extractor.to_dict()
    assert "keywords" in result
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) <= 5
    for kw in result["keywords"]:
        assert "term" in kw and "score" in kw
    # Test .to_df() and .to_list()
    df = extractor.to_df()
    assert not df.empty or len(result["keywords"]) == 0
    tuples = extractor.to_list()
    assert isinstance(tuples, list)
    # Test doc attribute
    extractor2 = TextacyKeywords(doc=nlp(simple_text), method="textrank", topn=5)
    extractor2()
    assert hasattr(extractor2, "doc")
    assert hasattr(extractor2.doc._, "keywords")
    assert extractor2.doc._.keywords == extractor2.to_dict()["keywords"]


def test_textacy_keywords_sgrank(simple_text, nlp):
    extractor = TextacyKeywords(text=simple_text, method="sgrank", topn=5)
    extractor()
    result = extractor.to_dict()
    assert "keywords" in result
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) <= 5
    for kw in result["keywords"]:
        assert "term" in kw and "score" in kw
    # Test doc input
    extractor2 = TextacyKeywords(doc=nlp(simple_text), method="sgrank", topn=5)
    extractor2()
    assert hasattr(extractor2, "doc")
    assert hasattr(extractor2.doc._, "keywords")
    assert extractor2.doc._.keywords == extractor2.to_dict()["keywords"]


def test_textacy_keywords_large_topn(simple_text):
    extractor = TextacyKeywords(text=simple_text, method="textrank", topn=100)
    extractor()
    assert len(extractor.to_dict()["keywords"]) <= 100


def test_textacy_keywords_only_stopwords(stopwords_text):
    extractor = TextacyKeywords(text=stopwords_text, method="textrank", topn=5)
    extractor()
    assert extractor.to_dict()["keywords"] == []


def test_textacy_keywords_only_punctuation(punctuation_text):
    extractor = TextacyKeywords(text=punctuation_text, method="textrank", topn=5)
    extractor()
    assert extractor.to_dict()["keywords"] == []


def test_textacy_keywords_repeated_words(repeated_text):
    extractor = TextacyKeywords(text=repeated_text, method="textrank", topn=3)
    extractor()
    kws = extractor.to_dict()["keywords"]
    assert len(kws) <= 3
    if kws:
        assert all(kw["term"].lower() == "lexos" for kw in kws)


def test_textacy_keywords_invalid_method(simple_text):
    with pytest.raises(ValidationError):
        TextacyKeywords(text=simple_text, method="invalid", topn=5)


def test_textacy_keywords_invalid_topn(simple_text):
    with pytest.raises(ValidationError):
        TextacyKeywords(text=simple_text, method="textrank", topn=0)
    with pytest.raises(ValidationError):
        TextacyKeywords(text=simple_text, method="textrank", topn=-1)


# ---------------- ZTestTopwords Tests ----------------

def test_ztest_topwords_basic(target_texts, background_texts, nlp):
    extractor = ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=5)
    extractor()
    result = extractor.to_dict()
    assert "topwords" in result
    assert isinstance(result["topwords"], list)
    assert len(result["topwords"]) <= 5
    for tw in result["topwords"]:
        assert "term" in tw and "z_score" in tw
    # Test .to_df() and .to_list()
    df = extractor.to_df()
    assert not df.empty or len(result["topwords"]) == 0
    tuples = extractor.to_list()
    assert isinstance(tuples, list)
    # Test doc attribute
    docs = [nlp(text) for text in target_texts]
    extractor2 = ZTestTopwords(target_docs=docs, background_texts=background_texts, topn=5, docs=docs)
    extractor2()
    assert hasattr(extractor2, "target_docs")
    assert hasattr(docs[0]._, "topwords")
    assert docs[0]._.topwords == extractor2.to_list()


def test_ztest_topwords_empty_input():
    extractor = ZTestTopwords(target_texts=[], background_texts=[], topn=5)
    extractor()
    assert extractor.to_dict()["topwords"] == []


def test_ztest_topwords_large_topn(target_texts, background_texts):
    extractor = ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=100)
    extractor()
    assert len(extractor.to_dict()["topwords"]) <= 100


def test_ztest_topwords_only_stopwords(stopwords_text):
    extractor = ZTestTopwords(target_texts=[stopwords_text], background_texts=["some other text"], topn=5)
    extractor()
    assert extractor.to_dict()["topwords"] == []


def test_ztest_topwords_repeated_words(repeated_text):
    extractor = ZTestTopwords(target_texts=[repeated_text], background_texts=["other words"], topn=3)
    extractor()
    tws = extractor.to_dict()["topwords"]
    assert len(tws) <= 3
    if tws:
        assert tws[0]["term"].lower() == "lexos"


def test_ztest_topwords_identical_target_background(identical_texts):
    extractor = ZTestTopwords(target_texts=identical_texts, background_texts=identical_texts, topn=5)
    extractor()
    assert extractor.to_dict()["topwords"] == []


def test_ztest_topwords_invalid_topn(target_texts, background_texts):
    with pytest.raises(ValidationError):
        ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=0)
    with pytest.raises(ValidationError):
        ZTestTopwords(target_texts=target_texts, background_texts=background_texts, topn=-1)