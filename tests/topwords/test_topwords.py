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
    return [
        "This document is about general topics.",
        "It does not mention special tools.",
    ]


@pytest.fixture
def identical_texts():
    return ["Lexos is great.", "Lexos is great."]


@pytest.fixture
def nlp():
    return spacy.blank("en")

# ---------------- TextacyKeywords Tests ----------------

def test_textacy_keywords_textrank(simple_text, nlp):
    extractor = TextacyKeywords(document=simple_text, method="textrank", topn=5)
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
    extractor2 = TextacyKeywords(document=nlp(simple_text), method="textrank", topn=5)
    extractor2()
    
    doc = nlp(simple_text)
    extractor3 = TextacyKeywords(document=doc, method="textrank", topn=5)
    extractor3()
    assert hasattr(doc._, "keywords")
    # The extension is set on the doc passed in
    assert doc._.keywords == extractor3.to_dict()["keywords"]

def test_textacy_keywords_sgrank(simple_text, nlp):
    extractor = TextacyKeywords(document=simple_text, method="sgrank", topn=5)
    extractor()
    result = extractor.to_dict()
    assert "keywords" in result
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) <= 5
    for kw in result["keywords"]:
        assert "term" in kw and "score" in kw
    # Test doc input
    doc = nlp(simple_text)
    extractor2 = TextacyKeywords(document=doc, method="sgrank", topn=5)
    extractor2()
    assert hasattr(doc._, "keywords")
    assert doc._.keywords == extractor2.to_dict()["keywords"]

def test_textacy_keywords_large_topn(simple_text):
    extractor = TextacyKeywords(document=simple_text, method="textrank", topn=100)
    extractor()
    assert len(extractor.to_dict()["keywords"]) <= 100

def test_textacy_keywords_only_stopwords(stopwords_text):
    extractor = TextacyKeywords(document=stopwords_text, method="textrank", topn=5)
    extractor()
    assert extractor.to_dict()["keywords"] == []

def test_textacy_keywords_only_punctuation(punctuation_text):
    extractor = TextacyKeywords(document=punctuation_text, method="textrank", topn=5)
    extractor()
    assert extractor.to_dict()["keywords"] == []

def test_textacy_keywords_repeated_words(repeated_text):
    extractor = TextacyKeywords(document=repeated_text, method="textrank", topn=3)
    extractor()
    kws = extractor.to_dict()["keywords"]
    assert len(kws) <= 3
    if kws:
        assert all(
            all(word == "lexos" for word in kw["term"].lower().replace(".","").split())
            for kw in kws
        )

def test_textacy_keywords_invalid_method(simple_text):
    with pytest.raises(ValidationError):
        TextacyKeywords(document=simple_text, method="invalid", topn=5)

def test_textacy_keywords_invalid_topn(simple_text):
    with pytest.raises(ValidationError):
        TextacyKeywords(document=simple_text, method="textrank", topn=0)
    with pytest.raises(ValidationError):
        TextacyKeywords(document=simple_text, method="textrank", topn=-1)

# ---------------- ZTestTopwords Tests ----------------

def test_ztest_topwords_basic(target_texts, background_texts, nlp):
    extractor = ZTestTopwords(
        target_documents=target_texts, background_documents=background_texts, topn=5
    )
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
    extractor2 = ZTestTopwords(
        target_documents=docs, background_documents=background_texts, topn=5, docs=docs
    )
    extractor2()
    assert hasattr(extractor2, "docs")
    assert hasattr(docs[0]._, "topwords")
    assert docs[0]._.topwords == extractor2.to_list()

def test_ztest_topwords_empty_input():
    extractor = ZTestTopwords(target_documents=[], background_documents=[], topn=5)
    extractor()
    assert extractor.to_dict()["topwords"] == []

def test_ztest_topwords_large_topn(target_texts, background_texts):
    extractor = ZTestTopwords(
        target_documents=target_texts, background_documents=background_texts, topn=100
    )
    extractor()
    assert len(extractor.to_dict()["topwords"]) <= 100

def test_ztest_topwords_only_stopwords(stopwords_text):
    extractor = ZTestTopwords(
        target_documents=[stopwords_text],
        background_documents=["some other text"],
        topn=5
    )
    extractor()
    topwords = extractor.to_dict()["topwords"]
    assert all(word['term'] not in stopwords_text.split() for word in topwords)

def test_ztest_topwords_repeated_words(repeated_text):
    extractor = ZTestTopwords(
        target_documents=[repeated_text], background_documents=["other words"], topn=3
    )
    extractor()
    tws = extractor.to_dict()["topwords"]
    assert len(tws) <= 3
    if tws:
        assert tws[0]["term"].lower() == "lexos"

def test_ztest_topwords_identical_target_background(identical_texts):
    extractor = ZTestTopwords(
        target_documents=identical_texts, background_documents=identical_texts, topn=5
    )
    extractor()
    assert extractor.to_dict()["topwords"] == []

def test_ztest_topwords_invalid_topn(target_texts, background_texts):
    with pytest.raises(ValidationError):
        ZTestTopwords(
            target_documents=target_texts, background_documents=background_texts, topn=0
        )
    with pytest.raises(ValidationError):
        ZTestTopwords(
            target_documents=target_texts, background_documents=background_texts, topn=-1
        )

def test_ztest_topwords_background_docs_direct_input(
    nlp, target_texts, background_texts
):
    """Test providing `background_docs` directly."""
    target_docs = [nlp(text) for text in target_texts]
    background_docs = [nlp(text) for text in background_texts]
    extractor = ZTestTopwords(
        target_documents=target_docs, background_documents=background_docs, topn=5
    )
    extractor()
    result = extractor.to_dict()
    assert "topwords" in result
    assert isinstance(result["topwords"], list)
    assert len(result["topwords"]) <= 5
    for tw in result["topwords"]:
        assert "term" in tw and "z_score" in tw

def test_ztest_topwords_missing_background_input():
    """Tests the ValueError when neither background_documents nor background_docs are provided."""
    with pytest.raises(
        ValueError,
        match="The 'background_documents' field must be provided.",
    ):
        ZTestTopwords(target_documents=["some text"], topn=5)()

def test_ztest_topwords_case_sensitivity_and_line_137(nlp):
    """Tests both case_sensitive=True and case_sensitive=False scenarios."""
    # Adjusted data to make 'apple' a more distinguishing term when case_sensitive=False
    target_texts_case_ci = [
        "Many Apple trees are here.",
        "I like apple pie.",
        "Sweet apple.",
    ]
    background_texts_case_ci = ["A tree is tall.", "I like fruit.", "Orange sweet."]

    # Test case_sensitive=True: "Apple" and "apple" should be distinct
    target_cs = ["Apple is a fruit.", "Apple computers."]
    background_cs = ["Eating an apple.", "red apple."]
    extractor_cs = ZTestTopwords(
        target_documents=target_cs,
        background_documents=background_cs,
        topn=5,
        case_sensitive=True,
        remove_stopwords=False,
    )
    extractor_cs()
    cs_topwords = extractor_cs.to_list()
    assert any(term == "Apple" for term, score in cs_topwords if score > 0)
    assert not any(term == "apple" for term, score in cs_topwords if score > 0)

    # Test case_sensitive=False: "Apple" and "apple" should be treated as the same "apple"
    extractor_ci = ZTestTopwords(
        target_documents=target_texts_case_ci,
        background_documents=background_texts_case_ci,
        topn=5,
        case_sensitive=False,
        remove_stopwords=False,
    )
    extractor_ci()
    ci_topwords = extractor_ci.to_list()
    assert any(term == "apple" for term, score in ci_topwords if score > 0)
    assert not any(
        term == "Apple" for term, _ in ci_topwords
    )

def test_ztest_topwords_missing_target_input():
    """Tests the ValueError when neither target_documents nor target_docs are provided."""
    with pytest.raises(
        ValueError, match="The 'target_documents' field must be provided."
    ):
        extractor = ZTestTopwords(background_documents=["some text"], topn=5)
        extractor()

def test_ztest_topwords_remove_digits_line_180(nlp):
    """Tests that digits are correctly removed when `remove_digits` is True."""
    target = ["Document with 123 numbers."]
    background = ["Another document with no digits."]
    extractor = ZTestTopwords(
        target_documents=target,
        background_documents=background,
        topn=5,
        remove_digits=True,
        remove_stopwords=False,
        remove_punct=False,
    )
    extractor()
    result = extractor.to_dict()["topwords"]
    assert not any(any(char.isdigit() for char in kw["term"]) for kw in result)

    extractor_keep_digits = ZTestTopwords(
        target_documents=["Document with 123 numbers."],
        background_documents=["Another document with 456 numbers."],
        topn=5,
        remove_digits=False,
        remove_stopwords=False,
        remove_punct=False,
    )
    extractor_keep_digits()
    result_keep_digits = extractor_keep_digits.to_dict()["topwords"]
    assert any(any(char.isdigit() for char in kw["term"]) for kw in result_keep_digits)

def test_ztest_topwords_denominator_is_zero_line_214_216(nlp):
    """Tests a scenario where a term is present in 100% of the tokens in both corpora."""
    target_all_same = ["testword", "testword", "testword"]
    background_all_same = ["testword", "testword", "testword"]

    extractor = ZTestTopwords(
        target_documents=target_all_same,
        background_documents=background_all_same,
        topn=1,
        remove_stopwords=False,
        remove_punct=False,
        remove_digits=False,
        case_sensitive=True,
    )
    extractor()
    assert extractor.to_dict()["topwords"] == []

def test_textacy_keywords_missing_text_or_doc_input():
    """Tests ValueError when neither 'document' is provided in TextacyKeywords."""
    extractor = TextacyKeywords(method="textrank", topn=5)
    with pytest.raises(ValueError, match="The 'document' field must be a string or a spaCy Doc."):
        extractor()