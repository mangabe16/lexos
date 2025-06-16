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
    extractor = ZTestTopwords(
        target_texts=target_texts, background_texts=background_texts, topn=5
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
        target_docs=docs, background_texts=background_texts, topn=5, docs=docs
    )
    extractor2()
    assert hasattr(extractor2, "target_docs")
    assert hasattr(docs[0]._, "topwords")
    assert docs[0]._.topwords == extractor2.to_list()


def test_ztest_topwords_empty_input():
    extractor = ZTestTopwords(target_texts=[], background_texts=[], topn=5)
    extractor()
    assert extractor.to_dict()["topwords"] == []


def test_ztest_topwords_large_topn(target_texts, background_texts):
    extractor = ZTestTopwords(
        target_texts=target_texts, background_texts=background_texts, topn=100
    )
    extractor()
    assert len(extractor.to_dict()["topwords"]) <= 100


def test_ztest_topwords_only_stopwords(stopwords_text):
    extractor = ZTestTopwords(
        target_texts=[stopwords_text], background_texts=["some other text"], topn=5
    )
    extractor()
    assert extractor.to_dict()["topwords"] == []


def test_ztest_topwords_repeated_words(repeated_text):
    extractor = ZTestTopwords(
        target_texts=[repeated_text], background_texts=["other words"], topn=3
    )
    extractor()
    tws = extractor.to_dict()["topwords"]
    assert len(tws) <= 3
    if tws:
        assert tws[0]["term"].lower() == "lexos"


def test_ztest_topwords_identical_target_background(identical_texts):
    extractor = ZTestTopwords(
        target_texts=identical_texts, background_texts=identical_texts, topn=5
    )
    extractor()
    assert extractor.to_dict()["topwords"] == []


def test_ztest_topwords_invalid_topn(target_texts, background_texts):
    with pytest.raises(ValidationError):
        ZTestTopwords(
            target_texts=target_texts, background_texts=background_texts, topn=0
        )
    with pytest.raises(ValidationError):
        ZTestTopwords(
            target_texts=target_texts, background_texts=background_texts, topn=-1
        )


def test_ztest_topwords_background_docs_direct_input(
    nlp, target_texts, background_texts
):
    """Test providing `background_docs` directly."""
    target_docs = [nlp(text) for text in target_texts]
    background_docs = [nlp(text) for text in background_texts]
    extractor = ZTestTopwords(
        target_docs=target_docs, background_docs=background_docs, topn=5
    )
    extractor()
    result = extractor.to_dict()
    assert "topwords" in result
    assert isinstance(result["topwords"], list)
    assert len(result["topwords"]) <= 5
    for tw in result["topwords"]:
        assert "term" in tw and "z_score" in tw


def test_ztest_topwords_missing_background_input():
    """Tests the ValueError when neither background_texts nor background_docs are provided."""
    with pytest.raises(
        ValueError,
        match="Either 'background_texts' or 'background_docs' must be provided.",
    ):
        ZTestTopwords(target_texts=["some text"], topn=5)()


def test_ztest_topwords_case_sensitivity_and_line_137(nlp):
    """
    Covers line 137: `token_text = token.lower_`
    Tests both case_sensitive=True and case_sensitive=False scenarios.
    """
    # Adjusted data to make 'apple' a more distinguishing term when case_sensitive=False
    target_texts_case_ci = [
        "Many Apple trees are here.",
        "I like apple pie.",
        "Sweet apple.",
    ]
    background_texts_case_ci = ["A tree is tall.", "I like fruit.", "Orange sweet."]

    # Test case_sensitive=True: "Apple" and "apple" should be distinct
    # For this test, let's make sure 'Apple' is unique to target and 'apple' (lowercase) is not present
    # in the expected top words from background for case sensitive comparison.
    target_cs = ["Apple is a fruit.", "Apple computers."]
    background_cs = ["Eating an apple.", "red apple."]
    extractor_cs = ZTestTopwords(
        target_texts=target_cs,
        background_texts=background_cs,
        topn=5,
        case_sensitive=True,
        remove_stopwords=False,
    )
    extractor_cs()
    cs_topwords = extractor_cs.to_list()
    # "Apple" (capitalized) should be distinguishing for the target.
    # "apple" (lowercase) from the background should have a negative Z-score or not appear in top positive.
    # Assert that 'Apple' (capitalized) is found with a positive Z-score.
    assert any(term == "Apple" for term, score in cs_topwords if score > 0)
    # Assert that 'apple' (lowercase) is NOT found with a positive Z-score.
    # It might have a negative score if it's more common in background, or just not in topN if not significant.
    assert not any(term == "apple" for term, score in cs_topwords if score > 0)

    # Test case_sensitive=False: "Apple" and "apple" should be treated as the same "apple"
    extractor_ci = ZTestTopwords(
        target_texts=target_texts_case_ci,
        background_texts=background_texts_case_ci,
        topn=5,
        case_sensitive=False,  # This is the key line to cover line 137
        remove_stopwords=False,
    )
    extractor_ci()
    ci_topwords = extractor_ci.to_list()
    # When case_sensitive=False, only "apple" (lowercase) should appear in the results
    # and it should be a distinguishing term for the target.
    assert any(term == "apple" for term, score in ci_topwords if score > 0)
    assert not any(
        term == "Apple" for term, _ in ci_topwords
    )  # Ensure no capitalized version


def test_ztest_topwords_missing_target_input():
    """
    Covers line 110: `raise ValueError("Either 'target_texts' or 'target_docs' must be provided.")`
    Tests the ValueError when neither target_texts nor target_docs are provided.
    """
    with pytest.raises(
        ValueError, match="Either 'target_texts' or 'target_docs' must be provided."
    ):
        extractor = ZTestTopwords(background_texts=["some text"], topn=5)
        extractor()  # Calling it triggers the internal checks


def test_ztest_topwords_remove_digits_line_180(nlp):
    """
    Covers line 180: `if self.remove_digits and token.is_digit: continue`
    Tests that digits are correctly removed when `remove_digits` is True.
    """
    target = ["Document with 123 numbers."]
    background = ["Another document with no digits."]
    extractor = ZTestTopwords(
        target_texts=target,
        background_texts=background,
        topn=5,
        remove_digits=True,  # This will activate the check
        remove_stopwords=False,  # Don't remove other things to keep words for comparison
        remove_punct=False,
    )
    extractor()
    result = extractor.to_dict()["topwords"]
    # Assert that no terms contain digits (e.g., "123")
    assert not any(any(char.isdigit() for char in kw["term"]) for kw in result)

    # Verify that if remove_digits is False, numbers *can* be included (optional additional check)
    extractor_keep_digits = ZTestTopwords(
        target_texts=["Document with 123 numbers."],
        background_texts=["Another document with 456 numbers."],
        topn=5,
        remove_digits=False,  # Don't remove digits
        remove_stopwords=False,
        remove_punct=False,
    )
    extractor_keep_digits()
    result_keep_digits = extractor_keep_digits.to_dict()["topwords"]
    # We can't guarantee '123' will be a top word, but we can check if it's potentially present
    assert any(any(char.isdigit() for char in kw["term"]) for kw in result_keep_digits)


def test_ztest_topwords_denominator_is_zero_line_214_216(nlp):
    """
    Covers line 214 (denominator calculation) and line 216 (`z = 0.0` when denominator is 0).
    This case happens when p is exactly 0 or 1.
    We create a scenario where a term is present in 100% of the tokens in both corpora,
    forcing p to be 1, which makes (1-p) be 0, and thus denominator becomes 0.
    """
    # Create a scenario where 'testword' is the ONLY word in both corpora.
    # This makes p1=1, p2=1, p=1, leading to denominator = 0.
    target_all_same = ["testword", "testword", "testword"]
    background_all_same = ["testword", "testword", "testword"]

    extractor = ZTestTopwords(
        target_texts=target_all_same,
        background_texts=background_all_same,
        topn=1,
        remove_stopwords=False,
        remove_punct=False,
        remove_digits=False,
        case_sensitive=True,
    )
    extractor()
    # Because p=1, z should be 0.0 (line 216 is hit).
    # Since we filter out 0.0 Z-scores, the list should be empty.
    assert extractor.to_dict()["topwords"] == []


def test_textacy_keywords_missing_text_or_doc_input():
    """
    Covers `raise ValueError("Either 'text' or 'doc' must be provided.")` in TextacyKeywords.__call__.
    """
    # Instantiate without 'text' or 'doc'
    extractor = TextacyKeywords(method="textrank", topn=5)
    with pytest.raises(ValueError, match="Either 'text' or 'doc' must be provided."):
        extractor()  # Calling it triggers the validation
