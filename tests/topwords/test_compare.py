"""Tests for compare.py module.

Coverage: 95%: Missing 348-349, 365, 490, 517, 602, 608, 635, 641-642

Last Update: November 8, 2025

Refactored to use instance-based API:
- Comparison now accepts comparison_instance (a pre-configured instance)
- Instead of passing class type + kwargs, users configure an instance first
- Example: ZTest(target_docs=[], comparison_docs=[], topn=3) then pass to Comparison
"""

import pandas as pd
import pytest
import spacy
from pydantic import ValidationError
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.topwords.compare import Comparison
from lexos.topwords.ztest import ZTest

# ---------------- Fixtures ----------------


@pytest.fixture
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "The cat sat on the mat with a hat.",
        "The dog barked loudly at the cat.",
        "A quick brown fox jumps over the lazy dog.",
        "The bird sang sweetly in the morning.",
        "The fish swam quickly through the water.",
    ]


@pytest.fixture
def sample_docs(nlp, sample_texts):
    """Create sample spaCy Doc objects."""
    return [nlp(text) for text in sample_texts]


@pytest.fixture
def class_texts():
    """Create sample class-based texts."""
    return {
        "Shakespeare": [
            "To be or not to be, that is the question.",
            "All the world's a stage, and all the men and women merely players.",
        ],
        "Marlowe": [
            "Is this the face that launched a thousand ships?",
            "Come live with me and be my love.",
        ],
        "Jonson": [
            "Drink to me only with thine eyes.",
            "Soul of the age, the applause, delight, the wonder of our stage.",
        ],
    }


@pytest.fixture
def class_docs(nlp, class_texts):
    """Create sample class-based spaCy Doc objects."""
    return {
        class_name: [nlp(text) for text in texts]
        for class_name, texts in class_texts.items()
    }


@pytest.fixture
def docs_with_author_extension(nlp):
    """Create spaCy Docs with custom author extension."""
    # Set up the custom extension
    if not Doc.has_extension("author"):
        Doc.set_extension("author", default=None, force=True)

    shakespeare_texts = [
        "To be or not to be, that is the question.",
        "All the world's a stage.",
    ]
    marlowe_texts = [
        "Is this the face that launched a thousand ships?",
        "Come live with me and be my love.",
    ]

    docs = []
    for text in shakespeare_texts:
        doc = nlp(text)
        doc._.author = "Shakespeare"
        docs.append(doc)

    for text in marlowe_texts:
        doc = nlp(text)
        doc._.author = "Marlowe"
        docs.append(doc)

    return docs


@pytest.fixture
def mock_comparison_class():
    """Create a mock comparison class for testing."""

    class MockTopWords:
        """Mock TopWords class that returns predictable results."""

        def __init__(self, target_docs, comparison_docs, **kwargs):
            self.target_docs = target_docs
            self.comparison_docs = comparison_docs
            self.kwargs = kwargs

        def __call__(self):
            """Return mock topwords results."""
            return {
                "topwords": [
                    {"term": "word1", "score": 0.95},
                    {"term": "word2", "score": 0.85},
                    {"term": "word3", "score": 0.75},
                ]
            }

    return MockTopWords


# ---------------- Basic Initialization Tests ----------------


class TestComparisonInitialization:
    """Test initialization of Comparison class."""

    def test_comparison_init_with_ztest_instance(self):
        """Test Comparison initialization with ZTest instance."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=10)
        comparison = Comparison(comparison_instance=ztest_instance)

        assert comparison is not None
        assert comparison.comparison_instance == ztest_instance
        assert comparison.output_format == "dict"
        assert comparison.document_labels is None

    def test_comparison_init_with_custom_output_format(self):
        """Test Comparison initialization with custom output format."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=10)
        comparison = Comparison(
            comparison_instance=ztest_instance, output_format="dataframe"
        )

        assert comparison.output_format == "dataframe"

    def test_comparison_init_with_document_labels(self):
        """Test Comparison initialization with document labels."""
        labels = ["Doc A", "Doc B", "Doc C"]
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=10)
        comparison = Comparison(
            comparison_instance=ztest_instance, document_labels=labels
        )

        assert comparison.document_labels == labels

    def test_comparison_instance_configuration_preserved(self):
        """Test that instance configuration is preserved."""
        ztest_instance = ZTest(
            target_docs=[], comparison_docs=[], topn=20, ngrams=2, case_sensitive=False
        )
        comparison = Comparison(comparison_instance=ztest_instance)

        # The instance configuration should be accessible
        assert comparison.comparison_instance.topn == 20
        assert comparison.comparison_instance.ngrams == 2
        assert comparison.comparison_instance.case_sensitive is False


# ---------------- compare_each_doc_to_corpus Tests ----------------


class TestCompareEachDocToCorpus:
    """Test compare_each_doc_to_corpus method."""

    def test_compare_with_string_documents(self, sample_texts):
        """Test comparing each document to corpus with strings."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_doc_to_corpus(sample_texts[:3])

        assert isinstance(result, list)
        assert len(result) == 3
        assert all("label" in r for r in result)
        assert all("topwords" in r for r in result)

    def test_compare_with_doc_objects(self, sample_docs):
        """Test comparing each document to corpus with spaCy Docs."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_doc_to_corpus(sample_docs[:3])

        assert isinstance(result, list)
        assert len(result) == 3

    def test_compare_with_document_labels(self, sample_texts):
        """Test comparing with custom document labels."""
        labels = ["Article 1", "Article 2", "Article 3"]
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(
            comparison_instance=ztest_instance, document_labels=labels
        )

        result = comparison.compare_each_doc_to_corpus(sample_texts[:3])

        assert result[0]["label"] == "Article 1"
        assert result[1]["label"] == "Article 2"
        assert result[2]["label"] == "Article 3"

    def test_compare_with_mismatched_labels(self, sample_texts):
        """Test that mismatched label count raises error."""
        labels = ["Article 1", "Article 2"]  # Only 2 labels for 3 docs
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(
            comparison_instance=ztest_instance, document_labels=labels
        )

        with pytest.raises(LexosException, match="Document labels count"):
            result = comparison.compare_each_doc_to_corpus(sample_texts[:3])

    def test_compare_with_output_format_override(self, sample_texts):
        """Test output format override parameter."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(
            comparison_instance=ztest_instance, output_format="dict"
        )

        result = comparison.compare_each_doc_to_corpus(
            sample_texts[:3], output_format="dataframe"
        )

        assert isinstance(result, pd.DataFrame)
        assert "label" in result.columns
        assert "term" in result.columns


# ---------------- compare_each_doc_to_other_classes Tests ----------------


class TestCompareEachDocToOtherClasses:
    """Test compare_each_doc_to_other_classes method."""

    def test_compare_with_class_documents_dict_strings(self, class_texts):
        """Test comparing with class_documents dict containing strings."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_doc_to_other_classes(
            class_documents=class_texts
        )

        assert isinstance(result, dict)
        assert "Shakespeare" in result
        assert "Marlowe" in result
        assert "Jonson" in result

    def test_compare_with_class_documents_dict_docs(self, class_docs):
        """Test comparing with class_documents dict containing Docs."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_doc_to_other_classes(
            class_documents=class_docs
        )

        assert isinstance(result, dict)

    def test_compare_with_docs_and_class_names(self, docs_with_author_extension):
        """Test comparing using docs and class_names parameters."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_doc_to_other_classes(
            docs=docs_with_author_extension, class_names=["author"]
        )

        assert isinstance(result, dict)
        assert "Shakespeare" in result
        assert "Marlowe" in result

    def test_compare_with_multiple_class_names(self, nlp):
        """Test with multiple class_names (uses first found)."""
        # Set up multiple extensions
        if not Doc.has_extension("author"):
            Doc.set_extension("author", default=None, force=True)
        if not Doc.has_extension("genre"):
            Doc.set_extension("genre", default=None, force=True)

        doc1 = nlp("Text one")
        doc1._.genre = "Poetry"
        doc2 = nlp("Text two")
        doc2._.genre = "Drama"

        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_doc_to_other_classes(
            docs=[doc1, doc2], class_names=["author", "genre"]
        )

        assert isinstance(result, dict)
        assert "Poetry" in result or "Drama" in result

    def test_compare_missing_extension_raises_error(self, nlp):
        """Test that missing extensions raise appropriate error."""
        docs = [nlp("Text one"), nlp("Text two")]
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        with pytest.raises(
            LexosException, match="does not have any of the specified custom extensions"
        ):
            result = comparison.compare_each_doc_to_other_classes(
                docs=docs, class_names=["nonexistent"]
            )

    def test_compare_conflicting_parameters_raises_error(self, class_docs, nlp):
        """Test that providing both class_documents and docs raises error."""
        docs = [nlp("Text one")]
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        with pytest.raises(
            LexosException, match="Cannot provide both 'class_documents' and 'docs'"
        ):
            result = comparison.compare_each_doc_to_other_classes(
                class_documents=class_docs, docs=docs, class_names=["author"]
            )

    def test_compare_docs_without_class_names_raises_error(self, nlp):
        """Test that docs without class_names raises error."""
        docs = [nlp("Text one"), nlp("Text two")]
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        with pytest.raises(LexosException, match="you must also provide 'class_names'"):
            result = comparison.compare_each_doc_to_other_classes(docs=docs)

    def test_compare_empty_docs_raises_error(self):
        """Test that empty docs list raises error."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        with pytest.raises(LexosException, match="Empty list of documents"):
            result = comparison.compare_each_doc_to_other_classes(
                docs=[], class_names=["author"]
            )

    def test_compare_no_parameters_raises_error(self):
        """Test that calling without parameters raises error."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        with pytest.raises(
            LexosException,
            match="You must provide either 'class_documents' .* or both 'docs' and 'class_names'",
        ):
            result = comparison.compare_each_doc_to_other_classes()


# ---------------- compare_each_class_to_other_classes Tests ----------------


class TestCompareEachClassToOtherClasses:
    """Test compare_each_class_to_other_classes method."""

    def test_compare_classes_with_dict_strings(self, class_texts):
        """Test comparing entire classes with string documents."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_class_to_other_classes(
            class_documents=class_texts
        )

        assert isinstance(result, dict)
        assert "Shakespeare" in result
        assert "Marlowe" in result
        assert "Jonson" in result

    def test_compare_classes_with_dict_docs(self, class_docs):
        """Test comparing entire classes with Doc objects."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_class_to_other_classes(
            class_documents=class_docs
        )

        assert isinstance(result, dict)

    def test_compare_classes_with_docs_and_class_names(
        self, docs_with_author_extension
    ):
        """Test comparing classes using docs and class_names."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        result = comparison.compare_each_class_to_other_classes(
            docs=docs_with_author_extension, class_names=["author"]
        )

        assert isinstance(result, dict)
        assert "Shakespeare" in result
        assert "Marlowe" in result

    def test_compare_classes_conflicting_parameters(self, class_docs, nlp):
        """Test that conflicting parameters raise error."""
        docs = [nlp("Text")]
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        comparison = Comparison(comparison_instance=ztest_instance)

        with pytest.raises(
            LexosException, match="Cannot provide both 'class_documents' and 'docs'"
        ):
            result = comparison.compare_each_class_to_other_classes(
                class_documents=class_docs, docs=docs, class_names=["author"]
            )


# ---------------- Helper Methods Tests ----------------


class TestHelperMethods:
    """Test private helper methods.

    These tests create a Comparison instance to access the helper methods.
    """

    @pytest.fixture
    def comparison_instance(self):
        """Create a Comparison instance for testing helper methods."""
        ztest_instance = ZTest(target_docs=[], comparison_docs=[], topn=3)
        return Comparison(comparison_instance=ztest_instance)

    def test_extract_text_from_strings(self, comparison_instance, sample_texts):
        """Test extracting text from string documents."""
        result = comparison_instance._extract_text_from_documents(sample_texts)

        assert result == sample_texts
        assert all(isinstance(text, str) for text in result)

    def test_extract_text_from_docs(self, comparison_instance, sample_docs):
        """Test extracting text from Doc objects."""
        result = comparison_instance._extract_text_from_documents(sample_docs)

        assert len(result) == len(sample_docs)
        assert all(isinstance(text, str) for text in result)

    def test_extract_text_from_mixed_raises_error(self, comparison_instance, nlp):
        """Test that unsupported document types raise error."""
        invalid_docs = ["string", nlp("doc"), 123]  # Invalid: includes integer

        with pytest.raises(LexosException, match="Unsupported document type"):
            result = comparison_instance._extract_text_from_documents(invalid_docs)

    def test_extract_text_from_class_documents(self, comparison_instance, class_texts):
        """Test extracting text from class documents dict."""
        result = comparison_instance._extract_text_from_class_documents(class_texts)

        assert set(result.keys()) == set(class_texts.keys())
        for class_name in class_texts:
            assert len(result[class_name]) == len(class_texts[class_name])

    def test_build_comparison_corpus(self, comparison_instance, sample_texts):
        """Test building comparison corpus excluding one document."""
        result = comparison_instance._build_comparison_corpus(sample_texts, 2)

        assert len(result) == len(sample_texts) - 1
        assert sample_texts[2] not in result
        assert sample_texts[0] in result
        assert sample_texts[1] in result

    def test_build_other_classes_comparison(self, comparison_instance, class_texts):
        """Test building comparison from other classes."""
        text_class_docs = comparison_instance._extract_text_from_class_documents(
            class_texts
        )
        result = comparison_instance._build_other_classes_comparison(
            text_class_docs, "Shakespeare"
        )

        # Should include Marlowe and Jonson, but not Shakespeare
        assert len(result) == len(class_texts["Marlowe"]) + len(class_texts["Jonson"])
        for text in class_texts["Shakespeare"]:
            assert text not in result

    def test_get_document_label_with_labels(self, comparison_instance):
        """Test getting document label when labels provided."""
        labels = ["Article 1", "Article 2", "Article 3"]
        comparison_instance.document_labels = labels

        assert comparison_instance._get_document_label(0) == "Article 1"
        assert comparison_instance._get_document_label(1) == "Article 2"

    def test_get_document_label_without_labels(self, comparison_instance):
        """Test getting document label when no labels provided."""
        comparison_instance.document_labels = None

        assert comparison_instance._get_document_label(0) == "Doc 1"
        assert comparison_instance._get_document_label(5) == "Doc 6"

    def test_get_class_document_label_with_map(self, comparison_instance):
        """Test getting class document label with mapping."""
        doc_map = {"some text": "Custom Label"}
        comparison_instance.document_to_label_map = doc_map

        result = comparison_instance._get_class_document_label(
            "some text", "ClassName", 0
        )
        assert result == "Custom Label"

    def test_get_class_document_label_without_map(self, comparison_instance):
        """Test getting class document label without mapping."""
        comparison_instance.document_to_label_map = {}

        result = comparison_instance._get_class_document_label("text", "Shakespeare", 2)
        assert result == "Shakespeare Doc 3"

    def test_validate_document_labels_success(self, comparison_instance):
        """Test validation succeeds with matching counts."""
        labels = ["A", "B", "C"]
        comparison_instance.document_labels = labels

        # Should not raise
        comparison_instance._validate_document_labels(3)

    def test_validate_document_labels_failure(self, comparison_instance):
        """Test validation fails with mismatched counts."""
        labels = ["A", "B"]
        comparison_instance.document_labels = labels

        with pytest.raises(LexosException, match="Document labels count"):
            comparison_instance._validate_document_labels(3)

    def test_build_class_dict_from_extensions(
        self, comparison_instance, docs_with_author_extension
    ):
        """Test building class dict from Doc extensions."""
        result = comparison_instance._build_class_dict_from_extensions(
            docs_with_author_extension, ["author"]
        )

        assert "Shakespeare" in result
        assert "Marlowe" in result
        assert len(result["Shakespeare"]) == 2
        assert len(result["Marlowe"]) == 2

    def test_build_class_dict_with_underscore_prefix(self, comparison_instance, nlp):
        """Test building class dict with _.extension format."""
        if not Doc.has_extension("category"):
            Doc.set_extension("category", default=None, force=True)

        doc1 = nlp("Text one")
        doc1._.category = "A"
        doc2 = nlp("Text two")
        doc2._.category = "B"

        result = comparison_instance._build_class_dict_from_extensions(
            [doc1, doc2], ["_.category"]
        )

        assert "A" in result
        assert "B" in result

    def test_build_class_dict_with_explicit_underscore_dot_format(
        self, comparison_instance, nlp
    ):
        """Test building class dict explicitly using _.extension to cover the underscore iteration."""
        if not Doc.has_extension("genre"):
            Doc.set_extension("genre", default=None, force=True)

        doc1 = nlp("First document")
        doc1._.genre = "Fiction"
        doc2 = nlp("Second document")
        doc2._.genre = "Non-Fiction"
        doc3 = nlp("Third document")
        doc3._.genre = "Fiction"

        # Use explicit _.genre format
        result = comparison_instance._build_class_dict_from_extensions(
            [doc1, doc2, doc3], ["_.genre"]
        )

        assert "Fiction" in result
        assert "Non-Fiction" in result
        assert len(result["Fiction"]) == 2
        assert len(result["Non-Fiction"]) == 1
        assert doc1 in result["Fiction"]
        assert doc3 in result["Fiction"]
        assert doc2 in result["Non-Fiction"]

    def test_build_class_dict_missing_extension_error(self):
        """Test that LexosException is raised when documents don't have specified extensions."""
        # Create docs without the extension we'll try to access
        nlp = spacy.blank("en")

        # Register a different extension (not the one we'll look for)
        if not Doc.has_extension("other_field"):
            Doc.set_extension("other_field", default=None)

        doc1 = nlp("Text 1")
        doc1._.other_field = "value1"

        doc2 = nlp("Text 2")
        doc2._.other_field = "value2"

        docs = [doc1, doc2]

        # Create a Comparison instance
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        # Try to build class dict with an extension that doesn't exist
        with pytest.raises(LexosException) as exc_info:
            comparison._build_class_dict_from_extensions(docs, ["missing_extension"])

        # Verify the error message mentions the missing extension
        assert "missing_extension" in str(exc_info.value)
        assert "does not have any of the specified custom extensions" in str(
            exc_info.value
        )
        assert "Available extensions:" in str(exc_info.value)
        assert "other_field" in str(exc_info.value)


# ---------------- Output Format Tests ----------------


class TestOutputFormatting:
    """Test output formatting methods.

    These tests create properly initialized Comparison instances for each test.
    """

    def test_format_output_dict(self):
        """Test formatting output as dict."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest, output_format="dict")
        data = {"key": "value"}

        result = comparison._format_output(data)
        assert result == data

    def test_format_output_with_override(self):
        """Test formatting output with format override."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest, output_format="dict")
        data = [
            {
                "label": "Doc 1",
                "topwords": [{"term": "word", "z_score": 0.5}],
            }
        ]

        result = comparison._format_output(data, output_format="dataframe")
        assert isinstance(result, pd.DataFrame)

    def test_to_dataframe_from_list(self):
        """Test converting list results to DataFrame."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = [
            {
                "label": "Doc 1",
                "topwords": [
                    {"term": "word1", "score": 0.9},
                    {"term": "word2", "score": 0.8},
                ],
            },
            {
                "label": "Doc 2",
                "topwords": [{"term": "word3", "score": 0.7}],
            },
        ]

        result = comparison._to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Total of 3 terms across all docs
        assert "label" in result.columns
        assert "term" in result.columns

    def test_to_dataframe_from_dict_class_results(self):
        """Test converting dict class results to DataFrame."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = {
            "Class1": {"topwords": [{"term": "word1", "score": 0.9}]},
            "Class2": {"topwords": [{"term": "word2", "score": 0.8}]},
        }

        result = comparison._to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert "group" in result.columns
        assert "label" in result.columns

    def test_to_dataframe_from_dict_grouped_results(self):
        """Test converting dict grouped results to DataFrame."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = {
            "Group1": [
                {
                    "label": "Doc 1",
                    "result": {"topwords": [{"term": "word1", "score": 0.9}]},
                }
            ],
            "Group2": [
                {
                    "label": "Doc 2",
                    "result": {"topwords": [{"term": "word2", "score": 0.8}]},
                }
            ],
        }

        result = comparison._to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert "group" in result.columns

    def test_to_dataframe_invalid_type_raises_error(self):
        """Test that invalid results type raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="Unsupported results type"):
            result = comparison._to_dataframe("invalid string data")

    def test_to_list_of_dicts_from_list(self):
        """Test converting list to list of dicts."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = [{"key": "value1"}, {"key": "value2"}]

        result = comparison._to_list_of_dicts(data)
        assert result == data

    def test_to_list_of_dicts_from_dict(self):
        """Test converting dict to list of dicts."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = {
            "Group1": [{"label": "Doc 1", "result": {}}],
            "Group2": [{"label": "Doc 2", "result": {}}],
        }

        result = comparison._to_list_of_dicts(data)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)
        assert all("group" in item for item in result)

    def test_to_list_of_dicts_invalid_type_raises_error(self):
        """Test that invalid type raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="Unsupported results type"):
            result = comparison._to_list_of_dicts(123)

    def test_extract_topwords(self):
        """Test extracting topwords from result dict."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = {"topwords": [{"term": "word", "score": 0.9}]}

        result = comparison._extract_topwords(data)
        assert result == [{"term": "word", "score": 0.9}]

    def test_extract_topwords_empty(self):
        """Test extracting topwords from empty result."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        result = comparison._extract_topwords({})
        assert result == []

    def test_extract_label(self):
        """Test extracting label from result dict."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        data = {"label": "Custom Label"}

        result = comparison._extract_label(data, "Default")
        assert result == "Custom Label"

    def test_extract_label_with_default(self):
        """Test extracting label uses default when not present."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        result = comparison._extract_label({}, "Default Label")
        assert result == "Default Label"


# ---------------- Integration-like Tests ----------------


class TestComparisonIntegration:
    """Test integration scenarios.

    These test the _prepare_class_documents helper method with properly
    initialized Comparison instances.
    """

    def test_prepare_class_documents_with_dict(self, class_texts):
        """Test _prepare_class_documents with dict input."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        result = comparison._prepare_class_documents(class_documents=class_texts)

        assert result == class_texts

    def test_prepare_class_documents_with_docs(self, docs_with_author_extension):
        """Test _prepare_class_documents with docs input."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        result = comparison._prepare_class_documents(
            docs=docs_with_author_extension, class_names=["author"]
        )

        assert "Shakespeare" in result
        assert "Marlowe" in result

    def test_prepare_class_documents_conflicting_params(
        self, class_texts, docs_with_author_extension
    ):
        """Test that providing both class_documents and docs raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="Cannot provide both"):
            result = comparison._prepare_class_documents(
                class_documents=class_texts,
                docs=docs_with_author_extension,
                class_names=["author"],
            )

    def test_prepare_class_documents_no_params(self):
        """Test that providing no params raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="You must provide either"):
            result = comparison._prepare_class_documents()

    def test_prepare_class_documents_docs_without_class_names(
        self, docs_with_author_extension
    ):
        """Test that docs without class_names raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="you must also provide 'class_names'"):
            result = comparison._prepare_class_documents(
                docs=docs_with_author_extension
            )

    def test_prepare_class_documents_empty_docs(self):
        """Test that empty docs raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="Empty list of documents"):
            result = comparison._prepare_class_documents(
                docs=[], class_names=["author"]
            )

    def test_prepare_class_documents_non_doc_objects(self):
        """Test that non-Doc objects in docs list raises error."""
        ztest = ZTest(target_docs=[], comparison_docs=[])
        comparison = Comparison(comparison_instance=ztest)

        with pytest.raises(LexosException, match="must be spaCy Doc objects"):
            result = comparison._prepare_class_documents(
                docs=["not a doc"], class_names=["author"]
            )
