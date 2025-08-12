"""test_cloud.py.

Coverage 100%

Last Update: August 12, 2025
"""

import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy
from pydantic import ValidationError

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.cloud import MultiCloud, WordCloud

# Test data
SAMPLE_TEXT = (
    "natural language processing machine learning artificial intelligence data science"
)
SAMPLE_DICT = {
    "natural": 10,
    "language": 8,
    "processing": 6,
    "machine": 4,
    "learning": 2,
}
SAMPLE_TOKENS = ["natural", "language", "processing", "machine", "learning"]
SAMPLE_DOCS = [
    "natural language processing",
    "machine learning algorithms",
    "data science analytics",
    "artificial intelligence systems",
]


class TestWordCloud:
    """Test the WordCloud class."""

    def test_wordcloud_string_initialization(self):
        """Test WordCloud initialization with string data."""
        wc = WordCloud(data=SAMPLE_TEXT)

        assert wc.data == SAMPLE_TEXT
        assert wc.height == 200
        assert wc.width == 200
        assert isinstance(wc.counts, dict)
        assert wc.cloud is not None
        assert len(wc.counts) == 9  # Number of unique words

    def test_wordcloud_dict_initialization(self):
        """Test WordCloud initialization with dictionary data."""
        wc = WordCloud(data=SAMPLE_DICT)

        assert wc.data == SAMPLE_DICT
        assert wc.counts == SAMPLE_DICT
        assert wc.cloud is not None

    def test_wordcloud_list_initialization(self):
        """Test WordCloud initialization with list data."""
        wc = WordCloud(data=SAMPLE_TOKENS)

        expected_counts = Counter(SAMPLE_TOKENS)
        assert wc.counts == dict(expected_counts)
        assert wc.cloud is not None

    def test_wordcloud_custom_dimensions(self):
        """Test WordCloud with custom dimensions."""
        wc = WordCloud(data=SAMPLE_TEXT, height=400, width=600)

        assert wc.height == 400
        assert wc.width == 600
        assert wc.opts["height"] == 400
        assert wc.opts["width"] == 600

    def test_wordcloud_with_limit(self):
        """Test WordCloud with term limit."""
        wc = WordCloud(data=SAMPLE_DICT, limit=3)

        assert len(wc.counts) <= 3
        # Should contain the top 3 most frequent terms
        top_terms = sorted(SAMPLE_DICT.items(), key=lambda x: x[1], reverse=True)[:3]
        for term, count in top_terms:
            assert term in wc.counts

    def test_wordcloud_with_title(self):
        """Test WordCloud with title."""
        title = "Test Word Cloud"
        wc = WordCloud(data=SAMPLE_TEXT, title=title)

        assert wc.title == title

    def test_wordcloud_with_custom_opts(self):
        """Test WordCloud with custom WordCloud options."""
        custom_opts = {
            "background_color": "black",
            "max_words": 100,
            "colormap": "viridis",
        }
        wc = WordCloud(data=SAMPLE_TEXT, opts=custom_opts)

        assert wc.opts["background_color"] == "black"
        assert wc.opts["max_words"] == 100
        assert wc.opts["colormap"] == "viridis"
        # Should still have height and width set
        assert "height" in wc.opts
        assert "width" in wc.opts

    def test_wordcloud_with_round_mask(self):
        """Test WordCloud with circular mask."""
        wc = WordCloud(data=SAMPLE_TEXT, round=100)

        assert wc.round == 100
        assert "mask" in wc.opts
        assert isinstance(wc.opts["mask"], np.ndarray)

    def test_wordcloud_with_figure_opts(self):
        """Test WordCloud with figure options."""
        figure_opts = {"figsize": (10, 8), "dpi": 150}
        wc = WordCloud(data=SAMPLE_TEXT, figure_opts=figure_opts)

        assert wc.figure_opts == figure_opts

    def test_wordcloud_empty_data(self):
        """Test WordCloud with empty data."""
        with pytest.raises(ValueError):
            wc = WordCloud(data="")

    def test_wordcloud_with_dataframe(self):
        """Test WordCloud with DataFrame data."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 3], "doc2": [1, 3, 0], "doc3": [0, 2, 1]},
            index=["term1", "term2", "term3"],
        )

        wc = WordCloud(data=df)

        assert isinstance(wc.counts, dict)
        assert wc.cloud is not None

    def test_wordcloud_with_spacy_doc(self, nlp):
        """Test WordCloud with spaCy Doc object."""
        doc = nlp(SAMPLE_TEXT)

        wc = WordCloud(data=doc)

        expected_counts = Counter([token.text for token in doc])
        assert wc.counts == dict(expected_counts)
        assert wc.cloud is not None

    def test_wordcloud_save(self):
        """Test saving WordCloud to file."""
        wc = WordCloud(data=SAMPLE_TEXT)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            wc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_wordcloud_save_no_cloud(self):
        """Test saving when no cloud object exists."""
        wc = WordCloud(data=SAMPLE_TEXT)
        wc.cloud = None  # Manually set to None

        with pytest.raises(LexosException) as exc_info:
            wc.save("test.png")

        assert "No WordCloud object to save" in str(exc_info.value)

    def test_wordcloud_show(self):
        """Test showing WordCloud without title."""
        wc = WordCloud(data=SAMPLE_TEXT)

        with (
            patch("matplotlib.pyplot.imshow") as mock_imshow,
            patch("matplotlib.pyplot.axis") as mock_axis,
        ):
            wc.show()
            mock_imshow.assert_called_once_with(wc.cloud)
            mock_axis.assert_called_once_with("off")

    def test_wordcloud_show_with_title(self):
        """Test showing WordCloud with title."""
        title = "Test Title"
        wc = WordCloud(data=SAMPLE_TEXT, title=title)

        with (
            patch("matplotlib.pyplot.imshow") as mock_imshow,
            patch("matplotlib.pyplot.axis") as mock_axis,
            patch("matplotlib.pyplot.title") as mock_title,
        ):
            wc.show()
            mock_imshow.assert_called_once_with(wc.cloud)
            mock_axis.assert_called_once_with("off")
            mock_title.assert_called_once_with(title)


class TestMultiCloud:
    """Test the MultiCloud class."""

    def test_multicloud_list_initialization(self):
        """Test MultiCloud initialization with list of documents."""
        mc = MultiCloud(data=SAMPLE_DOCS)

        assert len(mc.clouds) == 4
        assert mc.ncols == 3
        assert all(isinstance(cloud, WordCloud) for cloud in mc.clouds)

    def test_multicloud_custom_ncols(self):
        """Test MultiCloud with custom number of columns."""
        mc = MultiCloud(data=SAMPLE_DOCS, ncols=2)

        assert mc.ncols == 2
        assert len(mc.clouds) == 4

    def test_multicloud_with_limit(self):
        """Test MultiCloud with term limit."""
        mc = MultiCloud(data=SAMPLE_DOCS, limit=5)

        assert mc.limit == 5
        # Each cloud should respect the limit
        for cloud in mc.clouds:
            assert len(cloud.counts) <= 5

    def test_multicloud_with_custom_dimensions(self):
        """Test MultiCloud with custom dimensions."""
        mc = MultiCloud(data=SAMPLE_DOCS, height=300, width=400)

        assert mc.height == 300
        assert mc.width == 400
        # Each cloud should have the specified dimensions
        for cloud in mc.clouds:
            assert cloud.height == 300
            assert cloud.width == 400

    def test_multicloud_with_title_and_labels(self):
        """Test MultiCloud with title and labels."""
        title = "Multiple Word Clouds"
        labels = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        mc = MultiCloud(data=SAMPLE_DOCS, title=title, labels=labels)

        assert mc.title == title
        assert mc.labels == labels

    def test_multicloud_with_custom_opts(self):
        """Test MultiCloud with custom WordCloud options."""
        custom_opts = {"background_color": "black", "max_words": 50}
        mc = MultiCloud(data=SAMPLE_DOCS, opts=custom_opts)

        # Each cloud should have the custom options
        for cloud in mc.clouds:
            assert cloud.opts["background_color"] == "black"
            assert cloud.opts["max_words"] == 50

    def test_multicloud_with_round_mask(self):
        """Test MultiCloud with circular mask."""
        mc = MultiCloud(data=SAMPLE_DOCS, round=150)

        assert mc.round == 150
        # Each cloud should have the mask
        for cloud in mc.clouds:
            assert "mask" in cloud.opts

    def test_multicloud_with_custom_padding(self):
        """Test MultiCloud with custom padding."""
        mc = MultiCloud(data=SAMPLE_DOCS, padding=0.5)

        assert mc.padding == 0.5

    def test_multicloud_dataframe_processing(self):
        """Test MultiCloud with DataFrame data."""
        # Create a document-term matrix
        df = pd.DataFrame(
            {"term1": [2, 1, 0, 3], "term2": [1, 3, 2, 0], "term3": [0, 1, 1, 2]},
            index=["doc1", "doc2", "doc3", "doc4"],
        )

        mc = MultiCloud(data=df)

        assert len(mc.clouds) == 4  # One for each row
        assert all(isinstance(cloud, WordCloud) for cloud in mc.clouds)

    def test_multicloud_dataframe_with_doc_selection(self):
        """Test MultiCloud with DataFrame and document selection."""
        df = pd.DataFrame(
            {"term1": [2, 1, 0, 3], "term2": [1, 3, 2, 0], "term3": [0, 1, 1, 2]},
            index=["doc1", "doc2", "doc3", "doc4"],
        )

        mc = MultiCloud(data=df, docs=[0, 2])  # Select first and third docs

        assert len(mc.clouds) == 2

    def test_multicloud_dtm_processing(self):
        """Test MultiCloud with DTM data."""
        data = [
            ["term1", "term1"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])
        mc = MultiCloud(data=dtm)

        assert len(mc.clouds) == 3  # One for each document
        assert all(isinstance(cloud, WordCloud) for cloud in mc.clouds)

    def test_multicloud_with_dtm_object(self):
        """Test MultiCloud with DTM object to cover DTM processing path."""
        # Create sample documents
        docs = [
            ["natural", "language", "processing"],
            ["machine", "learning", "algorithms"],
            ["data", "science", "analytics"],
        ]

        # Create a DTM object
        dtm = DTM()
        dtm(docs=docs, labels=["doc1", "doc2", "doc3"])

        # Create MultiCloud with DTM - this will hit line 226
        mc = MultiCloud(data=dtm)

        assert len(mc.clouds) == 3
        assert all(isinstance(cloud, WordCloud) for cloud in mc.clouds)

        # Verify the DTM processing worked correctly
        assert "natural" in mc.clouds[0].counts
        assert "machine" in mc.clouds[1].counts
        assert "data" in mc.clouds[2].counts

    def test_multicloud_dtm_with_zero_counts(self):
        """Test MultiCloud DTM processing with zero counts to cover the filtering logic."""
        # Create DTM with some zero entries to test the count_value > 0 condition
        docs = [
            ["term1", "term1", "term2"],  # term1=2, term2=1, term3=0
            ["term2", "term3", "term3"],  # term1=0, term2=1, term3=2
            ["term1", "term3"],  # term1=1, term2=0, term3=1
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["doc1", "doc2", "doc3"])

        # This will process the DTM and hit line 226 multiple times
        # including cases where count_value > 0 is False (zero counts)
        mc = MultiCloud(data=dtm)

        assert len(mc.clouds) == 3

        # Verify zero counts are filtered out (line 226 condition)
        # doc1 should not have term3 (count=0)
        assert "term3" not in mc.clouds[0].counts
        # doc2 should not have term1 (count=0)
        assert "term1" not in mc.clouds[1].counts
        # doc3 should not have term2 (count=0)
        assert "term2" not in mc.clouds[2].counts

    def test_multicloud_dtm_single_doc_by_index(self):
        """Test MultiCloud with DTM selecting single document by index."""
        docs = [
            ["term1", "term2", "term3"],
            ["term4", "term5", "term6"],
            ["term7", "term8", "term9"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["doc1", "doc2", "doc3"])

        # Pass single integer instead of list - this hits line 226
        mc = MultiCloud(data=dtm, docs=1)  # Select just second document

        assert len(mc.clouds) == 1
        assert "term4" in mc.clouds[0].counts

    def test_multicloud_dtm_single_doc_by_label(self):
        """Test MultiCloud with DTM selecting single document by label."""
        docs = [
            ["apple", "banana", "cherry"],
            ["dog", "elephant", "fox"],
            ["guitar", "harmonica", "instrument"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["fruits", "animals", "music"])

        # Pass single string instead of list - this hits line 226
        mc = MultiCloud(data=dtm, docs="animals")  # Select just one document

        assert len(mc.clouds) == 1
        assert "dog" in mc.clouds[0].counts

    def test_multicloud_dataframe_single_doc_by_index(self):
        """Test MultiCloud with DataFrame selecting single document by index."""
        df = pd.DataFrame(
            {"term1": [2, 0, 1], "term2": [1, 3, 0], "term3": [0, 1, 2]},
            index=["doc1", "doc2", "doc3"],
        )

        # Pass single integer - this hits line 226 in DataFrame section
        mc = MultiCloud(data=df, docs=0)  # Select just first document

        assert len(mc.clouds) == 1

    def test_multicloud_dataframe_single_doc_by_label(self):
        """Test MultiCloud with DataFrame selecting single document by label."""
        df = pd.DataFrame(
            {"word1": [5, 0, 2], "word2": [1, 4, 0], "word3": [0, 2, 3]},
            index=["document_a", "document_b", "document_c"],
        )

        # Pass single string - this hits line 226 in DataFrame section
        mc = MultiCloud(data=df, docs="document_b")  # Select just one document

        assert len(mc.clouds) == 1
        assert "word2" in mc.clouds[0].counts

    def test_multicloud_dtm_single_document_total(self):
        """Test MultiCloud with DTM that has only one document total."""
        docs = [["only", "one", "document", "here"]]

        dtm = DTM()
        dtm(docs=docs, labels=["single_doc"])

        # When docs=None and there's only 1 document, range(1) creates range with 1 item
        # This could potentially hit the isinstance check depending on implementation
        mc = MultiCloud(
            data=dtm
        )  # docs=None, so uses range(1) which might be treated as single item

        assert len(mc.clouds) == 1
        assert "only" in mc.clouds[0].counts

    def test_multicloud_dtm_with_doc_selection(self):
        """Test MultiCloud with DTM and document selection."""
        data = [
            ["term1", "term1"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])
        mc = MultiCloud(data=dtm, docs=[0, 2])  # Select first and third docs

        assert len(mc.clouds) == 2

    def test_multicloud_unsupported_data_type(self):
        """Test MultiCloud with unsupported data type."""
        with pytest.raises(ValidationError) as exc_info:
            MultiCloud(data=12345)

    def test_multicloud_show_without_title(self):
        """Test MultiCloud show method without title."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])  # Use fewer docs for simplicity

        fig = mc.show()

        assert fig is not None
        assert mc.fig is not None
        assert isinstance(fig, plt.Figure)

    def test_multicloud_show_with_title(self):
        """Test MultiCloud show method with title."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2], title="Test Multi-Cloud")

        fig = mc.show()

        assert fig is not None
        assert mc.fig is not None
        assert mc.title == "Test Multi-Cloud"

    def test_multicloud_show_with_labels(self):
        """Test MultiCloud show method with labels."""
        labels = ["First Doc", "Second Doc"]
        mc = MultiCloud(data=SAMPLE_DOCS[:2], labels=labels)

        fig = mc.show()

        assert fig is not None
        assert mc.labels == labels

    def test_multicloud_show_single_column(self):
        """Test MultiCloud with single column layout."""
        mc = MultiCloud(data=SAMPLE_DOCS[:3], ncols=1)

        fig = mc.show()

        assert fig is not None
        assert mc.ncols == 1

    def test_multicloud_show_single_row(self):
        """Test MultiCloud with single row layout."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2], ncols=5)  # More columns than data

        fig = mc.show()

        assert fig is not None

    def test_multicloud_save_without_figure(self):
        """Test MultiCloud save without existing figure."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            mc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multicloud_save_with_existing_figure(self):
        """Test MultiCloud save with existing figure."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])
        mc.show()  # Generate figure first

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            mc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multicloud_get_clouds(self):
        """Test MultiCloud get_clouds method."""
        mc = MultiCloud(data=SAMPLE_DOCS)

        clouds = mc.get_clouds()

        assert clouds == mc.clouds
        assert len(clouds) == 4
        assert all(isinstance(cloud, WordCloud) for cloud in clouds)

    def test_multicloud_empty_data(self):
        """Test MultiCloud with empty list."""
        with pytest.raises(LexosException):
            mc = MultiCloud(data=[])

    def test_multicloud_empty_dtm(self):
        """Test MultiCloud with empty DTM."""
        with pytest.raises(LexosException):
            dtm = DTM()
            mc = MultiCloud(data=dtm)

    def test_multicloud_single_document(self):
        """Test MultiCloud with single document."""
        mc = MultiCloud(data=[SAMPLE_DOCS[0]])

        assert len(mc.clouds) == 1
        assert isinstance(mc.clouds[0], WordCloud)

    def test_multicloud_spacy_docs(self, nlp):
        """Test MultiCloud with spaCy Doc objects."""
        docs = [nlp(text) for text in SAMPLE_DOCS]

        mc = MultiCloud(data=docs)

        assert len(mc.clouds) == 4
        assert all(isinstance(cloud, WordCloud) for cloud in mc.clouds)

    def test_multicloud_figure_options(self):
        """Test MultiCloud with custom figure options."""
        figure_opts = {"figsize": (12, 8), "dpi": 100}
        mc = MultiCloud(data=SAMPLE_DOCS[:2], figure_opts=figure_opts)

        assert mc.figure_opts == figure_opts

    def test_multicloud_process_data_dtm_string_docs(self):
        """Test MultiCloud DTM processing with string document selection."""
        data = [
            ["term1", "term1"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]
        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])

        mc = MultiCloud(data=dtm, docs=["doc1", "doc3"])

        assert len(mc.clouds) == 2

    def test_multicloud_process_data_dataframe_string_docs(self):
        """Test MultiCloud DataFrame processing with string document selection."""
        df = pd.DataFrame(
            {"term1": [2, 1, 0], "term2": [1, 3, 2], "term3": [0, 1, 1]},
            index=["doc1", "doc2", "doc3"],
        )

        mc = MultiCloud(data=df, docs=["doc1", "doc3"])

        assert len(mc.clouds) == 2

    def test_multicloud_creation_failure(self):
        """Test MultiCloud when WordCloud creation fails."""
        # Create data that will cause WordCloud creation to fail
        invalid_data = [{"": 0}]  # Empty term with zero count

        with pytest.raises(ValidationError):
            MultiCloud(data=invalid_data)

    def test_multicloud_empty_dataframe(self):
        """Test MultiCloud with empty DataFrame to cover line 253."""
        # Create an empty DataFrame
        empty_df = pd.DataFrame()

        # This should raise LexosException with "Empty DataFrame provided."
        with pytest.raises(LexosException) as exc_info:
            MultiCloud(data=empty_df)

        assert "Empty DataFrame provided" in str(exc_info.value)

    def test_multicloud_dataframe_no_rows(self):
        """Test MultiCloud with DataFrame that has columns but no rows."""
        # Create DataFrame with columns but no data
        empty_rows_df = pd.DataFrame(columns=["term1", "term2", "term3"])

        # This should also trigger the empty check
        with pytest.raises(LexosException) as exc_info:
            MultiCloud(data=empty_rows_df)

        assert "Empty DataFrame provided" in str(exc_info.value)

    def test_multicloud_dataframe_no_columns(self):
        """Test MultiCloud with DataFrame that has rows but no columns."""
        # Create DataFrame with index but no columns
        empty_cols_df = pd.DataFrame(index=["doc1", "doc2"])

        # This should also trigger the empty check
        with pytest.raises(LexosException) as exc_info:
            MultiCloud(data=empty_cols_df)

        assert "Empty DataFrame provided" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_wordcloud_invalid_dimensions(self):
        """Test WordCloud with invalid dimensions."""
        with pytest.raises(ValueError):
            WordCloud(data=SAMPLE_TEXT, height=10)  # Too small

        with pytest.raises(ValueError):
            WordCloud(data=SAMPLE_TEXT, width=10)  # Too small

    def test_multicloud_invalid_ncols(self):
        """Test MultiCloud with invalid ncols."""
        with pytest.raises(ValueError):
            MultiCloud(data=SAMPLE_DOCS, ncols=0)

        with pytest.raises(ValueError):
            MultiCloud(data=SAMPLE_DOCS, ncols=-1)

    def test_multicloud_invalid_padding(self):
        """Test MultiCloud with invalid padding."""
        with pytest.raises(ValueError):
            MultiCloud(data=SAMPLE_DOCS, padding=-0.1)

        with pytest.raises(ValueError):
            MultiCloud(data=SAMPLE_DOCS, padding=1.1)

    def test_wordcloud_very_large_round(self):
        """Test WordCloud with very large round value."""
        wc = WordCloud(data=SAMPLE_TEXT, round=1000)

        assert wc.round == 1000
        assert "mask" in wc.opts

    def test_multicloud_no_figure_save_error(self):
        """Test MultiCloud save when figure generation fails."""
        mc = MultiCloud(data=SAMPLE_DOCS[:1])
        mc.fig = None
        with pytest.raises(LexosException):
            mc.save("test.png")


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_wordcloud_complete_workflow(self):
        """Test complete WordCloud workflow."""
        # Create WordCloud
        wc = WordCloud(
            data=SAMPLE_DICT, title="Integration Test", height=300, width=400, limit=5
        )

        # Verify creation
        assert wc.cloud is not None
        assert len(wc.counts) <= 5
        assert wc.title == "Integration Test"

        # Test save
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            wc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multicloud_complete_workflow(self):
        """Test complete MultiCloud workflow."""
        # Create MultiCloud
        mc = MultiCloud(
            data=SAMPLE_DOCS,
            title="Multi-Cloud Integration Test",
            labels=["NLP", "ML", "DS", "AI"],
            ncols=2,
            limit=3,
        )

        # Verify creation
        assert len(mc.clouds) == 4
        assert mc.title == "Multi-Cloud Integration Test"
        assert mc.labels == ["NLP", "ML", "DS", "AI"]

        # Generate figure
        fig = mc.show()
        assert fig is not None

        # Test save
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            mc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# Additional test functions for MultiCloud class


class TestMultiCloudAdditional:
    """Additional tests for MultiCloud class focusing on edge cases and specific scenarios."""

    def test_multicloud_with_nested_lists(self):
        """Test MultiCloud with nested list structure."""
        nested_data = [
            ["natural", "language", "processing", "natural"],
            ["machine", "learning", "algorithms", "machine", "learning"],
            ["data", "science", "analytics", "data"],
        ]

        mc = MultiCloud(data=nested_data)

        assert len(mc.clouds) == 3
        # Check that duplicate words are counted
        assert mc.clouds[0].counts["natural"] == 2
        assert mc.clouds[1].counts["machine"] == 2
        assert mc.clouds[1].counts["learning"] == 2

    def test_multicloud_dtm_with_zero_counts(self):
        """Test MultiCloud with DTM containing zero counts."""
        # Create DTM with some zero entries
        data = [
            ["term1", "term1", "term2"],  # doc1: term1=2, term2=1, term3=0
            ["term2", "term2", "term3"],  # doc2: term1=0, term2=2, term3=1
            ["term1", "term3", "term3"],  # doc3: term1=1, term2=0, term3=2
        ]

        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])
        mc = MultiCloud(data=dtm)

        assert len(mc.clouds) == 3
        # Verify zero counts are excluded
        assert "term3" not in mc.clouds[0].counts  # doc1 has 0 term3
        assert "term1" not in mc.clouds[1].counts  # doc2 has 0 term1
        assert "term2" not in mc.clouds[2].counts  # doc3 has 0 term2

    def test_multicloud_with_very_long_document_names(self):
        """Test MultiCloud with very long document names."""
        long_name = "very_long_document_name_" * 10
        df = pd.DataFrame(
            {"term1": [2, 1], "term2": [1, 3]}, index=["short_doc", long_name]
        )

        mc = MultiCloud(data=df, docs=[long_name])

        assert len(mc.clouds) == 1
        assert mc.clouds[0].counts == {"term1": 1, "term2": 3}

    def test_multicloud_with_unicode_terms(self):
        """Test MultiCloud with Unicode characters in terms."""
        unicode_data = ["café résumé naïve", "数据 科学 机器学习", "café résumé データ"]

        mc = MultiCloud(data=unicode_data)

        assert len(mc.clouds) == 3
        assert "café" in mc.clouds[0].counts
        assert "数据" in mc.clouds[1].counts
        assert "データ" in mc.clouds[2].counts

    def test_multicloud_large_number_of_documents(self):
        """Test MultiCloud with a large number of documents."""
        # Create 20 small documents
        large_data = [f"doc{i} term{i} word{i}" for i in range(20)]

        mc = MultiCloud(data=large_data, ncols=5)

        assert len(mc.clouds) == 20
        assert mc.ncols == 5

        # Test figure generation with many subplots
        fig = mc.show()
        assert fig is not None

    def test_multicloud_with_single_term_documents(self):
        """Test MultiCloud where each document has only one term."""
        single_term_data = [
            ["term1", "term1", "term1", "term1", "term1"],
            ["term2", "term2", "term2"],
            ["term3", "term3", "term3", "term3", "term3", "term3", "term3"],
        ]

        mc = MultiCloud(data=single_term_data)

        assert len(mc.clouds) == 3
        assert len(mc.clouds[0].counts) == 1
        assert len(mc.clouds[1].counts) == 1
        assert len(mc.clouds[2].counts) == 1

    def test_multicloud_with_identical_documents(self):
        """Test MultiCloud with identical documents."""
        identical_docs = ["same text content"] * 3

        mc = MultiCloud(data=identical_docs)

        assert len(mc.clouds) == 3
        # All clouds should have identical counts
        for cloud in mc.clouds:
            assert cloud.counts == {"same": 1, "text": 1, "content": 1}

    def test_multicloud_dtm_column_selection_by_name(self):
        """Test MultiCloud DTM with column selection by name."""
        data = [["term1", "term2"], ["term1", "term3"], ["term2", "term3"]]

        dtm = DTM()
        dtm(docs=data, labels=["doc_a", "doc_b", "doc_c"])

        mc = MultiCloud(data=dtm, docs=["doc_a", "doc_c"])

        assert len(mc.clouds) == 2

    def test_multicloud_dataframe_all_zero_document(self):
        """Test MultiCloud with DataFrame where one document has all zeros."""
        df = pd.DataFrame(
            {"term1": [2, 0, 1], "term2": [1, 0, 3], "term3": [0, 0, 1]},
            index=["doc1", "doc2", "doc3"],
        )

        with pytest.raises(LexosException):
            mc = MultiCloud(data=df)

    def test_multicloud_with_mixed_case_terms(self):
        """Test MultiCloud with mixed case terms."""
        mixed_case_data = [
            "Natural Language Processing",
            "MACHINE learning algorithms",
            "Data Science Analytics",
        ]

        mc = MultiCloud(data=mixed_case_data)

        assert len(mc.clouds) == 3
        # Verify case is preserved
        assert "Natural" in mc.clouds[0].counts
        assert "MACHINE" in mc.clouds[1].counts

    def test_multicloud_with_numeric_string_terms(self):
        """Test MultiCloud with numeric strings as terms."""
        numeric_data = [
            ["123", "456", "789"],
            ["123", "abc", "456"],
            ["xyz", "789", "123"],
        ]

        mc = MultiCloud(data=numeric_data)

        assert len(mc.clouds) == 3
        assert "123" in mc.clouds[0].counts
        assert "456" in mc.clouds[1].counts

    def test_multicloud_figure_generation_edge_cases(self):
        """Test MultiCloud figure generation with edge cases."""
        # Test with exactly 1 document (single subplot)
        mc_single = MultiCloud(data=[SAMPLE_DOCS[0]])
        fig_single = mc_single.show()
        assert fig_single is not None

        # Test with exactly 2 documents and ncols=1 (single column)
        mc_column = MultiCloud(data=SAMPLE_DOCS[:2], ncols=1)
        fig_column = mc_column.show()
        assert fig_column is not None

        # Test with more columns than documents
        mc_wide = MultiCloud(data=SAMPLE_DOCS[:2], ncols=10)
        fig_wide = mc_wide.show()
        assert fig_wide is not None

    def test_multicloud_with_custom_wordcloud_parameters(self):
        """Test MultiCloud with various WordCloud parameter combinations."""
        mc = MultiCloud(
            data=SAMPLE_DOCS[:2],
            height=150,
            width=250,
            limit=10,
            round=50,
            opts={
                "background_color": "white",
                "max_words": 20,
                "relative_scaling": 0.5,
                "min_font_size": 8,
            },
        )

        assert len(mc.clouds) == 2
        for cloud in mc.clouds:
            assert cloud.height == 150
            assert cloud.width == 250
            assert cloud.opts["background_color"] == "white"
            assert cloud.opts["max_words"] == 20

    def test_multicloud_with_empty_documents_in_list(self):
        """Test MultiCloud with some empty documents in the list."""
        mixed_data = [
            "natural language processing",
            "",  # Empty document
            "machine learning",
            "   ",  # Whitespace only
            "data science",
        ]

        with pytest.raises(LexosException):
            mc = MultiCloud(data=mixed_data)

    def test_multicloud_spacy_docs_with_custom_processing(self, nlp):
        """Test MultiCloud with spaCy docs using custom token processing."""
        # Create docs with punctuation and stopwords
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Data science combines statistics, programming, and domain expertise.",
        ]

        docs = [nlp(text) for text in texts]
        mc = MultiCloud(data=docs)

        assert len(mc.clouds) == 3
        # Should include all tokens (including punctuation and stopwords)
        assert "." in mc.clouds[0].counts  # Punctuation included
        assert "the" in mc.clouds[0].counts  # Stopwords included

    def test_multicloud_with_very_short_texts(self):
        """Test MultiCloud with very short texts."""
        short_texts = ["a", "b", "c", "d", "e"]

        mc = MultiCloud(data=short_texts)

        assert len(mc.clouds) == 5
        for i, cloud in enumerate(mc.clouds):
            expected_char = chr(ord("a") + i)
            assert cloud.counts == {expected_char: 1}

    def test_multicloud_dtm_document_ordering(self):
        """Test that MultiCloud preserves document ordering from DTM."""
        data = [["alpha", "beta"], ["gamma", "delta"], ["epsilon", "zeta"]]

        dtm = DTM()
        dtm(docs=data, labels=["first", "second", "third"])
        mc = MultiCloud(data=dtm)

        assert len(mc.clouds) == 3
        assert "alpha" in mc.clouds[0].counts  # First document
        assert "gamma" in mc.clouds[1].counts  # Second document
        assert "epsilon" in mc.clouds[2].counts  # Third document

    def test_multicloud_save_different_formats(self):
        """Test MultiCloud save with different file formats."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        formats = [".png", ".jpg", ".pdf", ".svg"]

        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                mc.save(tmp_path)
                assert Path(tmp_path).exists()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def test_multicloud_performance_with_large_vocabulary(self):
        """Test MultiCloud performance with documents having large vocabularies."""
        # Create documents with many unique terms
        large_vocab_docs = []
        for i in range(3):
            # Each doc has 50 unique terms
            terms = [f"term{j}_{i}" for j in range(50)]
            large_vocab_docs.append(" ".join(terms))

        mc = MultiCloud(data=large_vocab_docs, limit=20)  # Limit to manageable size

        assert len(mc.clouds) == 3
        for cloud in mc.clouds:
            assert len(cloud.counts) <= 20  # Should respect limit

    def test_multicloud_with_special_characters_in_terms(self):
        """Test MultiCloud with special characters in terms."""
        special_char_data = [
            "hello@world.com user#123 $money",
            "test-case under_score file.txt",
            "question? answer! end.",
        ]

        mc = MultiCloud(data=special_char_data)

        assert len(mc.clouds) == 3
        # Verify special characters are preserved
        assert "hello@world.com" in mc.clouds[0].counts
        assert "test-case" in mc.clouds[1].counts
        assert "question?" in mc.clouds[2].counts

    def test_multicloud_labels_longer_than_docs(self):
        """Test MultiCloud with more labels than documents."""
        mc = MultiCloud(
            data=SAMPLE_DOCS[:2], labels=["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"]
        )

        assert len(mc.clouds) == 2
        # Should only use first 2 labels
        fig = mc.show()
        assert fig is not None

    def test_multicloud_labels_shorter_than_docs(self):
        """Test MultiCloud with fewer labels than documents."""
        mc = MultiCloud(
            data=SAMPLE_DOCS,
            labels=["First", "Second"],  # Only 2 labels for 4 docs
        )

        assert len(mc.clouds) == 4
        # Should use provided labels for first 2, then default for rest
        fig = mc.show()
        assert fig is not None

    def test_multicloud_extreme_ncols_values(self):
        """Test MultiCloud with extreme ncols values."""
        # Very large ncols (larger than number of docs)
        mc_large = MultiCloud(data=SAMPLE_DOCS[:2], ncols=100)
        fig_large = mc_large.show()
        assert fig_large is not None

        # ncols = 1 (single column layout)
        mc_single = MultiCloud(data=SAMPLE_DOCS, ncols=1)
        fig_single = mc_single.show()
        assert fig_single is not None

    def test_multicloud_padding_edge_values(self):
        """Test MultiCloud with edge padding values."""
        # Minimum padding
        mc_min = MultiCloud(data=SAMPLE_DOCS[:2], padding=0.0)
        assert mc_min.padding == 0.0

        # Maximum padding
        mc_max = MultiCloud(data=SAMPLE_DOCS[:2], padding=1.0)
        assert mc_max.padding == 1.0

        # Both should generate figures successfully
        fig_min = mc_min.show()
        fig_max = mc_max.show()
        assert fig_min is not None
        assert fig_max is not None


# Fixtures
@pytest.fixture
def nlp():
    """Fixture spaCy pipeline."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    return pd.DataFrame(
        {"doc1": [2, 1, 0, 3], "doc2": [1, 3, 2, 0], "doc3": [0, 1, 1, 2]},
        index=["term1", "term2", "term3", "term4"],
    )


@pytest.fixture
def sample_dtm(sample_dataframe):
    """Fixture providing a sample DTM."""
    return DTM(sample_dataframe.T)  # Transpose for DTM format


@pytest.fixture
def cleanup_plots():
    """Fixture to cleanup matplotlib plots after tests."""
    yield
    plt.close("all")


# Use the cleanup fixture in tests that create plots
pytestmark = pytest.mark.usefixtures("cleanup_plots")


class TestMultiCloudShow:
    def test_multicloud_show_method(self):
        """Test MultiCloud show() method."""
        # Test with valid figure
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        # The figure should be created during initialization
        assert mc.fig is not None

        # show() should return the figure
        fig = mc.show()
        assert fig is not None
        assert fig == mc.fig
        assert isinstance(fig, plt.Figure)

    def test_multicloud_show_method_no_figure(self):
        """Test MultiCloud show() method when no figure exists."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        # Manually set fig to None to simulate missing figure
        mc.fig = None

        # show() should raise LexosException
        with pytest.raises(LexosException) as exc_info:
            mc.show()

        assert "No figure to show" in str(exc_info.value)

    def test_multicloud_show_method_with_title_and_labels(self):
        """Test MultiCloud show() method with title and labels."""
        title = "Test Multi-Cloud Visualization"
        labels = ["Document 1", "Document 2"]

        mc = MultiCloud(data=SAMPLE_DOCS[:2], title=title, labels=labels)

        # Figure should exist and be displayable
        fig = mc.show()
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Verify the figure has the expected properties
        assert mc.title == title
        assert mc.labels == labels

    def test_multicloud_show_method_after_save(self):
        """Test MultiCloud show() method after saving to file."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        # Save the figure first
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            mc.save(tmp_path)

            # show() should still work after saving
            fig = mc.show()
            assert fig is not None
            assert isinstance(fig, plt.Figure)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multicloud_show_method_multiple_calls(self):
        """Test MultiCloud show() method called multiple times."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        # Multiple calls should return the same figure
        fig1 = mc.show()
        fig2 = mc.show()
        fig3 = mc.show()

        assert fig1 is fig2 is fig3
        assert fig1 == mc.fig
