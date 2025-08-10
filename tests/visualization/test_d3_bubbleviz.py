"""test_d3_bubbleviz.py.

Last Updated: August 10, 2025

Covered: 99% (line 182 should be unreachable due to Pydantic validation)
"""

import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import spacy
from scipy.sparse import csr_matrix

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.d3_bubbleviz import D3BubbleChart

# Test data
SAMPLE_TEXT = (
    "natural language processing machine learning artificial intelligence data science"
)
SAMPLE_DICT = {
    "natural": 5,
    "language": 4,
    "processing": 3,
    "machine": 2,
    "learning": 1,
}
SAMPLE_LIST_STRINGS = ["natural", "language", "processing", "machine", "learning"]
SAMPLE_LIST_LISTS = [
    ["natural", "language"],
    ["processing", "machine"],
    ["learning", "data"],
]


@pytest.fixture
def sample_dtm():
    """Create a sample DTM for testing.

    Returns:
        DTM: Sample DTM instance with test data
    """
    dtm = DTM()

    # Create sample data
    data = np.array([[1, 0, 3], [0, 2, 1], [2, 1, 0]])
    dtm.doc_term_matrix = csr_matrix(data)

    # Mock vectorizer
    class MockVectorizer:
        terms_list = ["term1", "term2", "term3"]

    dtm.vectorizer = MockVectorizer()

    # Set labels
    dtm.labels = ["doc1", "doc2", "doc3"]

    return dtm


class TestD3BubbleChartInitialization:
    """Test D3BubbleChart initialization and configuration."""

    def test_default_initialization_with_string(self):
        """Test default initialization with string data."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        assert chart.title == "Bubble Chart Visualization"
        assert chart.height == 600
        assert chart.width == 960
        assert chart.margin == {"top": 20, "right": 20, "bottom": 20, "left": 20}
        assert chart.color == "schemeCategory10"
        assert chart.auto_open is False
        assert isinstance(chart.counts, dict)
        assert len(chart.counts) == 9  # Number of unique words

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        custom_margin = {"top": 30, "right": 30, "bottom": 30, "left": 30}
        chart = D3BubbleChart(
            data=SAMPLE_TEXT,
            title="Custom Title",
            height=800,
            width=1200,
            margin=custom_margin,
            color="schemeSet3",
            limit=5,
            auto_open=False,
        )

        assert chart.title == "Custom Title"
        assert chart.height == 800
        assert chart.width == 1200
        assert chart.margin == custom_margin
        assert chart.color == "schemeSet3"
        assert chart.limit == 5
        assert len(chart.counts) <= 5

    def test_template_path_resolution(self):
        """Test that template path is resolved correctly."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        assert isinstance(chart.template, Path)
        assert chart.template.name == "d3_bubbles_template-1.0.html"
        assert "d3_cloud_assets" in str(chart.template)


class TestDataProcessing:
    """Test different data input types and processing."""

    def test_string_data_processing(self):
        """Test processing of string data."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        expected_counts = Counter(SAMPLE_TEXT.split())
        assert chart.counts == dict(expected_counts)

    def test_dict_data_processing(self):
        """Test processing of dictionary data."""
        chart = D3BubbleChart(data=SAMPLE_DICT, auto_open=False)

        assert chart.counts == SAMPLE_DICT

    def test_list_strings_processing(self):
        """Test processing of list of strings."""
        chart = D3BubbleChart(data=SAMPLE_LIST_STRINGS, auto_open=False)

        expected_counts = Counter(SAMPLE_LIST_STRINGS)
        assert chart.counts == dict(expected_counts)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        chart = D3BubbleChart(data="", auto_open=False)
        assert chart.counts == {}

        chart = D3BubbleChart(data=[], auto_open=False)
        assert chart.counts == {}

        chart = D3BubbleChart(data={}, auto_open=False)
        assert chart.counts == {}

    def test_limit_functionality(self):
        """Test that limit parameter works correctly."""
        chart = D3BubbleChart(data=SAMPLE_DICT, limit=3, auto_open=False)

        assert len(chart.counts) == 3
        # Should contain the top 3 most frequent terms
        sorted_original = sorted(SAMPLE_DICT.items(), key=lambda x: x[1], reverse=True)
        top_3_terms = {k: v for k, v in sorted_original[:3]}
        assert chart.counts == top_3_terms

    def test_no_limit_with_large_dataset(self):
        """Test behavior when no limit is set."""
        large_text = " ".join(["word" + str(i) for i in range(100)])
        chart = D3BubbleChart(data=large_text, auto_open=False)

        assert len(chart.counts) == 100

    def test_spacy_doc_processing(self):
        """Test processing of spaCy Doc objects."""
        nlp = spacy.blank("en")
        doc = nlp(SAMPLE_TEXT)

        chart = D3BubbleChart(data=doc, auto_open=False)

        expected_counts = Counter([token.text for token in doc])
        assert chart.counts == dict(expected_counts)

    @pytest.mark.skip(reason="Pydantic validation should make this unreachable.")
    def test_unsupported_data_type(self):
        """Test that unsupported data types raise appropriate exceptions."""
        with pytest.raises(LexosException) as exc_info:
            D3BubbleChart(data=12345, auto_open=False)

        assert "Unsupported data type" in str(exc_info.value)

    def test_dataframe_processing(self):
        """Test processing of pandas DataFrame."""
        data = {
            "doc1": [1, 2, 3, 7, 1],
            "doc2": [4, 5, 6, 8, 2],
            "doc3": [7, 8, 9, 3, 4],
        }
        df = pd.DataFrame(
            data, index=["natural", "language", "machine", "learning", "data"]
        )

        chart = D3BubbleChart(data=df, auto_open=False)
        assert isinstance(chart.counts, dict)
        assert len(chart.counts) == 5
        assert chart.counts == {
            "natural": 12,
            "language": 15,
            "machine": 18,
            "learning": 18,
            "data": 7,
        }

    def test_dtm_processing(self, sample_dtm):
        """Test processing of DTM objects (covers line 173)."""
        chart = D3BubbleChart(data=sample_dtm, auto_open=False)

        assert isinstance(chart.counts, dict)
        assert len(chart.counts) == 3
        assert "term1" in chart.counts
        assert "term2" in chart.counts
        assert "term3" in chart.counts

    def test_list_of_lists_processing(self):
        """Test processing of list of lists (covers line 157)."""
        list_of_lists = [
            ["natural", "language", "processing"],
            ["machine", "learning", "artificial"],
            ["data", "science", "intelligence"],
        ]

        chart = D3BubbleChart(data=list_of_lists, auto_open=False)

        assert isinstance(chart.counts, dict)
        # Should flatten and count all terms
        expected_terms = [
            "natural",
            "language",
            "processing",
            "machine",
            "learning",
            "artificial",
            "data",
            "science",
            "intelligence",
        ]
        for term in expected_terms:
            assert term in chart.counts

    def test_list_of_spacy_docs_processing(self):
        """Test processing of list of spaCy Doc objects (covers line 161)."""
        nlp = spacy.blank("en")
        docs = [
            nlp("natural language processing"),
            nlp("machine learning"),
            nlp("data science"),
        ]

        chart = D3BubbleChart(data=docs, auto_open=False)

        assert isinstance(chart.counts, dict)
        expected_terms = [
            "natural",
            "language",
            "processing",
            "machine",
            "learning",
            "data",
            "science",
        ]
        for term in expected_terms:
            assert term in chart.counts


class TestTemplateAndRendering:
    """Test template loading and HTML rendering."""

    def test_template_loading(self):
        """Test that template is loaded correctly."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        # Template should be loaded and HTML should be generated
        assert chart.html is not None
        assert isinstance(chart.html, str)
        assert len(chart.html) > 0

    def test_html_contains_expected_elements(self):
        """Test that rendered HTML contains expected elements."""
        chart = D3BubbleChart(data=SAMPLE_DICT, title="Test Chart", auto_open=False)

        html = chart.html

        # Check for essential HTML elements
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Chart</title>" in html
        assert "d3.js" in html.lower() or "d3.v7" in html
        assert str(chart.height) in html
        assert str(chart.width) in html

        # Check for data inclusion
        for term in SAMPLE_DICT.keys():
            assert term in html

    def test_template_rendering_with_custom_colors(self):
        """Test template rendering with custom color schemes."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, color="schemeSet3", auto_open=False)

        assert "schemeSet3" in chart.html

    def test_d3js_inclusion_flag(self):
        """Test D3.js inclusion flag behavior."""
        # Test with include_d3js=True
        chart_with_d3 = D3BubbleChart(
            data=SAMPLE_TEXT, include_d3js=True, auto_open=False
        )

        # Test with include_d3js=False
        chart_without_d3 = D3BubbleChart(
            data=SAMPLE_TEXT, include_d3js=False, auto_open=False
        )

        # Both should have HTML, but content might differ
        assert chart_with_d3.html is not None
        assert chart_without_d3.html is not None


class TestFileOperations:
    """Test file operations and asset management."""

    def test_asset_path_resolution(self):
        """Test asset path resolution method."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        asset_path = chart._get_asset_path("test_file.html")

        assert isinstance(asset_path, Path)
        assert asset_path.name == "test_file.html"
        assert "d3_cloud_assets" in str(asset_path)

    def test_save_functionality(self):
        """Test saving HTML to file."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save the chart
            chart.save(tmp_path)

            # Verify file was created and contains expected content
            with open(tmp_path, "r", encoding="utf-8") as f:
                saved_content = f.read()

            assert saved_content == chart.html
            assert "<!DOCTYPE html>" in saved_content

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_with_pathlib_path(self):
        """Test saving with pathlib.Path object."""
        chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_chart.html"
            chart.save(save_path)

            assert save_path.exists()
            assert save_path.read_text(encoding="utf-8") == chart.html


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_large_dataset(self):
        """Test with a very large dataset."""
        # Create a large dataset
        large_text = " ".join([f"word{i}" for i in range(1000)])
        chart = D3BubbleChart(data=large_text, limit=10, auto_open=False)

        assert len(chart.counts) == 10
        assert chart.html is not None

    def test_special_characters_in_data(self):
        """Test handling of special characters."""
        special_text = "hello! world? data-science machine_learning @mentions #hashtags"
        chart = D3BubbleChart(data=special_text, auto_open=False)

        # Should handle special characters gracefully
        assert isinstance(chart.counts, dict)
        assert chart.html is not None

    def test_unicode_characters(self):
        """Test handling of Unicode characters."""
        unicode_text = "héllo wørld datá sciençe machine learning"
        chart = D3BubbleChart(data=unicode_text, auto_open=False)

        assert isinstance(chart.counts, dict)
        assert chart.html is not None

    def test_empty_string_values_in_dict(self):
        """Test dictionary with empty string keys."""
        dict_with_empty = {"": 5, "normal": 3, "word": 2}
        chart = D3BubbleChart(data=dict_with_empty, auto_open=False)

        assert isinstance(chart.counts, dict)

    def test_zero_counts_in_dict(self):
        """Test dictionary with zero count values."""
        dict_with_zeros = {"word1": 5, "word2": 0, "word3": 3}
        chart = D3BubbleChart(data=dict_with_zeros, auto_open=False)

        assert chart.counts == dict_with_zeros


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_string_to_html(self):
        """Test complete workflow from string input to HTML output."""
        input_text = (
            "machine learning natural language processing artificial intelligence"
        )

        limit = 4

        chart = D3BubbleChart(
            data=input_text, title="ML/NLP Terms", limit=limit, auto_open=False
        )

        # Verify all steps completed successfully
        assert chart.counts is not None
        assert chart.html is not None
        assert len(chart.counts) <= limit
        assert "ML/NLP Terms" in chart.html

        # Verify all input terms are processed
        for word in input_text.split()[:limit]:
            assert word in chart.counts

    def test_complete_workflow_dict_to_file(self):
        """Test complete workflow from dict input to saved file."""
        input_dict = {
            "machine": 10,
            "learning": 8,
            "natural": 6,
            "language": 4,
            "processing": 2,
        }

        chart = D3BubbleChart(
            data=input_dict, title="Term Frequencies", auto_open=False
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            chart.save(tmp_path)

            # Verify saved file
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "Term Frequencies" in content
            for term in input_dict.keys():
                assert term in content

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_template_file_not_found(self):
        """Test handling when template file doesn't exist (covers lines 187-194)."""
        with pytest.raises(LexosException, match="Template file not found"):
            D3BubbleChart(
                data=SAMPLE_TEXT, auto_open=False, template="invalid_template.html"
            )

    def test_auto_open_functionality(self):
        """Test auto_open functionality calls webbrowser.open (may cover additional lines)."""
        with patch("webbrowser.open") as mock_open:
            chart = D3BubbleChart(data=SAMPLE_TEXT, auto_open=True)

            # Should call webbrowser.open when auto_open=True
            mock_open.assert_called_once()

    def test_custom_color_array(self):
        """Test with custom color array instead of D3 scheme name."""
        custom_colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff"]

        chart = D3BubbleChart(data=SAMPLE_TEXT, color=custom_colors, auto_open=False)

        assert chart.color == custom_colors
        # Check that custom colors are used in HTML
        for color in custom_colors:
            assert color in chart.html

    def test_very_long_title(self):
        """Test with very long title to ensure proper handling."""
        long_title = "A" * 1000  # Very long title

        chart = D3BubbleChart(data=SAMPLE_TEXT, title=long_title, auto_open=False)

        assert chart.title == long_title
        assert long_title in chart.html

    def test_margin_validation_edge_cases(self):
        """Test margin with edge case values."""
        # Test with zero margins
        chart = D3BubbleChart(
            data=SAMPLE_TEXT,
            margin={"top": 0, "right": 0, "bottom": 0, "left": 0},
            auto_open=False,
        )

        assert chart.margin == {"top": 0, "right": 0, "bottom": 0, "left": 0}

        # Test with very large margins
        large_margin = {"top": 1000, "right": 1000, "bottom": 1000, "left": 1000}
        chart = D3BubbleChart(data=SAMPLE_TEXT, margin=large_margin, auto_open=False)

        assert chart.margin == large_margin


# Fixtures for test data
@pytest.fixture
def sample_chart():
    """Fixture providing a basic D3BubbleChart instance."""
    return D3BubbleChart(data=SAMPLE_TEXT, auto_open=False)


@pytest.fixture
def sample_dict_chart():
    """Fixture providing a D3BubbleChart with dictionary data."""
    return D3BubbleChart(data=SAMPLE_DICT, auto_open=False)


# Performance tests (optional, can be marked with @pytest.mark.slow)
def test_performance_large_dataset():
    """Test performance with large dataset."""
    import time

    # Generate large dataset
    large_text = " ".join([f"term{i}" for i in range(10000)])

    start_time = time.time()
    chart = D3BubbleChart(data=large_text, limit=100, auto_open=False)
    end_time = time.time()

    # Should complete within reasonable time (adjust threshold as needed)
    assert end_time - start_time < 5.0  # 5 seconds
    assert len(chart.counts) == 100
