"""test_resources.py.

Last Update: 2025-01-14.
"""
import pytest
from lexos.scrubber.resources import HTMLTextExtractor


def test_html_text_extractor_basic():
    """Test basic HTML parsing and text extraction."""
    html_content = "<html><body><p>Test</p><p>Content</p></body></html>"
    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    result = extractor.get_text()
    assert result == "TestContent"

def test_html_text_extractor_with_separator():
    """Test text extraction with separator."""
    html_content = "<html><body><p>Test</p><p>Content</p></body></html>"
    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    result = extractor.get_text(sep=" ")
    assert result == "Test Content"

def test_html_text_extractor_empty():
    """Test handling of empty data."""
    extractor = HTMLTextExtractor()
    result = extractor.get_text()
    assert result == ""

def test_html_text_extractor_handle_data():
    """Test handling of data elements."""
    extractor = HTMLTextExtractor()
    extractor.handle_data("Sample")
    extractor.handle_data("Text")
    result = extractor.get_text()
    assert result == "SampleText"
