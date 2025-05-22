"""test_replace.py.

Last Update: 2025-01-14.
"""
import pytest
from lexos.scrubber.replace import (
    currency_symbols,
    digits,
    emails,
    emojis,
    hashtags,
    phone_numbers,
    punctuation,
    special_characters,
    tag_map,
    urls,
    user_handles,
    pattern,
    process_tag_replace_options,
)


def test_currency_symbols():
    """Test replacing currency symbols."""
    text = "The price is $100."
    expected = "The price is _CUR_100."
    assert currency_symbols(text) == expected

def test_digits():
    """Test replacing digits."""
    text = "My phone number is 123-456-7890."
    expected = "My phone number is _DIGIT_-_DIGIT_-_DIGIT_."
    assert digits(text) == expected

def test_emails():
    """Test replacing email addresses."""
    text = "Contact me at example@example.com."
    expected = "Contact me at _EMAIL_."
    assert emails(text) == expected

def test_emojis():
    """Test replacing emojis."""
    text = "I am happy 😊."
    expected = "I am happy _EMOJI_."
    assert emojis(text) == expected

def test_hashtags():
    """Test replacing hashtags."""
    text = "This is a #test."
    expected = "This is a _HASHTAG_."
    assert hashtags(text) == expected

def test_pattern():
    """Test replacing patterns."""
    text = "This is a test."
    pattern_dict = {"test": "_PATTERN_"}
    expected = "This is a _PATTERN_."
    assert pattern(text, pattern=pattern_dict) == expected

def test_phone_numbers():
    """Test replacing phone numbers."""
    text = "Call me at 123-456-7890 or 1.123.456.7890."
    expected = "Call me at _PHONE_ or _PHONE_."
    assert phone_numbers(text) == expected

def test_process_tag_replace_options_remove_tag():
    text = "<p>This is a <b>test</b>.</p>"
    result = process_tag_replace_options(text, "b", "remove_tag", "")
    # Should remove <b> and </b> tags, but keep content
    assert result == "<p>This is a  test .</p>"

def test_process_tag_replace_options_remove_element():
    text = "<p>This is a <b>test</b>.</p>"
    result = process_tag_replace_options(text, "b", "remove_element", "")
    # Should remove <b>test</b> entirely
    assert result == "<p>This is a  .</p>"

def test_process_tag_replace_options_replace_element():
    text = "<p>This is a <b>test</b>.</p>"
    result = process_tag_replace_options(text, "b", "replace_element", "_BOLD_")
    # Should replace <b>test</b> with _BOLD_
    assert result == "<p>This is a _BOLD_.</p>"

def test_process_tag_replace_options_default():
    text = "<p>This is a <b>test</b>.</p>"
    result = process_tag_replace_options(text, "b", "unknown_action", "")
    # Should leave text unchanged
    assert result == "<p>This is a <b>test</b>.</p>"

def test_punctuation():
    """Test replacing punctuation."""
    text = "Hello, world!"
    assert punctuation(text) == "Hello  world "
    assert punctuation(text, only="!") == "Hello, world "
    assert punctuation(text, exclude=",") == "Hello, world "

def test_special_characters():
    """Test replacing special characters."""
    text = "This is a test & example."
    ruleset = {"&": "and"}
    expected = "This is a test and example."
    assert special_characters(text, ruleset=ruleset) == expected

def test_special_characters_html_unescape():
    """Test replacing special characters with is_html=True (HTML unescape branch)."""
    text = "This &amp; that &lt;test&gt;"
    expected = "This & that <test>"
    assert special_characters(text, is_html=True) == expected

def test_tag_map():
    """Test replacing tags using tag_map."""
    text = "<html><body><p>This is a test.   </p></body></html>"
    tag_map_dict = {"p": {"action": "remove_tag", "attribute": ""}}
    assert tag_map(text, map=tag_map_dict) == "<html><body> This is a test.    </body></html>"
    assert tag_map(text, map=tag_map_dict, remove_whitespace=True) == "<html><body> This is a test. </body></html>"

def test_urls():
    """Test replacing URLs."""
    text = "Visit https://example.com for more info."
    expected = "Visit _URL_ for more info."
    assert urls(text) == expected

def test_user_handles():
    """Test replacing user handles."""
    text = "Follow me on Twitter @example."
    expected = "Follow me on Twitter _USER_."
    assert user_handles(text) == expected
