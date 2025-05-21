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

def test_phone_numbers():
    """Test replacing phone numbers."""
    text = "Call me at 123-456-7890 or 1.123.456.7890."
    expected = "Call me at _PHONE_ or _PHONE_."
    assert phone_numbers(text) == expected

def test_punctuation():
    """Test replacing punctuation."""
    text = "Hello, world!"
    expected = "Hello  world "
    assert punctuation(text) == expected

def test_special_characters():
    """Test replacing special characters."""
    text = "This is a test & example."
    ruleset = {"&": "and"}
    expected = "This is a test and example."
    assert special_characters(text, ruleset=ruleset) == expected

def test_tag_map():
    """Test replacing tags using tag_map."""
    text = "<html><body><p>This is a test.</p></body></html>"
    tag_map_dict = {"p": {"action": "remove_tag", "attribute": ""}}
    expected = "<html><body> This is a test. </body></html>"
    assert tag_map(text, map=tag_map_dict) == expected

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
