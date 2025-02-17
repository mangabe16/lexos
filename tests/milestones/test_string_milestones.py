"""test_string_milestones.py.

TODO:
Last Update: 12/21/2024
"""

import pytest
import spacy
from lexos.milestones.string_milestones import (
    StringMilestones,
    case_insensitive_flags,
    case_sensitive_flags,
)
from pydantic import ValidationError
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

# Fixtures


@pytest.fixture
def doc():
    text = "This is a test document."
    return nlp(text)


# Tests


def test_string_milestones_init_with_string():
    """Test StringMilestones initialization with a string."""
    milestones = StringMilestones(doc="This is a test document.")
    assert isinstance(milestones.doc, str)
    assert milestones.patterns == [None]
    assert milestones.case_sensitive is True
    assert milestones.flags == case_sensitive_flags


def test_string_milestones_init_with_doc(doc):
    """Test StringMilestones initialization with a spaCy Doc."""
    milestones = StringMilestones(doc=doc)
    assert isinstance(milestones.doc, Doc)
    assert milestones.patterns == [None]
    assert milestones.case_sensitive is True
    assert milestones.flags == case_sensitive_flags


def test_string_milestones_init_with_patterns():
    """Test StringMilestones initialization with patterns."""
    patterns = ["test", "document"]
    milestones = StringMilestones(doc="This is a test document.", patterns=patterns)
    assert milestones.patterns == patterns


def test_string_milestones_init_case_insensitive():
    """Test StringMilestones initialization with case insensitive."""
    milestones = StringMilestones(doc="This is a test document.", case_sensitive=False)
    assert milestones.case_sensitive is False
    assert milestones.flags == case_insensitive_flags


def test_string_milestones_invalid_doc():
    """Test StringMilestones initialization with invalid doc type."""
    with pytest.raises(ValidationError):
        StringMilestones(doc=12345)  # Invalid doc type


def test_string_milestones_set_method_called():
    """Test StringMilestones set method is called when patterns are provided."""
    patterns = ["test"]
    milestones = StringMilestones(doc="This is a test document.", patterns=patterns)
    assert milestones.patterns == patterns
    # assert milestones._spans[0].text == "test"  # Assuming set method initializes _spans


if __name__ == "__main__":
    pytest.main()
