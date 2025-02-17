"""test_scrubber.py.

Last Update: 20 January 2025
"""

from functools import partial

import catalogue
import pytest

from lexos.exceptions import LexosException
from lexos.scrubber.registry import scrubber_components
from lexos.scrubber.scrubber_bk import Pipe, Scrubber, scrub


@pytest.fixture
def scrubber():
    return Scrubber()

def test_pipe_class():
    """Test the Pipe class."""
    pipe = Pipe(name="digits", opts={"only": ["1"]})
    assert pipe.name == "digits"
    assert pipe.opts == {"only": ["1"]}
    assert pipe.factory == scrubber_components

def test_add_pipe_string(scrubber):
    """Test adding a named component to the scrubber pipeline."""
    scrubber.add_pipe("lower_case")
    assert len(scrubber._components) == 1
    assert scrubber._components[0].name == "lower_case"
    assert scrubber.pipes == ["lower_case"]

def test_add_pipe_partial(scrubber):
    """Test adding a partial containing a named component and kwargs to the scrubber pipeline."""
    from lexos.scrubber.remove import digits
    scrubber.add_pipe(partial(digits, only=["1"]))
    assert len(scrubber._components) == 1
    assert scrubber._components[0].name == "digits"
    assert scrubber._components[0].opts == {"only": ["1"]}
    assert scrubber.pipes == ["digits"]

def test_add_pipe_tuple(scrubber):
    """Test adding a tuple containing a named component and kwargs to the scrubber pipeline."""
    scrubber.add_pipe(("digits", {"only": ["1"]}))
    assert len(scrubber._components) == 1
    assert scrubber._components[0].name == "digits"
    assert scrubber._components[0].opts == {"only": ["1"]}
    assert scrubber.pipes == ["digits"]

def test_add_pipe_pipe(scrubber):
    """Test adding a Pipe object to the scrubber pipeline."""
    pipe = Pipe(name="digits", opts={"only": ["1"]})
    scrubber.add_pipe(pipe)
    assert len(scrubber._components) == 1
    assert isinstance(scrubber._components[0], Pipe)
    assert scrubber._components[0].name == "digits"
    assert scrubber._components[0].opts == {"only": ["1"]}
    assert scrubber.pipes == ["digits"]

def test_add_pipe_multiple_string(scrubber):
    """Test adding multiple named components to the scrubber pipeline."""
    scrubber.add_pipe(["lower_case", "digits"])
    assert len(scrubber._components) == 2
    assert scrubber._components[0].name == "lower_case"

def test_add_pipe_multiple_tuple(scrubber):
    """Test adding multiple tuples to the scrubber pipeline."""
    pipeline = [("lower_case", {}), ("digits", {"only": ["1"]})]
    scrubber.add_pipe(pipeline)
    assert len(scrubber._components) == 2
    assert scrubber._components[0].name == "lower_case"

def test_add_pipe_multiple_pipe(scrubber):
    """Test adding multiple Pipe objects to the scrubber pipeline."""
    pipeline = [Pipe(name="lower_case", opts={}), Pipe(name="digits", opts={"only": ["1"]})]
    scrubber.add_pipe(pipeline)
    assert len(scrubber._components) == 2
    assert scrubber._components[0].name == "lower_case"

def test_add_pipe_first(scrubber):
    """Test adding a named component to the start of the scrubber pipeline."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", first=True)
    assert scrubber._components[0].name == "digits"

def test_add_pipe_last(scrubber):
    """Test adding a named component to the end of the scrubber pipeline."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", last=True)
    assert scrubber._components[0].name == "lower_case"

def test_add_pipe_multiple_positions(scrubber):
    """Test adding a named component with multiple positions set."""
    scrubber.add_pipe("lower_case")
    with pytest.raises(LexosException, match="Only one of before"):
        scrubber.add_pipe("digits", first=True, last=True)

def test_add_pipe_before(scrubber):
    """Test adding a component before another."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", before="lower_case")
    assert scrubber._components[0].name == "digits"

def test_add_pipe_after(scrubber):
    """Test adding a component after another."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", after="lower_case")
    assert scrubber._components[0].name == "lower_case"

def test_add_pipe_before_after_invalid(scrubber):
    """Test adding a component before or after a non-existent component."""
    scrubber.add_pipe("lower_case")
    with pytest.raises(LexosException, match="The component name"):
        scrubber.add_pipe("digits", before="invalid_pipe")
    with pytest.raises(LexosException, match="The component name"):
        scrubber.add_pipe("digits", after="invalid_pipe")

def test_add_pipe_numeric_before(scrubber):
    """Test adding a component before or after a non-existent component."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits")
    scrubber.add_pipe("remove_whitespace", before=1)
    assert scrubber._components[1].name == "remove_whitespace"

def test_add_pipe_numeric_after(scrubber):
    """Test adding a component before or after a non-existent component."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits")
    scrubber.add_pipe("remove_whitespace", after=0)
    assert scrubber._components[1].name == "remove_whitespace"

def test_pipe(scrubber):
    """Test scrubbing texts with the pipeline."""

    scrubber.add_pipe("lower_case")
    texts = ["Apple1", "Banana2"]
    result = list(scrubber.pipe(texts))
    assert result == ["apple1", "banana2"]

def test_pipe_disable(scrubber):
    """Test scrubbing texts with an iterable as the pipeline."""

    pipeline = ["lower_case", "digits"]
    scrubber.add_pipe(pipeline)
    texts = ["Apple1", "Banana2"]
    result = list(scrubber.pipe(texts, disable=["digits"]))
    assert result == ["apple1", "banana2"]

def test_pipe_cfg(scrubber):
    """Test scrubbing texts with the pipeline."""

    scrubber.add_pipe(("digits", {"only": ["1"]}))
    component_cfg = {"digits": {"only": ["2"]}}
    texts = ["Apple1", "Banana2"]
    result = list(scrubber.pipe(texts, component_cfg=component_cfg))
    assert result == ["Apple1", "Banana"]

def test_remove_pipe(scrubber):
    """Test removing a pipe from the scrubber."""
    pipe = Pipe(name="test_pipe")
    scrubber.add_pipe(pipe)
    scrubber.remove_pipe("test_pipe")
    assert len(scrubber._components) == 0

def test_reset(scrubber):
    """Test resetting the scrubber pipeline."""
    pipe = Pipe(name="test_pipe")
    scrubber.add_pipe(pipe)
    scrubber.reset()
    assert len(scrubber._components) == 0

def test_scrub(scrubber):
    """Test scrubbing a text with the pipeline."""
    scrubber.add_pipe(("digits", {"only": ["1"]}))
    result = scrubber.scrub("apple1")
    assert result == "apple"

def test_invalid_pipe_index(scrubber):
    """Test invalid pipe index raises LexosException."""
    with pytest.raises(LexosException):
        scrubber._get_pipe_index(before=0, after=1)

def test_invalid_pipe_name(scrubber):
    """Test invalid pipe name raises LexosException."""
    with pytest.raises(LexosException):
        scrubber._get_pipe_index(before="invalid_pipe")

def test_empty_pipeline(scrubber):
    """Test scrubbing with an empty pipeline returns the text."""
    assert scrubber.scrub("text") == "text"

def test_scrub_with_callable():
    def mock_callable(text):
        return text.upper()
    result = scrub("test", [mock_callable])
    assert result == "TEST"

def test_scrub_with_partial():
    def mock_callable(text, suffix):
        return text + suffix

    mock_partial = partial(mock_callable, suffix="!")
    result = scrub("test", [mock_partial])
    assert result == "test!"

def test_scrub_with_tuple():
    def mock_callable(text, prefix):
        return prefix + text

    result = scrub("test", [(mock_callable, {"prefix": "Hello, "})])
    assert result == "Hello, test"

def test_scrub_with_string():
    result = scrub("TEST", ["lower_case"])
    assert result == "test"

def test_scrub_with_invalid_component():
    with pytest.raises(LexosException):
        scrub("test", ["nonexistent_component"])

def test_scrub_with_registry_error():
    from unittest.mock import Mock
    mock_factory = Mock()
    mock_factory.get.side_effect = catalogue.RegistryError("Component not found")

    with pytest.raises(LexosException):
        scrub("test", ["nonexistent_component"], factory=mock_factory)
