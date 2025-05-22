"""test_pipeline.py.

Last Update: 2025-01-14.
"""
from functools import partial

import pytest
from lexos.scrubber.pipeline import (
    pipe,
    make_pipeline,
    make_pipeline_from_tuple,
)


def sample_function(x, y=0):
    return x + y

def test_pipe_no_args():
    """Test pipe with no arguments."""
    func = pipe(sample_function)
    assert func(2, 3) == 5
    assert func.__name__ == "sample_function"

def test_pipe_positional_args():
    """Test pipe with positional arguments."""
    func = pipe(sample_function, 2)
    assert func(3) == 5
    assert func.__name__ == "sample_function"

def test_pipe_keyword_args():
    """Test pipe with keyword arguments."""
    func = pipe(sample_function, y=3)
    assert func(2) == 5
    assert func.__name__ == "sample_function"

def test_pipe_positional_and_keyword_args():
    """Test pipe with both positional and keyword arguments."""
    func = pipe(sample_function, 2, y=3)
    assert func() == 5
    assert func.__name__ == "sample_function"


def dummy_upper(text: str) -> str:
    return text.upper()

def dummy_replace_a_with_x(text: str) -> str:
    return text.replace('A', 'X')

def test_make_pipeline_basic():
    func1 = pipe(dummy_upper)
    func2 = pipe(dummy_replace_a_with_x)
    pipeline = make_pipeline(func1, func2)
    #pipeline = make_pipeline(dummy_upper, dummy_replace_a_with_x)
    result = pipeline("a cat and a bat")
    # Should first uppercase, then replace 'A' with 'X'
    assert result == "X CXT XND X BXT"

def test_make_pipeline_with_pipe():
    def add_prefix(text: str, prefix: str) -> str:
        return f"{prefix}{text}"
    pipeline = make_pipeline(
        dummy_upper,
        pipe(add_prefix, prefix=">>")
    )
    result = pipeline("abc")
    assert result == ">>ABC"

def dummy_add_exclamation(text: str) -> str:
    return text + "!"

def test_make_pipeline_from_tuple_with_callables():
    funcs = (dummy_upper, dummy_add_exclamation)
    pipeline = make_pipeline_from_tuple(funcs)
    result = pipeline("hello")
    assert result == "HELLO!"
