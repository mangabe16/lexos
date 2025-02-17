"""test_pipeline.py.

Last Update: 2025-01-14.
"""
from functools import partial

import pytest
from lexos.scrubber.pipeline import pipe


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

