"""test_prepare_data.py.

Last updated: 2025-27-04.
Last tested: 2025-27-04.
"""

import gzip
import json
import os
import tempfile
import zipfile as zf
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lexos.topic_modeling.dfr_browser.prepare_data import (
    convert_state,
    info_stub,
    transform_dt,
    transform_topic_weights,
    write_dt,
    write_tw,
)

#### Common Fixtures


@pytest.fixture
def mock_state_file():
    """Creates a temporary mock MALLET state file for testing.

    Returns:
        str: Path to the temporary gzipped state file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as temp:
        # Sample MALLET state file format
        content = (
            b"#doc source pos typeindex token topic\n"
            b"#alpha : 0.5 0.5 0.5\n"
            b"#beta : 0.01\n"
            b"0 en.txt 0 0 word1 0\n"
            b"0 en.txt 1 1 word2 1\n"
            b"0 en.txt 2 2 word3 0\n"
            b"1 fr.txt 0 3 word4 1\n"
            b"1 fr.txt 1 4 word5 2\n"
            b"1 fr.txt 2 0 word1 0\n"
            b"2 de.txt 0 5 word6 0\n"
            b"2 de.txt 1 6 word7 2\n"
            b"2 de.txt 2 7 word8 1\n"
        )

        # Write to gzipped file
        with gzip.open(temp.name, "wb") as f:
            f.write(content)

    yield temp.name

    # Clean up
    os.unlink(temp.name)


#### Tests for convert_state function


def test_convert_state_basic_functionality(mock_state_file):
    """Tests the basic functionality of convert_state.

    Verifies that convert_state correctly reads the state file and calls
    write_tw and write_dt with the expected arguments.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    # Mock the write functions
    with (
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.write_tw"
        ) as mock_write_tw,
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.write_dt"
        ) as mock_write_dt,
    ):
        # Call the function
        convert_state(mock_state_file)

        # Check that write_tw and write_dt were called
        assert mock_write_tw.called
        assert mock_write_dt.called

        # Check that the alpha values were correctly extracted
        alpha_arg = mock_write_tw.call_args[0][0]
        assert alpha_arg == [0.5, 0.5, 0.5]

        # Check that the topic count is correct (3 topics: 0, 1, 2)
        tw_arg = mock_write_tw.call_args[0][1]
        assert len(tw_arg) == 3


def test_convert_state_topic_words_extraction(mock_state_file):
    """Tests that convert_state correctly extracts topic-word associations.

    Verifies that the function correctly builds the topic-word matrix from
    the state file.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    # Mock the transform_topic_weights function to capture its inputs
    original_transform = transform_topic_weights
    transform_inputs = {}

    def mock_transform(weights, vocab, n):
        # Store inputs for later verification
        topic_idx = len(transform_inputs)
        transform_inputs[topic_idx] = {"weights": weights.copy(), "vocab": vocab.copy()}
        return original_transform(weights, vocab, n)

    with (
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.transform_topic_weights",
            side_effect=mock_transform,
        ),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.write_tw"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.write_dt"),
    ):
        convert_state(mock_state_file)

        # Check that topic 0 has the right word counts
        topic0_weights = transform_inputs[0]["weights"]
        topic0_words_count = len(topic0_weights)
        assert topic0_words_count >= 3  # word1 appears twice, word3, word6

        # Check that topic 1 has the right word counts
        topic1_weights = transform_inputs[1]["weights"]
        topic1_words_count = len(topic1_weights)
        assert topic1_words_count >= 3  # word2, word4, word8

        # Check that topic 2 has the right word counts
        topic2_weights = transform_inputs[2]["weights"]
        topic2_words_count = len(topic2_weights)
        assert topic2_words_count >= 2  # word5, word7


def test_convert_state_document_topic_extraction(mock_state_file):
    """Tests that convert_state correctly extracts document-topic assignments.

    Verifies that the function correctly builds the document-topic matrix from
    the state file.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    # Mock the transform_dt function to capture its input
    transform_dt_input = None
    original_transform_dt = transform_dt

    def mock_transform_dt(dt_matrix):
        nonlocal transform_dt_input
        transform_dt_input = [
            list(row) for row in dt_matrix
        ]  # Deep copy to prevent modification
        return original_transform_dt(dt_matrix)

    with (
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.transform_dt",
            side_effect=mock_transform_dt,
        ),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.write_tw"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.write_dt"),
    ):
        convert_state(mock_state_file)

        # Check that we have the right number of topics and documents
        assert len(transform_dt_input) == 3  # 3 topics
        assert len(transform_dt_input[0]) == 3  # 3 documents

        # Check document-topic counts (from our mock data):
        # Document 0: 2 words in topic 0, 1 word in topic 1, 0 words in topic 2
        assert transform_dt_input[0][0] == 2  # Doc 0, Topic 0
        assert transform_dt_input[1][0] == 1  # Doc 0, Topic 1
        assert transform_dt_input[2][0] == 0  # Doc 0, Topic 2

        # Document 1: 1 word in topic 0, 1 word in topic 1, 1 word in topic 2
        assert transform_dt_input[0][1] == 1  # Doc 1, Topic 0
        assert transform_dt_input[1][1] == 1  # Doc 1, Topic 1
        assert transform_dt_input[2][1] == 1  # Doc 1, Topic 2

        # Document 2: 1 word in topic 0, 1 word in topic 1, 1 word in topic 2
        assert transform_dt_input[0][2] == 1  # Doc 2, Topic 0
        assert transform_dt_input[1][2] == 1  # Doc 2, Topic 1
        assert transform_dt_input[2][2] == 1  # Doc 2, Topic 2


def test_convert_state_custom_output_paths(mock_state_file):
    """Tests convert_state with custom output paths.

    Verifies that the function correctly uses custom paths for the
    output files.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    custom_tw_file = "custom_tw.json"
    custom_dt_file = "custom_dt.json.zip"

    with (
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.write_tw"
        ) as mock_write_tw,
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.write_dt"
        ) as mock_write_dt,
    ):
        convert_state(
            state_file=mock_state_file, tw_file=custom_tw_file, dt_file=custom_dt_file
        )

        # Check that write functions were called with custom paths
        mock_write_tw.assert_called_once()
        assert mock_write_tw.call_args[0][2] == custom_tw_file

        mock_write_dt.assert_called_once()
        assert mock_write_dt.call_args[0][1] == custom_dt_file


def test_convert_state_custom_topic_count(mock_state_file):
    """Tests convert_state with custom topic count parameter.

    Verifies that the function correctly limits the number of topics
    processed based on the n parameter.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    custom_n = 10  # Limit to top 10 words per topic

    with patch(
        "lexos.topic_modeling.dfr_browser.prepare_data.transform_topic_weights"
    ) as mock_transform:
        mock_transform.return_value = {"words": [], "weights": []}

        with (
            patch("lexos.topic_modeling.dfr_browser.prepare_data.write_tw"),
            patch("lexos.topic_modeling.dfr_browser.prepare_data.write_dt"),
        ):
            convert_state(state_file=mock_state_file, n=custom_n)

            # Check that transform_topic_weights was called with n=10
            assert mock_transform.called
            calls = mock_transform.call_args_list
            for call_args in calls:
                assert call_args[0][2] == custom_n


def test_convert_state_empty_document(mock_state_file):
    """Tests convert_state with an empty document at the end.

    Verifies that the function correctly handles the case where the last
    document has no topics assigned.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    # Create a state file with an empty document at the end
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as temp:
        content = (
            b"#doc source pos typeindex token topic\n"
            b"#alpha : 0.5 0.5\n"
            b"#beta : 0.01\n"
            b"0 en.txt 0 1 word1 0\n"
            b"0 en.txt 1 2 word2 1\n"
            b"1 fr.txt 0 3 word3 0\n"
            b"1 fr.txt 1 4 word4 1\n"
            b"2 de.txt 0 0 empty 0\n"  # This will be processed as doc 2
            b"3 it.txt 0 0 none 0\n"  # This will create an empty doc entry for doc 3
        )

        with gzip.open(temp.name, "wb") as f:
            f.write(content)

        # Mock the transform_dt function to capture its input
        transform_dt_input = None

        def mock_transform_dt(dt_matrix):
            nonlocal transform_dt_input
            transform_dt_input = [list(row) for row in dt_matrix]
            return {"i": [], "p": [], "x": []}

        with (
            patch(
                "lexos.topic_modeling.dfr_browser.prepare_data.transform_dt",
                side_effect=mock_transform_dt,
            ),
            patch("lexos.topic_modeling.dfr_browser.prepare_data.write_tw"),
            patch("lexos.topic_modeling.dfr_browser.prepare_data.write_dt"),
        ):
            convert_state(temp.name)

            # Check we have 4 documents (including the empty one)
            assert len(transform_dt_input[0]) == 4

    # Clean up
    os.unlink(temp.name)


def test_convert_state_large_topic_numbers(mock_state_file):
    """Tests convert_state with large, non-consecutive topic numbers.

    Verifies that the function correctly handles the case where topic
    numbers are large and non-consecutive.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    # Create a state file with non-consecutive topic numbers
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as temp:
        content = (
            b"#doc source pos typeindex token topic\n"
            b"#alpha : 0.5 0.5 0.5 0.5 0.5\n"
            b"#beta : 0.01\n"
            b"0 en.txt 0 0 word1 0\n"
            b"0 en.txt 1 1 word2 1\n"
            b"0 en.txt 2 2 word3 0\n"
            b"1 fr.txt 0 3 word4 1\n"
            b"1 fr.txt 1 4 word5 2\n"
            b"1 fr.txt 2 5 word1 3\n"
            b"2 de.txt 0 6 word6 4\n"
            b"2 de.txt 1 7 word7 5\n"
            b"2 de.txt 2 8 word8 6\n"
            b"1 fr.txt 2 9 word1 7\n"
            b"1 fr.txt 2 10 word1 8\n"
            b"1 fr.txt 2 11 word1 9\n"
        )

        with gzip.open(temp.name, "wb") as f:
            f.write(content)

        with (
            patch(
                "lexos.topic_modeling.dfr_browser.prepare_data.write_tw"
            ) as mock_write_tw,
            patch("lexos.topic_modeling.dfr_browser.prepare_data.write_dt"),
        ):
            convert_state(temp.name)

            # We should have 10 topics (0-9)
            tw_arg = mock_write_tw.call_args[0][1]
            assert len(tw_arg) == 10

    # Clean up
    os.unlink(temp.name)


def test_convert_state_beta_extraction(mock_state_file, capsys):
    """Tests that convert_state correctly extracts and prints the beta value.

    Verifies that the function correctly extracts the beta value from
    the state file header and prints it.

    Args:
        mock_state_file: Path to the mock gzipped state file.
        capsys: Pytest fixture to capture stdout/stderr.
    """
    with (
        patch("lexos.topic_modeling.dfr_browser.prepare_data.write_tw"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.write_dt"),
    ):
        convert_state(mock_state_file)

        # Check that the beta value was printed
        captured = capsys.readouterr()
        assert "beta value, not saved in a file: 0.01" in captured.out


def test_convert_state_integration(mock_state_file):
    """Integration test for convert_state.

    Tests the entire flow of convert_state without mocking the transform
    functions to verify the end-to-end behavior.

    Args:
        mock_state_file: Path to the mock gzipped state file.
    """
    # Create temporary output files
    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tw_file,
        tempfile.NamedTemporaryFile(delete=False, suffix=".json.zip") as dt_file,
    ):
        # Call convert_state with real output files
        convert_state(
            state_file=mock_state_file, tw_file=tw_file.name, dt_file=dt_file.name
        )

        # Verify that the tw file was created with valid JSON
        with open(tw_file.name, "r") as f:
            tw_data = json.load(f)

            # Check structure
            assert "alpha" in tw_data
            assert "tw" in tw_data
            assert len(tw_data["alpha"]) == 3
            assert len(tw_data["tw"]) == 3

            # Check that each topic has words and weights
            for topic in tw_data["tw"]:
                assert "words" in topic
                assert "weights" in topic
                assert isinstance(topic["words"], list)
                assert isinstance(topic["weights"], list)

    # Clean up
    os.unlink(tw_file.name)
    os.unlink(dt_file.name)


#### Tests for info_stub function


def test_info_stub_default_properties():
    """Tests info_stub with default properties.

    Creates a temporary file and checks that info_stub correctly writes
    the default JSON structure to the file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name

    try:
        # Call the function
        with patch("builtins.print") as mock_print:
            info_stub(filepath)

            # Check that success message was printed
            mock_print.assert_called_once()
            assert filepath in mock_print.call_args[0][0]
            assert "Created stub file" in mock_print.call_args[0][0]

        # Verify the file content
        with open(filepath, "r") as f:
            content = json.load(f)

            # Check default fields
            assert content["title"] == ""
            assert content["meta_info"] == "<h2></h2>"
            assert "VIS" in content
            assert content["VIS"]["overview_words"] == 15
    finally:
        # Clean up
        os.unlink(filepath)


def test_info_stub_with_custom_properties():
    """Tests info_stub with custom properties.

    Verifies that info_stub correctly merges custom properties with
    default values in the output JSON.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name

    # Custom properties
    custom_props: dict[str, Any] = {
        "title": "Test Title",
        "VIS": {"overview_words": 20, "custom_setting": True},
        "extra_field": "extra value",
    }

    try:
        # Call the function with custom properties
        info_stub(filepath, custom_props)

        # Verify the file content
        with open(filepath, "r") as f:
            content = json.load(f)

            # Check custom fields
            assert content["title"] == "Test Title"
            assert content["meta_info"] == "<h2></h2>"  # Unchanged default
            assert content["VIS"]["overview_words"] == 20
            assert content["VIS"]["custom_setting"] is True
            assert content["extra_field"] == "extra value"
    finally:
        # Clean up
        os.unlink(filepath)


def test_info_stub_with_none_properties():
    """Tests info_stub with None properties.

    Verifies that info_stub correctly handles None for the properties
    parameter by using the default values.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name

    try:
        # Call the function with properties=None
        info_stub(filepath, None)

        # Verify the file content is the same as default
        with open(filepath, "r") as f:
            content = json.load(f)

            # Check default fields
            assert content["title"] == ""
            assert content["meta_info"] == "<h2></h2>"
            assert "VIS" in content
            assert content["VIS"]["overview_words"] == 15
    finally:
        # Clean up
        os.unlink(filepath)


def test_info_stub_overwrites_existing_file():
    """Tests that info_stub overwrites an existing file.

    Verifies that info_stub correctly overwrites an existing file
    instead of appending to it.
    """
    # Create a temporary file with some initial content
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
        temp_file.write('{"existing": "content"}')
        filepath = temp_file.name

    try:
        # Call the function
        info_stub(filepath)

        # Verify the file content was overwritten
        with open(filepath, "r") as f:
            content = json.load(f)

            # Check that old content is gone
            assert "existing" not in content

            # Check default fields
            assert content["title"] == ""
            assert content["meta_info"] == "<h2></h2>"
            assert "VIS" in content
    finally:
        # Clean up
        os.unlink(filepath)


def test_info_stub_handles_file_error():
    """Tests info_stub error handling for file operations.

    Verifies that info_stub correctly catches and reports file
    operation errors without crashing.
    """
    # Use an invalid file path
    invalid_filepath = "/nonexistent/directory/info.json"

    # Call the function
    with patch("builtins.print") as mock_print:
        info_stub(invalid_filepath)

        # Check that error message was printed
        mock_print.assert_called_once()
        assert "An error occurred" in mock_print.call_args[0][0]


def test_info_stub_handles_json_serialization_error():
    """Tests info_stub error handling for JSON serialization errors.

    Verifies that info_stub correctly catches and reports errors
    when trying to serialize unserializable objects.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name

    # Properties with unserializable object
    bad_props = {
        "title": "Test",
        "unserializable": set([1, 2, 3]),  # Sets can't be serialized to JSON
    }

    try:
        # Call the function
        with patch("builtins.print") as mock_print:
            info_stub(filepath, bad_props)

            # Check that error message was printed
            mock_print.assert_called_once()
            assert "An error occurred" in mock_print.call_args[0][0]

    finally:
        # Clean up
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_info_stub_with_nested_properties():
    """Tests info_stub with deeply nested properties.

    Verifies that info_stub correctly handles deeply nested
    property structures in the output JSON.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name

    # Nested properties
    nested_props = {"level1": {"level2": {"level3": {"value": 42}}}}

    try:
        # Call the function
        info_stub(filepath, nested_props)

        # Verify the file content
        with open(filepath, "r") as f:
            content = json.load(f)

            # Check nested fields
            assert content["level1"]["level2"]["level3"]["value"] == 42

            # Default fields should still be there
            assert content["title"] == ""
            assert content["meta_info"] == "<h2></h2>"
            assert content["VIS"]["overview_words"] == 15
    finally:
        # Clean up
        os.unlink(filepath)


def test_info_stub_json_indentation():
    """Tests that info_stub indents JSON output properly.

    Verifies that info_stub correctly formats the JSON output
    with the specified indentation.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name

    try:
        # Call the function
        info_stub(filepath)

        # Read the file as text to check formatting
        with open(filepath, "r") as f:
            content = f.read()

            # Check for indentation
            assert '    "title": ""' in content
            assert '    "VIS": {' in content
            assert '        "overview_words": 15' in content
    finally:
        # Clean up
        os.unlink(filepath)


#### Test transform_dt function


def test_transform_dt_basic():
    """Tests basic functionality of transform_dt.

    Tests the function with a simple document-topic matrix to verify
    the correct sparse matrix representation is produced.
    """
    # Simple document-topic matrix with 3 topics and 4 documents
    dt = [
        [2, 0, 1, 0],  # Topic 0
        [0, 3, 0, 1],  # Topic 1
        [1, 0, 2, 0],  # Topic 2
    ]

    result = transform_dt(dt)

    # Check structure
    assert "i" in result
    assert "p" in result
    assert "x" in result

    # Check p array (pointers)
    # p[0] = 0 (start)
    # p[1] = 2 (2 non-zero entries in topic 0)
    # p[2] = 4 (2 more non-zero entries in topic 1)
    # p[3] = 6 (2 more non-zero entries in topic 2)
    assert result["p"] == [0, 2, 4, 6]

    # Check document indices and values
    # For topic 0: doc 0 has value 2, doc 2 has value 1
    # For topic 1: doc 1 has value 3, doc 3 has value 1
    # For topic 2: doc 0 has value 1, doc 2 has value 2
    assert result["i"] == [0, 2, 1, 3, 0, 2]
    assert result["x"] == [2, 1, 3, 1, 1, 2]


def test_transform_dt_empty_matrix():
    """Tests transform_dt with an empty document-topic matrix.

    Verifies the function handles the case where the matrix has no documents.
    """
    # Empty document-topic matrix (no documents)
    dt = [[], [], []]  # 3 topics, 0 documents

    result = transform_dt(dt)

    assert result["p"] == [0, 0, 0, 0]  # One entry per topic + initial 0
    assert result["i"] == []
    assert result["x"] == []


def test_transform_dt_sparse_matrix():
    """Tests transform_dt with a very sparse document-topic matrix.

    Verifies the function correctly handles matrices where most entries are zero.
    """
    # Sparse matrix with 3 topics and 10 documents, but only a few non-zero entries
    dt = [
        [0, 0, 0, 5, 0, 0, 0, 0, 0, 0],  # Topic 0: only doc 3 has value
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],  # Topic 1: only doc 7 has value
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],  # Topic 2: only doc 9 has value
    ]

    result = transform_dt(dt)

    # Check p array
    assert result["p"] == [0, 1, 2, 3]

    # Check document indices and values
    assert result["i"] == [3, 7, 9]
    assert result["x"] == [5, 3, 2]


def test_transform_dt_dense_matrix():
    """Tests transform_dt with a dense document-topic matrix.

    Verifies the function correctly handles matrices where most entries are non-zero.
    """
    # Dense matrix with 2 topics and 3 documents, all non-zero
    dt = [
        [1, 2, 3],  # Topic 0
        [4, 5, 6],  # Topic 1
    ]

    result = transform_dt(dt)

    # Check p array
    assert result["p"] == [0, 3, 6]

    # Check document indices and values
    assert result["i"] == [0, 1, 2, 0, 1, 2]
    assert result["x"] == [1, 2, 3, 4, 5, 6]


def test_transform_dt_single_document():
    """Tests transform_dt with a document-topic matrix containing a single document.

    Verifies the function correctly handles matrices with only one document.
    """
    # Matrix with 3 topics and 1 document
    dt = [
        [5],  # Topic 0
        [0],  # Topic 1
        [2],  # Topic 2
    ]

    result = transform_dt(dt)

    # Check p array
    assert result["p"] == [0, 1, 1, 2]

    # Check document indices and values
    assert result["i"] == [0, 0]
    assert result["x"] == [5, 2]


def test_transform_dt_single_topic():
    """Tests transform_dt with a document-topic matrix containing a single topic.

    Verifies the function correctly handles matrices with only one topic.
    """
    # Matrix with 1 topic and 4 documents
    dt = [
        [0, 3, 0, 2]  # Topic 0
    ]

    result = transform_dt(dt)

    # Check p array
    assert result["p"] == [0, 2]

    # Check document indices and values
    assert result["i"] == [1, 3]
    assert result["x"] == [3, 2]


def test_transform_dt_all_zeros():
    """Tests transform_dt with a document-topic matrix containing only zeros.

    Verifies the function correctly handles matrices where all values are zero.
    """
    # Matrix with all zeros
    dt = [
        [0, 0, 0],  # Topic 0
        [0, 0, 0],  # Topic 1
        [0, 0, 0],  # Topic 2
    ]

    result = transform_dt(dt)

    # Check p array - all topics have 0 non-zero elements
    assert result["p"] == [0, 0, 0, 0]

    # Check document indices and values - should be empty
    assert result["i"] == []
    assert result["x"] == []


def test_transform_dt_sequential_topics():
    """Tests transform_dt with sequential non-zero entries across topics.

    Verifies the function correctly builds the sparse representation
    when non-zero entries appear in different documents for each topic.
    """
    # Each topic has non-zero entries for different documents
    dt = [
        [1, 2, 0, 0],  # Topic 0: docs 0, 1
        [0, 0, 3, 0],  # Topic 1: doc 2
        [0, 0, 0, 4],  # Topic 2: doc 3
    ]

    result = transform_dt(dt)

    # Check p array
    assert result["p"] == [0, 2, 3, 4]

    # Check document indices match the pattern described
    assert result["i"] == [0, 1, 2, 3]
    assert result["x"] == [1, 2, 3, 4]


#### Test transform_topic_weights function


def test_transform_topic_weights_basic():
    """Tests basic functionality of transform_topic_weights.

    Tests if the function correctly sorts words by their weights and returns
    the top n words and weights in the correct format.
    """
    # Sample inputs
    weights = [10, 5, 15, 7]
    vocab = {0: "apple", 1: "banana", 2: "cherry", 3: "date"}
    n = 3

    # Expected output: cherry (15), apple (10), date (7)
    expected = {"words": ["cherry", "apple", "date"], "weights": [15, 10, 7]}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_limit():
    """Tests transform_topic_weights with n limit.

    Tests that the function correctly limits the output to the top n items.
    """
    weights = [5, 10, 3, 8, 12]
    vocab = {0: "word0", 1: "word1", 2: "word2", 3: "word3", 4: "word4"}
    n = 2

    # Expected: only top 2 words by weight - word4 (12) and word1 (10)
    expected = {"words": ["word4", "word1"], "weights": [12, 10]}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_n_larger_than_input():
    """Tests transform_topic_weights when n exceeds input length.

    Verifies the function correctly handles cases where n is larger
    than the number of weights available.
    """
    weights = [5, 10, 3]
    vocab = {0: "word0", 1: "word1", 2: "word2"}
    n = 10  # Larger than input

    # Expected: all words, sorted by weight
    expected = {"words": ["word1", "word0", "word2"], "weights": [10, 5, 3]}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_tied_values():
    """Tests transform_topic_weights with tied weight values.

    Tests that the function handles ties in weights correctly. Note that
    the sorting algorithm's behavior with ties is implementation-dependent.
    """
    weights = [5, 10, 5, 10]
    vocab = {0: "word0", 1: "word1", 2: "word2", 3: "word3"}
    n = 4

    result = transform_topic_weights(weights, vocab, n)

    # Check structure and length
    assert "words" in result
    assert "weights" in result
    assert len(result["words"]) == 4
    assert len(result["weights"]) == 4

    # Check that the sorting worked (both 10s come before both 5s)
    assert set(result["weights"][:2]) == {10}
    assert set(result["weights"][2:]) == {5}


def test_transform_topic_weights_empty():
    """Tests transform_topic_weights with empty inputs.

    Tests that the function correctly handles empty weight and vocabulary lists.
    """
    weights = []
    vocab = {}
    n = 5

    expected = {"words": [], "weights": []}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_single_item():
    """Tests transform_topic_weights with a single item.

    Verifies the function works correctly with just one weight-vocabulary pair.
    """
    weights = [42]
    vocab = {0: "singleword"}
    n = 1

    expected = {"words": ["singleword"], "weights": [42]}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_dictionary_vocab():
    """Tests transform_topic_weights with a dictionary vocabulary.

    Verifies the function works correctly when the vocabulary is provided as a
    dictionary mapping indices to words, as happens in the parent function.
    """
    weights = [5, 10, 15]
    vocab = {0: "apple", 1: "banana", 2: "cherry"}
    n = 3

    expected = {"words": ["cherry", "banana", "apple"], "weights": [15, 10, 5]}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_zero_values():
    """Tests transform_topic_weights with zero values.

    Tests that the function correctly handles words with zero weights.
    """
    weights = [5, 0, 10, 0]
    vocab = {0: "word0", 1: "word1", 2: "word2", 3: "word3"}
    n = 4

    expected = {"words": ["word2", "word0", "word1", "word3"], "weights": [10, 5, 0, 0]}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_negative_values():
    """Tests transform_topic_weights with negative values.

    Tests that the function correctly sorts weights when negative values are present.
    """
    weights = [5, -10, 15, -3]
    vocab = {0: "word0", 1: "word1", 2: "word2", 3: "word3"}
    n = 4

    expected = {
        "words": ["word2", "word0", "word3", "word1"],
        "weights": [15, 5, -3, -10],
    }

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


def test_transform_topic_weights_n_zero():
    """Tests transform_topic_weights with n=0.

    Tests that the function returns empty lists when n is zero.
    """
    weights = [5, 10, 15]
    vocab = {0: "word0", 1: "word1", 2: "word2"}
    n = 0

    expected = {"words": [], "weights": []}

    result = transform_topic_weights(weights, vocab, n)
    assert result == expected


#### Test write_dt function


def test_write_dt_basic():
    """Tests basic functionality of write_dt.

    Creates a temporary output file and verifies that the function:
    1. Creates a zip file at the specified path
    2. Adds a file named "dt.json" to the zip
    3. The content of dt.json matches the JSON serialization of the input dictionary
    """
    # Create a simple document-topic matrix
    dtj = {"i": [0, 1, 2], "p": [0, 3], "x": [10, 20, 30]}

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print"):
            write_dt(dtj, output_file)

        # Verify zip file exists and contains correct content
        with zf.ZipFile(output_file, "r") as zip_ref:
            # Check if dt.json is in the zip
            assert "dt.json" in zip_ref.namelist()

            # Extract and verify content
            with zip_ref.open("dt.json") as dt_file:
                content = json.loads(dt_file.read())
                assert content == dtj
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_dt_complex():
    """Tests write_dt with a more complex document-topic matrix.

    Tests the function with a larger, more complex dictionary that represents
    a realistic document-topic matrix.
    """
    # Create a more complex document-topic matrix
    dtj = {
        "i": [0, 1, 2, 3, 10, 11, 12, 20, 21, 30],
        "p": [0, 3, 7, 9, 10],
        "x": [0.3, 0.5, 0.2, 0.4, 0.6, 0.8, 0.1, 0.7, 0.9, 0.5],
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print"):
            write_dt(dtj, output_file)

        # Verify zip file exists and contains correct content
        with zf.ZipFile(output_file, "r") as zip_ref:
            # Check if dt.json is in the zip
            assert "dt.json" in zip_ref.namelist()

            # Extract and verify content
            with zip_ref.open("dt.json") as dt_file:
                content = json.loads(dt_file.read())
                assert content == dtj
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_dt_overwrite_existing():
    """Tests write_dt overwrites an existing zip file.

    Verifies that the function properly overwrites an existing zip file
    rather than appending to it.
    """
    # Create initial and replacement document-topic matrices
    initial_dtj = {"i": [0], "p": [0, 1], "x": [100]}
    replacement_dtj = {"i": [1], "p": [0, 1], "x": [200]}

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
        output_file = temp.name

    try:
        # First, create a zip file with initial content
        with zf.ZipFile(output_file, "w") as z:
            z.writestr("dt.json", json.dumps(initial_dtj))
            z.writestr("extra.txt", "This file should be removed")

        # Now call write_dt to overwrite it
        with patch("builtins.print"):
            write_dt(replacement_dtj, output_file)

        # Verify the zip file was overwritten (not appended to)
        with zf.ZipFile(output_file, "r") as zip_ref:
            # Should only contain dt.json
            assert zip_ref.namelist() == ["dt.json"]

            # Content should be the replacement data
            with zip_ref.open("dt.json") as dt_file:
                content = json.loads(dt_file.read())
                assert content == replacement_dtj
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_dt_empty_dict():
    """Tests write_dt with an empty dictionary.

    Verifies that the function correctly handles an empty document-topic matrix.
    """
    # Empty document-topic matrix
    dtj = {"i": [], "p": [], "x": []}

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print"):
            write_dt(dtj, output_file)

        # Verify zip file exists and contains correct content
        with zf.ZipFile(output_file, "r") as zip_ref:
            # Extract and verify content
            with zip_ref.open("dt.json") as dt_file:
                content = json.loads(dt_file.read())
                assert content == dtj
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_dt_print_message():
    """Tests that write_dt prints the expected message.

    Verifies that the function outputs the correct message indicating
    where the document-topic matrix was written.
    """
    dtj = {"i": [0], "p": [0, 1], "x": [10]}
    output_file = "test_output.zip"

    # Mock both the zipfile creation and the print function
    with (
        patch("zipfile.ZipFile") as mock_zipfile,
        patch("builtins.print") as mock_print,
    ):
        # Configure the mock
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        # Call the function
        write_dt(dtj, output_file)

        # Verify the print message
        mock_print.assert_called_once_with(f"Wrote sparse doc-topics to {output_file}")


def test_write_dt_json_serialization():
    """Tests that write_dt correctly serializes the dictionary to JSON.

    Verifies that the function properly converts the dictionary to JSON
    before writing it to the zip file.
    """
    # Document-topic matrix with various Python types
    dtj = {
        "i": [0, 1, 2],
        "p": [0, 3],
        "x": [10.5, 20.3, 30.7],
        "metadata": {"description": "Test data", "created": "2025-04-24"},
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print"):
            write_dt(dtj, output_file)

        # Verify JSON serialization was correct
        with zf.ZipFile(output_file, "r") as zip_ref:
            with zip_ref.open("dt.json") as dt_file:
                # Read as text to verify JSON formatting
                content = dt_file.read().decode("utf-8")
                # Parse back to verify valid JSON
                parsed = json.loads(content)
                assert parsed == dtj
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


#### Test write_tw function


def test_write_tw_basic():
    """Tests basic functionality of write_tw.

    This test verifies that write_tw correctly writes a JSON file with the
    expected structure and content.
    """
    # Test data
    alpha = [0.1, 0.2, 0.3]
    tw = [
        {"words": ["apple", "banana"], "weights": [10, 5]},
        {"words": ["cat", "dog"], "weights": [8, 7]},
        {"words": ["red", "blue"], "weights": [12, 6]},
    ]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print") as mock_print:
            write_tw(alpha, tw, output_file)

        # Verify file exists
        assert os.path.exists(output_file)

        # Read back the file and verify content
        with open(output_file, "r") as f:
            content = json.load(f)
            assert content == {"alpha": alpha, "tw": tw}

        # Verify print message
        mock_print.assert_called_once()
        assert (
            f"Wrote topic-words information to {output_file}"
            in mock_print.call_args[0][0]
        )
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_tw_empty_data():
    """Tests write_tw with empty data.

    This test verifies that write_tw correctly handles empty alpha and tw values.
    """
    # Empty test data
    alpha = []
    tw = []

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print"):
            write_tw(alpha, tw, output_file)

        # Read back the file and verify content
        with open(output_file, "r") as f:
            content = json.load(f)
            assert content == {"alpha": alpha, "tw": tw}
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_tw_overwrite_existing():
    """Tests write_tw overwrites an existing file.

    This test verifies that write_tw correctly overwrites an existing file
    rather than appending to it.
    """
    # Initial and new data
    initial_alpha = [0.1]
    initial_tw = [{"words": ["old"], "weights": [1]}]

    new_alpha = [0.2, 0.3]
    new_tw = [{"words": ["new"], "weights": [2]}]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
        output_file = temp.name

    try:
        # First, write the initial data
        with open(output_file, "w") as f:
            json.dump({"alpha": initial_alpha, "tw": initial_tw}, f)

        # Call the function with new data
        with patch("builtins.print"):
            write_tw(new_alpha, new_tw, output_file)

        # Read back the file and verify it contains the new data
        with open(output_file, "r") as f:
            content = json.load(f)
            assert content == {"alpha": new_alpha, "tw": new_tw}
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_tw_complex_data():
    """Tests write_tw with complex nested data.

    This test verifies that write_tw correctly handles complex nested
    data structures in the topic-word matrix.
    """
    # More complex test data
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
    tw = [
        {
            "words": ["apple", "banana", "cherry", "date", "elderberry"],
            "weights": [10.5, 8.3, 7.1, 5.9, 4.7],
            "metadata": {"category": "fruits", "language": "english"},
        },
        {
            "words": ["cat", "dog", "elephant", "fox", "giraffe"],
            "weights": [9.8, 8.6, 7.4, 6.2, 5.0],
            "metadata": {"category": "animals", "language": "english"},
        },
    ]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
        output_file = temp.name

    try:
        # Call the function
        with patch("builtins.print"):
            write_tw(alpha, tw, output_file)

        # Read back the file and verify content
        with open(output_file, "r") as f:
            content = json.load(f)
            assert content == {"alpha": alpha, "tw": tw}
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_tw_file_error():
    """Tests write_tw error handling for file operations.

    This test verifies that attempting to write to an invalid location
    raises the appropriate exception.
    """
    # Test data
    alpha = [0.1, 0.2]
    tw = [{"words": ["test"], "weights": [1]}]

    # Invalid file path
    invalid_path = "/nonexistent/directory/output.json"

    # Attempt to write to invalid path should raise an exception
    with pytest.raises(FileNotFoundError):
        write_tw(alpha, tw, invalid_path)


def test_write_tw_print_message():
    """Tests that write_tw prints the expected message.

    This test verifies that the function outputs the correct message indicating
    where the topic-word matrix was written.
    """
    # Test data
    alpha = [0.1]
    tw = [{"words": ["test"], "weights": [1]}]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
        output_file = temp.name

    try:
        # Call the function with print mocked
        with patch("builtins.print") as mock_print:
            write_tw(alpha, tw, output_file)

            # Verify print was called with the correct message
            expected_message = f"Wrote topic-words information to {output_file}"
            mock_print.assert_called_once_with(expected_message)
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)
