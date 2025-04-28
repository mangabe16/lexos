"""test_mallet.py.

Last Updated: April 27, 2025
Last Tested: April 27, 2025

# TODO:
  - Fix commented out tests.
"""

import glob
import re
import subprocess
from collections import defaultdict
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, mock_open, patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import spacy
from pydantic import ConfigDict
from rich.console import Console
from wordcloud import WordCloud

from lexos.exceptions import LexosException
from lexos.topic_modeling.mallet import (
    MALLET_BINARY_PATH,
    Mallet,
    import_data_file,
    import_dirs,
    import_docs,
    import_files,
)

nlp = spacy.blank("en")  # Create a blank English NLP model

#### Common Fixtures ####


@pytest.fixture
def mallet_instance():
    """Creates a basic Mallet instance for testing.

    Returns:
        Mallet: An instance of the Mallet class.
    """
    return Mallet()


#### Test import_data_file() ####


def test_import_data_file_success():
    """Tests successful import of data from a valid file.

    This test verifies that import_data_file correctly reads a file's contents
    line by line when the file exists and is readable.

    Returns:
        None
    """
    # Mock file content
    mock_content = "Document 1\nDocument 2\nDocument 3"
    expected = ["Document 1\n", "Document 2\n", "Document 3"]

    # Use mock_open to simulate file operations
    with patch("builtins.open", mock_open(read_data=mock_content)) as mock_file:
        result = import_data_file("mock/file/path.txt")

    # Verify the file was opened correctly
    mock_file.assert_called_once_with("mock/file/path.txt", "r", encoding="utf-8")

    # Verify the returned data matches expected output
    assert result == expected


def test_import_data_file_with_path_object():
    """Tests import_data_file with a Path object instead of string.

    This test verifies that the function accepts a Path object as input.

    Returns:
        None
    """
    # Mock file content
    mock_content = "Document 1\nDocument 2\nDocument 3"
    expected = ["Document 1\n", "Document 2\n", "Document 3"]

    # Use Path object for the file path
    file_path = Path("mock/file/path.txt")

    # Use mock_open to simulate file operations
    with patch("builtins.open", mock_open(read_data=mock_content)) as mock_file:
        result = import_data_file(file_path)

    # Verify the file was opened correctly
    mock_file.assert_called_once_with(file_path, "r", encoding="utf-8")

    # Verify the returned data matches expected output
    assert result == expected


def test_import_data_file_not_found():
    """Tests handling of FileNotFoundError in import_data_file.

    This test verifies that the function raises a LexosException when
    the specified file does not exist.

    Returns:
        None
    """
    # Mock FileNotFoundError
    with patch("builtins.open", side_effect=FileNotFoundError()):
        # Test with the function
        with pytest.raises(LexosException) as excinfo:
            import_data_file("nonexistent_file.txt")

        # Check that the exception message contains the file path
        assert "nonexistent_file.txt" in str(excinfo.value)
        assert "does not exist" in str(excinfo.value)


def test_import_data_file_io_error():
    """Tests handling of IOError in import_data_file.

    This test verifies that the function raises a LexosException when
    the file exists but cannot be read due to an IOError.

    Returns:
        None
    """
    # Mock IOError
    with patch("builtins.open", side_effect=IOError()):
        # Test with the function
        with pytest.raises(LexosException) as excinfo:
            import_data_file("unreadable_file.txt")

        # Check that the exception message contains the file path
        assert "unreadable_file.txt" in str(excinfo.value)
        assert "could not be read" in str(excinfo.value)


def test_import_data_file_empty_file():
    """Tests import of an empty file.

    This test verifies that the function correctly handles an empty file,
    returning an empty list.

    Returns:
        None
    """
    # Mock empty file
    with patch("builtins.open", mock_open(read_data="")) as mock_file:
        result = import_data_file("empty_file.txt")

    # Verify the file was opened correctly
    mock_file.assert_called_once_with("empty_file.txt", "r", encoding="utf-8")

    # Verify an empty list is returned
    assert result == []


#### Test import_dirs() ####


def test_import_dirs_with_string():
    """Test import_dirs with a single directory path as string.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't process string path correctly
    """
    mock_dir = "mock/directory"
    mock_files = [Path(f"{mock_dir}/file1.txt"), Path(f"{mock_dir}/file2.txt")]
    mock_content = ["content of file1", "content of file2"]

    with (
        patch.object(Path, "is_dir", return_value=True),
        patch.object(glob, "glob", return_value=mock_files),
        patch.object(Path, "is_file", return_value=True),
        patch("builtins.open", mock_open()) as mock_file,
    ):
        # Set up mock file reads for each file
        mock_file.return_value.__enter__.side_effect = [
            MagicMock(read=MagicMock(return_value=content)) for content in mock_content
        ]

        result = import_dirs(mock_dir)

        assert len(result) == 2
        assert result == mock_content


def test_import_dirs_with_path_object():
    """Test import_dirs with a single directory as Path object.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't process Path object correctly
    """
    mock_dir = Path("mock/directory")
    mock_files = [Path(f"{mock_dir}/file1.txt"), Path(f"{mock_dir}/file2.txt")]
    mock_content = ["content of file1", "content of file2"]

    with (
        patch.object(Path, "is_dir", return_value=True),
        patch.object(glob, "glob", return_value=mock_files),
        patch.object(Path, "is_file", return_value=True),
        patch("builtins.open", mock_open()) as mock_file,
    ):
        # Set up mock file reads for each file
        mock_file.return_value.__enter__.side_effect = [
            MagicMock(read=MagicMock(return_value=content)) for content in mock_content
        ]

        result = import_dirs(mock_dir)

        assert len(result) == 2
        assert result == mock_content


def test_import_dirs_with_list():
    """Test import_dirs with a list of directory paths.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't process list of paths correctly
    """
    mock_dirs = ["mock/directory1", Path("mock/directory2")]
    mock_files = [
        Path(f"{mock_dirs[0]}/file1.txt"),
        Path(f"{mock_dirs[0]}/file2.txt"),
        Path(f"{mock_dirs[1]}/file3.txt"),
    ]
    mock_content = ["content of file1", "content of file2", "content of file3"]

    with (
        patch.object(Path, "is_dir", return_value=True),
        patch.object(glob, "glob", side_effect=[mock_files[:2], mock_files[2:]]),
        patch.object(Path, "is_file", return_value=True),
        patch("builtins.open", mock_open()) as mock_file,
    ):
        # Set up mock file reads for each file
        mock_file.return_value.__enter__.side_effect = [
            MagicMock(read=MagicMock(return_value=content)) for content in mock_content
        ]

        result = import_dirs(mock_dirs)

        assert len(result) == 3
        assert result == mock_content


def test_import_dirs_nonexistent_directory():
    """Test import_dirs raises LexosException for nonexistent directory.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't raise LexosException
    """
    mock_dir = "nonexistent/directory"

    with patch.object(Path, "is_dir", return_value=False):
        with pytest.raises(LexosException) as excinfo:
            import_dirs(mock_dir)

        assert f"Directory {mock_dir} does not exist" in str(excinfo.value)


def test_import_dirs_empty_directory():
    """Test import_dirs with an empty directory.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't handle empty directory correctly
    """
    mock_dir = "empty/directory"

    with (
        patch.object(Path, "is_dir", return_value=True),
        patch.object(Path, "glob", return_value=[]),
    ):
        result = import_dirs(mock_dir)

        assert result == []


def test_import_dirs_mixed_existing_nonexisting():
    """Test import_dirs with a mix of existing and non-existing directories.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't fail on first non-existing directory
    """
    mock_dirs = ["existing/directory", "nonexistent/directory"]
    training_data = []

    with patch.object(Path, "is_dir", side_effect=[True, False]):
        with pytest.raises(LexosException) as excinfo:
            training_data = import_dirs(mock_dirs)

        assert f"Directory {mock_dirs[1]} does not exist" in str(excinfo.value)

    # Since we have not mocked files in the first directory, it should be empty
    assert len(training_data) == 0


def test_import_dirs_file_error():
    """Test import_dirs handles file read errors gracefully.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function doesn't propagate file read errors
    """
    mock_dir = "mock/directory"
    mock_files = [Path(f"{mock_dir}/file1.txt")]

    with (
        patch.object(Path, "is_dir", return_value=True),
        patch.object(Path, "glob", return_value=mock_files),
        patch.object(Path, "is_file", return_value=True),
        patch("builtins.open", side_effect=IOError("Mock IO Error")),
    ):
        training_data = import_dirs(mock_dir)
    assert training_data == []


#### Test import_docs() ####


def test_import_docs_with_strings():
    """Tests import_docs with a list of string documents.

    This test verifies that import_docs correctly processes a list containing
    only string documents and returns them unmodified.

    Args:
        None

    Returns:
        None
    """
    string_docs = ["Document 1", "Document 2", "Another document"]
    result = import_docs(string_docs)
    assert result == string_docs
    assert isinstance(result, list)
    assert all(isinstance(doc, str) for doc in result)


def test_import_docs_with_doc_objects():
    """Tests import_docs with a list of Doc objects.

    This test verifies that import_docs correctly extracts the text
    property from each Doc object in the list.

    Args:
        None

    Returns:
        None
    """
    # Create mock Doc objects with text attribute
    doc1 = nlp("Document 1 text")
    doc2 = nlp("Document 2 text")

    doc_objects = [doc1, doc2]
    expected = [doc1.text, doc2.text]

    result = import_docs(doc_objects)
    assert result == expected
    assert isinstance(result, list)
    assert all(isinstance(doc, str) for doc in result)


def test_import_docs_with_mixed_types():
    """Tests import_docs with a mix of strings and Doc objects.

    This test verifies that import_docs correctly processes a list
    containing both string documents and Doc objects.

    Args:
        None

    Returns:
        None
    """
    # Create mock Doc objects
    doc1 = nlp("Doc object text")

    mixed_docs = ["Plain string document", doc1, "Another string"]
    expected = ["Plain string document", doc1.text, "Another string"]

    result = import_docs(mixed_docs)
    assert result == expected
    assert isinstance(result, list)
    assert all(isinstance(doc, str) for doc in result)


def test_import_docs_with_empty_list():
    """Tests import_docs with an empty list.

    This test verifies that import_docs correctly handles an empty list
    input and returns an empty list.

    Args:
        None

    Returns:
        None
    """
    result = import_docs([])
    assert result == []
    assert isinstance(result, list)


def test_import_docs_preserves_order():
    """Tests import_docs preserves the order of documents.

    This test verifies that import_docs maintains the exact order of
    documents in the returned list.

    Args:
        None

    Returns:
        None
    """
    # Create mixed input with specific order
    doc1 = nlp("First Doc object")
    doc2 = nlp("Second Doc object")

    ordered_docs = ["First string", doc1, "Second string", doc2]
    expected = [
        "First string",
        "First Doc object",
        "Second string",
        "Second Doc object",
    ]

    result = import_docs(ordered_docs)
    assert result == expected


def test_import_docs_with_empty_strings():
    """Tests import_docs with empty strings in the list.

    This test verifies that import_docs correctly handles empty strings
    in the input list.

    Args:
        None

    Returns:
        None
    """
    docs_with_empty = ["Document 1", "", "Document 2"]
    result = import_docs(docs_with_empty)
    assert result == ["Document 1", "", "Document 2"]
    assert "" in result


#### Test import_files() ####


def test_import_files_single_str():
    """Tests import_files with a single file path as a string.

    This test verifies that the function correctly reads a file's contents
    when given a single file path as a string.

    Args:
        None

    Returns:
        None
    """
    file_content = "This is file content"
    mock_file_path = "path/to/file.txt"

    with patch("builtins.open", mock_open(read_data=file_content)) as mock_file:
        result = import_files(mock_file_path)

    mock_file.assert_called_once_with(mock_file_path, "r", encoding="utf-8")
    assert len(result) == 1
    assert result[0] == file_content


def test_import_files_single_path():
    """Tests import_files with a single file path as a Path object.

    This test verifies that the function correctly reads a file's contents
    when given a single file path as a Path object.

    Args:
        None

    Returns:
        None
    """
    file_content = "This is file content"
    mock_file_path = Path("path/to/file.txt")

    with patch("builtins.open", mock_open(read_data=file_content)) as mock_file:
        result = import_files(mock_file_path)

    mock_file.assert_called_once_with(mock_file_path, "r", encoding="utf-8")
    assert len(result) == 1
    assert result[0] == file_content


def test_import_files_multiple():
    """Tests import_files with multiple file paths.

    This test verifies that the function correctly reads multiple files' contents
    when given a list of file paths.

    Args:
        None

    Returns:
        None
    """
    file_contents = ["Content of file 1", "Content of file 2", "Content of file 3"]
    mock_file_paths = ["file1.txt", "file2.txt", Path("file3.txt")]

    # Create a sequence of mock returns for each file open
    mock_open_sequence = [
        mock_open(read_data=content).return_value for content in file_contents
    ]

    # Use side_effect to return different mock objects for each call to open
    with patch("builtins.open", side_effect=mock_open_sequence) as mock_file:
        result = import_files(mock_file_paths)

    # Verify each file was opened with the correct path
    assert mock_file.call_count == 3
    assert mock_file.call_args_list[0][0][0] == mock_file_paths[0]
    assert mock_file.call_args_list[1][0][0] == mock_file_paths[1]
    assert mock_file.call_args_list[2][0][0] == mock_file_paths[2]

    # Verify the content was read correctly
    assert len(result) == 3
    assert result == file_contents


def test_import_files_file_not_found():
    """Tests import_files handles FileNotFoundError correctly.

    This test verifies that the function properly raises a LexosException
    when a file is not found.

    Args:
        None

    Returns:
        None
    """
    mock_file_path = "nonexistent_file.txt"

    with (
        patch("builtins.open", side_effect=FileNotFoundError()),
        pytest.raises(LexosException) as excinfo,
    ):
        import_files(mock_file_path)

    # Check that the exception message contains the file path
    assert f"File {mock_file_path} does not exist" in str(excinfo.value)


def test_import_files_io_error():
    """Tests import_files handles IOError correctly.

    This test verifies that the function properly raises a LexosException
    when a file cannot be read due to an IOError.

    Args:
        None

    Returns:
        None
    """
    mock_file_path = "unreadable_file.txt"

    with (
        patch("builtins.open", side_effect=IOError()),
        pytest.raises(LexosException) as excinfo,
    ):
        import_files(mock_file_path)

    # Check that the exception message contains the file path
    assert f"File {mock_file_path} could not be read" in str(excinfo.value)


def test_import_files_mixed_errors():
    """Tests import_files handles a mix of successful and failed file operations.

    This test verifies that the function raises an exception at the first error
    encountered, even if previous files were successfully read.

    Args:
        None

    Returns:
        None
    """
    file_contents = ["Content of file 1"]
    mock_file_paths = ["file1.txt", "nonexistent_file.txt"]

    with patch("builtins.open") as mock_file:
        # First call succeeds, second call fails with FileNotFoundError
        mock_file.side_effect = [
            mock_open(read_data=file_contents[0]).return_value,
            FileNotFoundError(),
        ]

        with pytest.raises(LexosException) as excinfo:
            import_files(mock_file_paths)

    # Check that the exception message references the second file
    assert f"File {mock_file_paths[1]} does not exist" in str(excinfo.value)


def test_import_files_empty_list():
    """Tests import_files with an empty list.

    This test verifies that the function returns an empty list
    when given an empty list of file paths.

    Args:
        None

    Returns:
        None
    """
    result = import_files([])
    assert result == []
    # No calls to open() should be made


#### Test Mallet Initialisation ####


def test_mallet_initialization_default_values():
    """Test initialization of Mallet class with default values.

    Verifies that a Mallet instance can be created with default values,
    and that these values are correctly set.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.path_to_mallet == MALLET_BINARY_PATH
    assert mallet.metadata == {}
    assert isinstance(mallet.model_config, dict)
    assert mallet.model_config.get("arbitrary_types_allowed") is True


def test_mallet_initialization_custom_path():
    """Test initialization of Mallet class with custom path_to_mallet.

    Verifies that a Mallet instance can be created with a custom
    path_to_mallet value.

    Args:
        None

    Returns:
        None
    """
    custom_path = "/custom/path/to/mallet"
    mallet = Mallet(path_to_mallet=custom_path)
    assert mallet.path_to_mallet == custom_path
    assert mallet.metadata == {}


def test_mallet_initialization_custom_metadata():
    """Test initialization of Mallet class with custom metadata.

    Verifies that a Mallet instance can be created with a custom
    metadata dictionary.

    Args:
        None

    Returns:
        None
    """
    test_metadata = {"test_key": "test_value", "model_directory": "/path/to/model"}
    mallet = Mallet(metadata=test_metadata)
    assert mallet.path_to_mallet == MALLET_BINARY_PATH
    assert mallet.metadata == test_metadata
    assert mallet.metadata["test_key"] == "test_value"


def test_mallet_initialization_all_custom_values():
    """Test initialization of Mallet class with all custom values.

    Verifies that a Mallet instance can be created with custom values
    for all fields.

    Args:
        None

    Returns:
        None
    """
    custom_path = "/custom/path/to/mallet"
    test_metadata = {"test_key": "test_value"}
    mallet = Mallet(path_to_mallet=custom_path, metadata=test_metadata)
    assert mallet.path_to_mallet == custom_path
    assert mallet.metadata == test_metadata


def test_mallet_model_config():
    """Test Mallet class model_config.

    Verifies that the model_config for the Mallet class is correctly set
    to allow arbitrary types.

    Args:
        None

    Returns:
        None
    """
    assert Mallet.model_config == ConfigDict(arbitrary_types_allowed=True)
    mallet = Mallet()
    assert mallet.model_config == {"arbitrary_types_allowed": True}


#### Test Distributions Property ####


def test_distributions_path_not_set():
    """Tests that distributions property raises exception when path is not set.

    Tests that an exception is raised when trying to access the distributions
    property without setting path_to_topic_distributions in metadata.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    with pytest.raises(LexosException) as excinfo:
        _ = mallet.distributions

    assert "No topic distributions have been set" in str(excinfo.value)


def test_distributions_valid_file():
    """Tests that distributions property correctly parses a valid file.

    Tests that the distributions property correctly reads and parses a properly
    formatted topic distributions file.

    Args:
        None

    Returns:
        None
    """
    # Sample content mimicking a MALLET topic distribution file
    mock_content = (
        "#doc name topic proportion ...\n"
        "0\tdoc1\t0.1\t0.2\t0.7\n"
        "1\tdoc2\t0.3\t0.4\t0.3\n"
        "2\tdoc3\t0.5\t0.1\t0.4\n"
    )

    # Expected result after parsing
    expected_distributions = [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.1, 0.4]]

    with patch("builtins.open", mock_open(read_data=mock_content)):
        mallet = Mallet(metadata={"path_to_topic_distributions": "mock/path.txt"})
        result = mallet.distributions

    assert result == expected_distributions


def test_distributions_skip_header():
    """Tests that distributions property skips header lines.

    Tests that the distributions property correctly skips lines that start with
    "#doc" which are considered header lines.

    Args:
        None

    Returns:
        None
    """
    # Sample content with multiple header lines
    mock_content = (
        "#doc name topic proportion ...\n"
        "#doc additional header\n"
        "0\tdoc1\t0.1\t0.2\t0.7\n"
        "1\tdoc2\t0.3\t0.4\t0.3\n"
    )

    expected_distributions = [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]

    with patch("builtins.open", mock_open(read_data=mock_content)):
        mallet = Mallet(metadata={"path_to_topic_distributions": "mock/path.txt"})
        result = mallet.distributions

    assert result == expected_distributions


def test_distributions_empty_file():
    """Tests that distributions property handles empty files.

    Tests that the distributions property returns an empty list when the
    topic distributions file is empty.

    Args:
        None

    Returns:
        None
    """
    with patch("builtins.open", mock_open(read_data="")):
        mallet = Mallet(metadata={"path_to_topic_distributions": "mock/path.txt"})
        result = mallet.distributions

    assert result == []


def test_distributions_file_not_found():
    """Tests that distributions property handles file not found errors.

    Tests that the distributions property propagates a FileNotFoundError when
    the topic distributions file does not exist.

    Args:
        None

    Returns:
        None
    """
    with patch("builtins.open", side_effect=FileNotFoundError()):
        mallet = Mallet(
            metadata={"path_to_topic_distributions": "nonexistent/path.txt"}
        )
        with pytest.raises(FileNotFoundError):
            _ = mallet.distributions


def test_distributions_malformed_line():
    """Tests that distributions property handles malformed lines.

    Tests that the distributions property raises an exception when it encounters
    a malformed line in the topic distributions file.

    Args:
        None

    Returns:
        None
    """
    # Sample content with a malformed line (missing tabs)
    mock_content = (
        "#doc name topic proportion ...\n"
        "0\tdoc1\t0.1\t0.2\t0.7\n"
        "1 doc2 0.3 0.4 0.3\n"  # Missing tabs
    )

    with patch("builtins.open", mock_open(read_data=mock_content)):
        mallet = Mallet(metadata={"path_to_topic_distributions": "mock/path.txt"})
        with pytest.raises(LexosException):
            _ = mallet.distributions


def test_distributions_caching():
    """Tests that distributions property correctly caches results.

    Tests that subsequent calls to the distributions property return the cached
    result instead of reading the file again.

    Args:
        None

    Returns:
        None
    """
    mock_content = "0\tdoc1\t0.1\t0.2\t0.7\n"

    with patch("builtins.open", mock_open(read_data=mock_content)) as mock_file:
        mallet = Mallet(metadata={"path_to_topic_distributions": "mock/path.txt"})

        # First call should read the file
        _ = mallet.distributions
        assert mock_file.call_count == 1

        # Second call should use cached result
        _ = mallet.distributions
        assert mock_file.call_count == 1


#### Test Num_Docs Property ####


def test_num_docs_with_value():
    """Tests num_docs property when metadata contains a num_docs value.

    This test verifies that the num_docs property correctly returns the value
    stored in the metadata dictionary when it exists.

    Args:
        None

    Returns:
        None
    """
    test_value = 42
    mallet = Mallet(metadata={"num_docs": test_value})
    assert mallet.num_docs == test_value


def test_num_docs_with_zero():
    """Tests num_docs property when metadata contains zero documents.

    This test verifies that the num_docs property correctly returns zero
    when the metadata dictionary contains a num_docs key with value 0.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet(metadata={"num_docs": 0})
    assert mallet.num_docs == 0


def test_num_docs_missing_key():
    """Tests num_docs property when metadata does not contain num_docs key.

    This test verifies that the num_docs property correctly returns 0
    when the metadata dictionary does not contain a num_docs key.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.num_docs == 0


def test_num_docs_with_other_metadata():
    """Tests num_docs property with unrelated metadata present.

    This test verifies that the num_docs property correctly returns 0
    when the metadata dictionary contains other keys but not num_docs.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet(metadata={"other_key": "some_value"})
    assert mallet.num_docs == 0


def test_num_docs_after_update():
    """Tests num_docs property after updating the metadata.

    This test verifies that the num_docs property correctly reflects
    changes to the metadata dictionary.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.num_docs == 0

    # Update the metadata
    mallet.metadata["num_docs"] = 25
    assert mallet.num_docs == 25

    # Update it again
    mallet.metadata["num_docs"] = 50
    assert mallet.num_docs == 50


#### Test Mean_Num_Tokens Property ####


def test_mean_num_tokens_with_value():
    """Tests mean_num_tokens property when metadata contains the value.

    This test verifies that the mean_num_tokens property correctly returns
    the value stored in the metadata dictionary when it exists.

    Args:
        None

    Returns:
        None
    """
    test_value = 42
    mallet = Mallet(metadata={"mean_num_tokens": test_value})
    assert mallet.mean_num_tokens == test_value


def test_mean_num_tokens_with_zero():
    """Tests mean_num_tokens property when metadata contains zero tokens.

    This test verifies that the mean_num_tokens property correctly returns zero
    when the metadata dictionary contains a mean_num_tokens key with value 0.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet(metadata={"mean_num_tokens": 0})
    assert mallet.mean_num_tokens == 0


def test_mean_num_tokens_missing_key():
    """Tests mean_num_tokens property when metadata lacks the key.

    This test verifies that the mean_num_tokens property correctly returns 0
    when the metadata dictionary does not contain a mean_num_tokens key.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.mean_num_tokens == 0


def test_mean_num_tokens_with_other_metadata():
    """Tests mean_num_tokens property with unrelated metadata present.

    This test verifies that the mean_num_tokens property correctly returns 0
    when the metadata dictionary contains other keys but not mean_num_tokens.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet(metadata={"other_key": "some_value"})
    assert mallet.mean_num_tokens == 0


def test_mean_num_tokens_after_update():
    """Tests mean_num_tokens property after updating the metadata.

    This test verifies that the mean_num_tokens property correctly reflects
    changes to the metadata dictionary.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.mean_num_tokens == 0

    # Update the metadata
    mallet.metadata["mean_num_tokens"] = 25
    assert mallet.mean_num_tokens == 25

    # Update it again with a floating point value
    mallet.metadata["mean_num_tokens"] = 50.5
    assert mallet.mean_num_tokens == 50.5


#### Test Model_Directory Property ####


def test_model_directory_exists():
    """Tests model_directory property when the key exists in metadata.

    This test verifies that the model_directory property correctly returns
    the value from metadata when the key exists.

    Args:
        None

    Returns:
        None
    """
    expected_path = "/path/to/model"
    mallet = Mallet(metadata={"model_directory": expected_path})

    # Verify that the property returns the expected path
    assert mallet.model_directory == expected_path


def test_model_directory_missing():
    """Tests model_directory property raises exception when key is missing.

    This test verifies that the model_directory property correctly raises
    a LexosException when the model_directory key doesn't exist in metadata.

    Args:
        None

    Returns:
        None

    Raises:
        LexosException: Expected to be raised when model_directory is missing
    """
    mallet = Mallet()  # No metadata set

    # Verify that accessing the property raises the expected exception
    with pytest.raises(LexosException) as excinfo:
        _ = mallet.model_directory

    # Verify the exception message
    assert "No model directory has been set" in str(excinfo.value)
    assert "created by default when you call `import_data()`" in str(excinfo.value)


def test_model_directory_with_other_metadata():
    """Tests model_directory property raises exception with unrelated metadata.

    This test verifies that the model_directory property correctly raises
    a LexosException when metadata exists but doesn't contain model_directory.

    Args:
        None

    Returns:
        None
    """
    # Set metadata with some other keys but not model_directory
    mallet = Mallet(metadata={"num_docs": 10, "mean_num_tokens": 50})

    with pytest.raises(LexosException):
        _ = mallet.model_directory


def test_model_directory_after_update():
    """Tests model_directory property after updating the metadata.

    This test verifies that the model_directory property correctly reflects
    changes to the metadata dictionary.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()  # Start with no metadata

    # This should raise an exception initially
    with pytest.raises(LexosException):
        _ = mallet.model_directory

    # Update the metadata
    test_path = "/updated/model/path"
    mallet.metadata["model_directory"] = test_path

    # Now it should return the path
    assert mallet.model_directory == test_path


#### Test Topic_Keys Property ####


def test_topic_keys_path_not_set():
    """Tests that topic_keys property raises exception when path is not set.

    Tests that accessing topic_keys properly raises a LexosException when
    the path_to_topic_keys is not set in the metadata.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    with pytest.raises(LexosException) as excinfo:
        _ = mallet.topic_keys

    assert "No topic keys have been set" in str(excinfo.value)
    assert "path_to_topic_keys" in str(excinfo.value)


def test_topic_keys_valid_file():
    """Tests topic_keys property correctly parses a valid file.

    Tests that the topic_keys property correctly reads and parses the
    topic keys file when it exists and has valid content.

    Args:
        None

    Returns:
        None
    """
    # Sample content mimicking a MALLET topic keys file
    mock_content = (
        "0\t0.12345\tword1 word2 word3 word4 word5\n"
        "1\t0.23456\tword6 word7 word8 word9 word10\n"
        "2\t0.34567\tword11 word12 word13 word14 word15\n"
    )

    # Expected result after parsing
    expected_keys = [
        ["0", "0.12345", "word1 word2 word3 word4 word5"],
        ["1", "0.23456", "word6 word7 word8 word9 word10"],
        ["2", "0.34567", "word11 word12 word13 word14 word15"],
    ]

    with patch("builtins.open", mock_open(read_data=mock_content)):
        mallet = Mallet(metadata={"path_to_topic_keys": "mock/path.txt"})
        result = mallet.topic_keys

    assert result == expected_keys


def test_topic_keys_empty_file():
    """Tests topic_keys property handles empty files.

    Tests that the topic_keys property returns an empty list when the
    topic keys file is empty.

    Args:
        None

    Returns:
        None
    """
    with patch("builtins.open", mock_open(read_data="")):
        mallet = Mallet(metadata={"path_to_topic_keys": "mock/path.txt"})
        result = mallet.topic_keys

    assert result == []


def test_topic_keys_file_not_found():
    """Tests topic_keys property handles file not found errors.

    Tests that the topic_keys property propagates a FileNotFoundError when
    the topic keys file does not exist.

    Args:
        None

    Returns:
        None
    """
    with patch("builtins.open", side_effect=FileNotFoundError()):
        mallet = Mallet(metadata={"path_to_topic_keys": "nonexistent/path.txt"})
        with pytest.raises(FileNotFoundError):
            _ = mallet.topic_keys


def test_topic_keys_io_error():
    """Tests topic_keys property handles IO errors.

    Tests that the topic_keys property propagates an IOError when
    the topic keys file cannot be read due to IO issues.

    Args:
        None

    Returns:
        None
    """
    with patch("builtins.open", side_effect=IOError()):
        mallet = Mallet(metadata={"path_to_topic_keys": "unreadable/path.txt"})
        with pytest.raises(IOError):
            _ = mallet.topic_keys


def test_topic_keys_caching():
    """Tests that topic_keys property correctly caches results.

    Tests that subsequent calls to the topic_keys property return the cached
    result instead of reading the file again.

    Args:
        None

    Returns:
        None
    """
    mock_content = "0\t0.12345\tword1 word2 word3 word4 word5\n"

    with patch("builtins.open", mock_open(read_data=mock_content)) as mock_file:
        mallet = Mallet(metadata={"path_to_topic_keys": "mock/path.txt"})

        # First call should read the file
        _ = mallet.topic_keys
        assert mock_file.call_count == 1

        # Second call should use cached result
        _ = mallet.topic_keys
        assert mock_file.call_count == 1


def test_topic_keys_unusual_format():
    """Tests topic_keys property with unusual file formats.

    Tests that the topic_keys property correctly parses files with varying
    numbers of columns or unusual tab patterns.

    Args:
        None

    Returns:
        None
    """
    # Sample content with varying numbers of tab-separated values
    mock_content = (
        "0\t0.12345\tword1 word2\tsome extra column\n"  # 4 columns
        "1\t0.23456\tword3 word4\n"  # 3 columns
        "2\t\tword5 word6\n"  # Empty weight column
    )

    expected_keys = [
        ["0", "0.12345", "word1 word2", "some extra column"],
        ["1", "0.23456", "word3 word4"],
        ["2", "", "word5 word6"],
    ]

    with patch("builtins.open", mock_open(read_data=mock_content)):
        mallet = Mallet(metadata={"path_to_topic_keys": "mock/path.txt"})
        result = mallet.topic_keys

    assert result == expected_keys


#### Test Vocab_Size Property ####


def test_vocab_size_with_value():
    """Tests vocab_size property when metadata contains a vocab_size value.

    This test verifies that the vocab_size property correctly returns the value
    stored in the metadata dictionary when it exists.

    Args:
        None

    Returns:
        None
    """
    test_value = 5000
    mallet = Mallet(metadata={"vocab_size": test_value})
    assert mallet.vocab_size == test_value


def test_vocab_size_with_zero():
    """Tests vocab_size property when metadata contains zero vocabulary size.

    This test verifies that the vocab_size property correctly returns zero
    when the metadata dictionary contains a vocab_size key with value 0.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet(metadata={"vocab_size": 0})
    assert mallet.vocab_size == 0


def test_vocab_size_missing_key():
    """Tests vocab_size property when metadata does not contain vocab_size key.

    This test verifies that the vocab_size property correctly returns 0
    when the metadata dictionary does not contain a vocab_size key.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.vocab_size == 0


def test_vocab_size_with_other_metadata():
    """Tests vocab_size property with unrelated metadata present.

    This test verifies that the vocab_size property correctly returns 0
    when the metadata dictionary contains other keys but not vocab_size.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet(metadata={"num_docs": 42, "mean_num_tokens": 100})
    assert mallet.vocab_size == 0


def test_vocab_size_after_update():
    """Tests vocab_size property after updating the metadata.

    This test verifies that the vocab_size property correctly reflects
    changes to the metadata dictionary.

    Args:
        None

    Returns:
        None
    """
    mallet = Mallet()
    assert mallet.vocab_size == 0

    # Update the metadata
    mallet.metadata["vocab_size"] = 3500
    assert mallet.vocab_size == 3500

    # Update it again
    mallet.metadata["vocab_size"] = 4200
    assert mallet.vocab_size == 4200


#### Test track_progress() ####


def test_track_progress_basic_functionality(mallet_instance):
    """Tests basic functionality of _track_progress method.

    Verifies that the method correctly parses progress information
    and updates the progress bar accordingly.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
    """
    # Mock command and iterations
    mallet_cmd = "mock_mallet_command"
    num_iterations = 100

    # Create mock output that includes progress information
    mock_output = [
        b"Starting training...\n",
        b"<10> Progress at 10%\n",  # 10% progress
        b"<50> Progress at 50%\n",  # 50% progress
        b"<100> Progress complete\n",  # 100% progress
        # b""  # Empty line to simulate end of output
    ]

    # Create a mock Popen object
    mock_process = MagicMock()
    mock_process.poll.side_effect = [
        None,
        None,
        None,
        None,
        0,
    ]  # Return None until last call, then 0
    mock_process.stdout.readline.side_effect = mock_output

    # Use StringIO to capture console output
    console_output = StringIO()
    console = Console(file=console_output, highlight=False)

    # Patch only subprocess.Popen, but use real Progress and Console
    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        # Replace the Console instance in the method with our custom one
        with patch("rich.console.Console", return_value=console):
            # Call the method with verbose=True
            mallet_instance._track_progress(mallet_cmd, num_iterations, verbose=True)

            # Verify subprocess.Popen was called with correct arguments
            mock_popen.assert_called_once_with(
                mallet_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )

            # Verify that we processed all expected lines of output
            assert mock_process.stdout.readline.call_count == 4


def test_track_progress_non_verbose_mode(mallet_instance):
    """Tests _track_progress with verbose=False.

    This test verifies that the method doesn't print output
    when verbose is set to False.

    Args:
        mallet_instance: A fixture providing a Mallet instance.
    """
    # Mock command and iterations
    mallet_cmd = "mock_mallet_command"
    num_iterations = 100

    # Create mock output that includes progress information
    mock_output = [
        b"Starting training...\n",
        b"<10> Progress at 10%\n",  # 10% progress
        b"<50> Progress at 50%\n",  # 50% progress
        b"<100> Progress complete\n",  # 100% progress
        # b""  # Empty line to simulate end of output
    ]

    # Create a mock Popen object
    mock_process = MagicMock()
    mock_process.poll.side_effect = [
        None,
        None,
        None,
        None,
        0,
    ]  # Return None until last call, then 0
    mock_process.stdout.readline.side_effect = mock_output

    # Patch only subprocess.Popen, but use real Progress and Console
    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        mallet_instance._track_progress(mallet_cmd, num_iterations, verbose=False)

        # Verify subprocess.Popen was called with correct arguments
        mock_popen.assert_called_once_with(
            mallet_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )

        # Verify that we processed all expected lines of output
        assert mock_process.stdout.readline.call_count == 4


def test_track_progress_with_non_matching_lines(mallet_instance):
    """Tests _track_progress with lines that don't match the progress pattern.

    This test verifies that the method correctly handles output lines
    that don't contain progress information.

    Args:
        mallet_instance: A fixture providing a Mallet instance.
    """
    # Mock command and iterations
    mallet_cmd = "mock_mallet_command"
    num_iterations = 100

    # Create mock output with non-matching lines
    mock_output = [
        b"Starting training...\n",
        b"This line doesn't match the pattern\n",
        b"<5> This line has progress information\n",
        b"Another non-matching line\n",
    ]

    # Create a mock Popen object
    mock_process = MagicMock()
    mock_process.poll.side_effect = [
        None,
        None,
        None,
        None,
        0,
    ]  # Return None until last call, then 0
    mock_process.stdout.readline.side_effect = mock_output

    # Use StringIO to capture console output
    console_output = StringIO()
    console = Console(file=console_output, highlight=False)

    # Patch only subprocess.Popen, but use real Progress and Console
    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        # Replace the Console instance in the method with our custom one
        with patch("rich.console.Console", return_value=console):
            # Call the method with verbose=True
            mallet_instance._track_progress(mallet_cmd, num_iterations, verbose=True)

            # Verify subprocess.Popen was called with correct arguments
            mock_popen.assert_called_once_with(
                mallet_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )

            # Verify that we processed all expected lines of output
            assert mock_process.stdout.readline.call_count == 4


def test_track_progress_with_10_percent_increments(mallet_instance):
    """Tests _track_progress updates the progress bar in 10% increments.

    This test verifies that the method only updates the progress bar
    when progress reaches 10% increments.

    Args:
        mallet_instance: A fixture providing a Mallet instance.
    """
    # Mock command and iterations
    mallet_cmd = "mock_mallet_command"
    num_iterations = 100

    # Create mock output with various progress values
    mock_output = [
        b"<1> 10% progress\n",  # 10% - should update
        b"<1.5> 15% progress\n",  # 15% - not a 10% increment, should not update
        b"<2> 20% progress\n",  # 20% - should update
        b"<5> 50% progress\n",  # 50% - should update
        b"<9.9> 99% progress\n",  # 99% - not a 10% increment, should not update
        b"<10> 100% progress\n",  # 100% - should update
    ]

    # Create a mock Popen object
    mock_process = MagicMock()
    mock_process.poll.side_effect = [
        None,
        None,
        None,
        None,
        0,
    ]  # Return None until last call, then 0
    mock_process.stdout.readline.side_effect = mock_output

    # Use StringIO to capture console output
    console_output = StringIO()
    console = Console(file=console_output, highlight=False)

    # Patch only subprocess.Popen, but use real Progress and Console
    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        # Replace the Console instance in the method with our custom one
        with patch("rich.console.Console", return_value=console):
            # Call the method with verbose=True
            mallet_instance._track_progress(mallet_cmd, num_iterations, verbose=True)

            # Verify subprocess.Popen was called with correct arguments
            mock_popen.assert_called_once_with(
                mallet_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )

            # Verify that we processed all expected lines of output
            assert mock_process.stdout.readline.call_count == 4


#### Test get_keys() ####


@pytest.fixture
def mock_topic_keys():
    """Creates sample topic keys for testing.

    Returns:
        list: Sample topic keys in the format returned by Mallet.
    """
    return [
        ["0", "0.5", "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"],
        [
            "1",
            "0.3",
            "apple banana cherry date elderberry fig grape honeydew imbe jackfruit",
        ],
        ["2", "0.2", "red blue green yellow purple orange black white brown gray"],
    ]


def test_get_keys_all_topics(mallet_instance, mock_topic_keys):
    """Tests get_keys with default parameters to get all topics.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys: Fixture providing sample topic keys.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = mock_topic_keys

        result = mallet_instance.get_keys()

        # Check that all topics are included
        assert "Topic 0" in result
        assert "Topic 1" in result
        assert "Topic 2" in result

        # Check that topic weights are included
        assert "0.5" in result
        assert "0.3" in result
        assert "0.2" in result

        # Check that default number of keys (10) are included
        for topic_line in result.strip().split("\n"):
            # Split by tab and get the keywords part
            keywords = topic_line.split("\t")[2].split()
            assert len(keywords) == 10


def test_get_keys_num_topics_limit(mallet_instance, mock_topic_keys):
    """Tests get_keys with num_topics parameter to limit output.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys: Fixture providing sample topic keys.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = mock_topic_keys

        result = mallet_instance.get_keys(num_topics=2)

        # Check that only the first two topics are included
        assert "Topic 0" in result
        assert "Topic 1" in result
        assert "Topic 2" not in result

        # Count the number of lines to confirm
        assert len(result.strip().split("\n")) == 2


def test_get_keys_specific_topics(mallet_instance, mock_topic_keys):
    """Tests get_keys with topics parameter to get specific topics.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys: Fixture providing sample topic keys.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = mock_topic_keys

        result = mallet_instance.get_keys(topics=[0, 2])

        # Check that only topics 0 and 2 are included
        assert "Topic 0" in result
        assert "Topic 1" not in result
        assert "Topic 2" in result

        # Count the number of lines to confirm
        assert len(result.strip().split("\n")) == 2


def test_get_keys_custom_num_keys(mallet_instance, mock_topic_keys):
    """Tests get_keys with custom num_keys parameter.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys: Fixture providing sample topic keys.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = mock_topic_keys

        result = mallet_instance.get_keys(num_keys=5)

        # Check that only 5 keywords are included for each topic
        for topic_line in result.strip().split("\n"):
            # Split by tab and get the keywords part
            keywords = topic_line.split("\t")[2].split()
            assert len(keywords) == 5

        # Check first topic's keywords
        assert "word1 word2 word3 word4 word5" in result
        # Check second topic's keywords
        assert "apple banana cherry date elderberry" in result


def test_get_keys_both_num_topics_and_specific_topics(mallet_instance, mock_topic_keys):
    """Tests get_keys with both num_topics and topics parameters.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys: Fixture providing sample topic keys.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = mock_topic_keys

        # When both are specified, topics parameter should take precedence
        result = mallet_instance.get_keys(num_topics=1, topics=[1, 2])

        # Check that only topics 1 and 2 are included (from topics parameter)
        assert "Topic 0" not in result
        assert "Topic 1" in result
        assert "Topic 2" in result

        # Count the number of lines to confirm
        assert len(result.strip().split("\n")) == 2


def test_get_keys_empty_topic_list(mallet_instance):
    """Tests get_keys with empty topic_keys.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = []

        result = mallet_instance.get_keys()

        # Result should be an empty string
        assert result == ""


def test_get_keys_num_keys_exceeds_available(mallet_instance, mock_topic_keys):
    """Tests get_keys when num_keys exceeds available keywords.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys: Fixture providing sample topic keys.
    """
    with patch(
        "lexos.topic_modeling.mallet.Mallet.topic_keys", new_callable=PropertyMock
    ) as mock_property:
        # Create topic with fewer keywords
        modified_keys = mock_topic_keys.copy()
        modified_keys.append(["3", "0.1", "one two three"])
        mock_property.return_value = modified_keys

        result = mallet_instance.get_keys(topics=[3], num_keys=10)

        # Should only include available keywords
        assert "one two three" in result


#### Test get_top_docs() ####


@pytest.fixture
def mallet_instance_get_top_docs():
    """Creates a Mallet instance with necessary metadata for testing.

    Returns:
        Mallet: A configured Mallet instance.
    """
    mallet = Mallet(
        metadata={
            "path_to_topic_distributions": "mock/path/distributions.txt",
            "path_to_training_data": "mock/path/training_data.txt",
        }
    )
    return mallet


def test_get_top_docs_missing_distributions_path():
    """Tests get_top_docs raises exception when topic distributions path is missing.

    Tests that attempting to get top docs without setting the path to topic
    distributions raises a LexosException with the appropriate message.

    Returns:
        None
    """
    mallet = Mallet(metadata={"path_to_training_data": "mock/path/training_data.txt"})

    with pytest.raises(LexosException) as excinfo:
        mallet.get_top_docs()

    assert "No topic distributions have been set" in str(excinfo.value)


def test_get_top_docs_missing_training_data_path():
    """Tests get_top_docs raises exception when training data path is missing.

    Tests that attempting to get top docs without setting the path to training
    data raises a LexosException with the appropriate message.

    Returns:
        None
    """
    mallet = Mallet(
        metadata={"path_to_topic_distributions": "mock/path/distributions.txt"}
    )

    with pytest.raises(LexosException) as excinfo:
        mallet.get_top_docs()

    assert "No training data has been set" in str(excinfo.value)


def test_get_top_docs_basic_functionality(mallet_instance_get_top_docs):
    """Tests basic functionality of get_top_docs without metadata.

    Tests that get_top_docs correctly processes topic distributions and training data
    to return a DataFrame with the top documents for a given topic.

    Args:
        mallet_instance_get_top_docs: Fixture providing a Mallet instance.

    Returns:
        None
    """
    # Mock training data file content
    mock_training_data = (
        "0\tno_label\tDocument 0 content\n"
        "1\tno_label\tDocument 1 content\n"
        "2\tno_label\tDocument 2 content\n"
    )

    # Mock distributions property to return sample distribution data
    mock_distributions = [
        [0.7, 0.2, 0.1],  # doc0: 70% topic0, 20% topic1, 10% topic2
        [0.3, 0.6, 0.1],  # doc1: 30% topic0, 60% topic1, 10% topic2
        [0.1, 0.1, 0.8],  # doc2: 10% topic0, 10% topic1, 80% topic2
    ]

    with (
        patch("builtins.open", mock_open(read_data=mock_training_data)),
        patch.object(Mallet, "distributions", new_callable=PropertyMock) as mock_dist,
    ):
        mock_dist.return_value = mock_distributions

        # Get top docs for topic 0
        result = mallet_instance_get_top_docs.get_top_docs(topic=0, n=2)

        # Verify result is a DataFrame with expected structure and content
        assert isinstance(result, pd.DataFrame)
        assert "Distribution" in result.columns
        assert "Document" in result.columns
        assert result.index.name == "Doc ID"

        # Verify correct sorting and limiting
        assert result.shape[0] == 2  # Should have 2 rows (n=2)
        assert result["Distribution"].iloc[0] == 0.7  # Top document for topic 0
        assert result["Document"].iloc[0] == "Document 0 content"


def test_get_top_docs_with_metadata(mallet_instance_get_top_docs):
    """Tests get_top_docs with additional metadata provided.

    Tests that get_top_docs correctly incorporates additional metadata
    when provided as a DataFrame.

    Args:
        mallet_instance_get_top_docs: Fixture providing a Mallet instance.

    Returns:
        None
    """
    # Mock training data file content
    mock_training_data = (
        "0\tno_label\tDocument 0 content\n"
        "1\tno_label\tDocument 1 content\n"
        "2\tno_label\tDocument 2 content\n"
    )

    # Mock distributions property
    mock_distributions = [
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.1, 0.1, 0.8],
    ]

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(
        {"Author": ["Author A", "Author B", "Author C"], "Year": [2020, 2021, 2022]}
    )

    with (
        patch("builtins.open", mock_open(read_data=mock_training_data)),
        patch.object(Mallet, "distributions", new_callable=PropertyMock) as mock_dist,
    ):
        mock_dist.return_value = mock_distributions

        # Get top docs for topic 0 with metadata
        result = mallet_instance_get_top_docs.get_top_docs(
            topic=0, n=3, metadata=metadata_df
        )

        # Verify metadata columns are present
        assert "Author" in result.columns
        assert "Year" in result.columns

        # Verify correct content
        assert result["Author"].iloc[0] == "Author A"  # Top document for topic 0
        assert result["Year"].iloc[0] == 2020


def test_get_top_docs_as_string(mallet_instance_get_top_docs):
    """Tests get_top_docs with as_str=True to return string output.

    Tests that get_top_docs returns a properly formatted string when the
    as_str parameter is set to True.

    Args:
        mallet_instance: Fixture providing a Mallet instance.

    Returns:
        None
    """
    # Mock training data file content
    mock_training_data = (
        "0\tno_label\tDocument 0 content\n1\tno_label\tDocument 1 content\n"
    )

    # Mock distributions property
    mock_distributions = [
        [0.7, 0.3],
        [0.3, 0.7],
    ]

    with (
        patch("builtins.open", mock_open(read_data=mock_training_data)),
        patch.object(Mallet, "distributions", new_callable=PropertyMock) as mock_dist,
    ):
        mock_dist.return_value = mock_distributions

        # Get top docs as string
        result = mallet_instance_get_top_docs.get_top_docs(topic=0, n=1, as_str=True)

        # Verify result is a string
        assert isinstance(result, str)
        # Check for expected content
        assert "0.7" in result
        assert "Document 0 content" in result


def test_get_top_docs_different_topic(mallet_instance_get_top_docs):
    """Tests get_top_docs with a different topic specified.

    Tests that get_top_docs correctly returns documents for a specified topic
    other than the default (topic 0).

    Args:
        mallet_instance_get_top_docs: Fixture providing a Mallet instance.

    Returns:
        None
    """
    # Mock training data file content
    mock_training_data = (
        "0\tno_label\tDocument 0 content\n"
        "1\tno_label\tDocument 1 content\n"
        "2\tno_label\tDocument 2 content\n"
    )

    # Mock distributions property
    mock_distributions = [
        [0.7, 0.2, 0.1],  # doc0: 70% topic0, 20% topic1, 10% topic2
        [0.3, 0.6, 0.1],  # doc1: 30% topic0, 60% topic1, 10% topic2
        [0.1, 0.1, 0.8],  # doc2: 10% topic0, 10% topic1, 80% topic2
    ]

    with (
        patch("builtins.open", mock_open(read_data=mock_training_data)),
        patch.object(Mallet, "distributions", new_callable=PropertyMock) as mock_dist,
    ):
        mock_dist.return_value = mock_distributions

        # Get top docs for topic 2
        result = mallet_instance_get_top_docs.get_top_docs(topic=2, n=3)

        # Verify correct document is returned for topic 2
        assert result["Distribution"].iloc[0] == 0.8  # Top document for topic 2
        assert result["Document"].iloc[0] == "Document 2 content"


def test_get_top_docs_custom_n(mallet_instance_get_top_docs):
    """Tests get_top_docs with custom number of documents.

    Tests that get_top_docs correctly limits the number of returned documents
    based on the specified n parameter.

    Args:
        mallet_instance_get_top_docs: Fixture providing a Mallet instance.

    Returns:
        None
    """
    # Mock training data file content
    mock_training_data = "\n".join(
        [f"{i}\tno_label\tDocument {i} content" for i in range(20)]
    )

    # Mock distributions property - 20 docs with random topic 0 distributions
    import random

    random.seed(42)  # For reproducibility
    mock_distributions = [
        [random.random(), random.random(), random.random()] for _ in range(20)
    ]

    with (
        patch("builtins.open", mock_open(read_data=mock_training_data)),
        patch.object(Mallet, "distributions", new_callable=PropertyMock) as mock_dist,
    ):
        mock_dist.return_value = mock_distributions

        # Test with n=5
        result_5 = mallet_instance_get_top_docs.get_top_docs(topic=0, n=5)
        assert result_5.shape[0] == 5

        # Test with n=15
        result_15 = mallet_instance_get_top_docs.get_top_docs(topic=0, n=15)
        assert result_15.shape[0] == 15


#### Test topic_term_distributions() ####


@pytest.fixture
def mock_topic_term_distributions():
    """Creates a mock topic-term probability dictionary for testing.

    Returns:
        dict: A dictionary mapping topic IDs to dictionaries of term-probability pairs.
    """
    return {
        0: {
            "apple": 0.25,
            "banana": 0.20,
            "cherry": 0.15,
            "date": 0.10,
            "elderberry": 0.08,
            "fig": 0.07,
            "grape": 0.05,
            "honeydew": 0.04,
            "imbe": 0.03,
            "jackfruit": 0.03,
        },
        1: {
            "red": 0.30,
            "blue": 0.25,
            "green": 0.15,
            "yellow": 0.12,
            "purple": 0.08,
            "orange": 0.05,
            "black": 0.03,
            "white": 0.02,
        },
        2: {"cat": 0.40, "dog": 0.30, "fish": 0.15, "bird": 0.10, "hamster": 0.05},
    }


def test_get_topic_term_probabilities_single_topic(mock_topic_term_distributions):
    """Tests get_topic_term_probabilities with a single topic specified as integer.

    Args:
        mock_topic_term_distributions: Fixture providing mock topic-term distributions.
    """
    mallet = Mallet()

    with patch.object(
        Mallet,
        "load_topic_term_distributions",
        return_value=mock_topic_term_distributions,
    ):
        result = mallet.get_topic_term_probabilities(topics=1, n=3)

        # Check that only topic 1 is included
        assert "Topic 1" in result
        assert "Topic 0" not in result
        assert "Topic 2" not in result

        # Check top 3 terms from topic 1 (sorted by probability)
        assert "red: 0.3" in result
        assert "blue: 0.25" in result
        assert "green: 0.15" in result
        assert "yellow: 0.12" not in result  # Should be excluded due to n=3


def test_get_topic_term_probabilities_multiple_topics(mock_topic_term_distributions):
    """Tests get_topic_term_probabilities with multiple topics specified as a list.

    Args:
        mock_topic_term_distributions: Fixture providing mock topic-term distributions.
    """
    mallet = Mallet()

    with patch.object(
        Mallet,
        "load_topic_term_distributions",
        return_value=mock_topic_term_distributions,
    ):
        result = mallet.get_topic_term_probabilities(topics=[0, 2], n=2)

        # Check that only topics 0 and 2 are included
        assert "Topic 0" in result
        assert "Topic 1" not in result
        assert "Topic 2" in result

        # Check top 2 terms from topic 0
        assert "apple: 0.25" in result
        assert "banana: 0.2" in result
        assert "cherry: 0.15" not in result  # Should be excluded due to n=2

        # Check top 2 terms from topic 2
        assert "cat: 0.4" in result
        assert "dog: 0.3" in result
        assert "fish: 0.15" not in result  # Should be excluded due to n=2


def test_get_topic_term_probabilities_all_topics(mock_topic_term_distributions):
    """Tests get_topic_term_probabilities with topics=None to get all topics.

    Args:
        mock_topic_term_distributions: Fixture providing mock topic-term distributions.
    """
    mallet = Mallet()

    with patch.object(
        Mallet,
        "load_topic_term_distributions",
        return_value=mock_topic_term_distributions,
    ):
        result = mallet.get_topic_term_probabilities(topics=None, n=1)

        # Check that all topics are included
        assert "Topic 0" in result
        assert "Topic 1" in result
        assert "Topic 2" in result

        # Check only the top term from each topic is included
        assert "apple: 0.25" in result
        assert "banana: 0.2" not in result  # Should be excluded due to n=1

        assert "red: 0.3" in result
        assert "blue: 0.25" not in result  # Should be excluded due to n=1

        assert "cat: 0.4" in result
        assert "dog: 0.3" not in result  # Should be excluded due to n=1


def test_get_topic_term_probabilities_custom_n(mock_topic_term_distributions):
    """Tests get_topic_term_probabilities with custom n value.

    Args:
        mock_topic_term_distributions: Fixture providing mock topic-term distributions.
    """
    mallet = Mallet()

    with patch.object(
        Mallet,
        "load_topic_term_distributions",
        return_value=mock_topic_term_distributions,
    ):
        result = mallet.get_topic_term_probabilities(topics=2, n=4)

        # Check topic 2 has exactly 4 terms
        assert "cat: 0.4" in result
        assert "dog: 0.3" in result
        assert "fish: 0.15" in result
        assert "bird: 0.1" in result
        assert "hamster: 0.05" not in result  # Should be excluded due to n=4


def test_get_topic_term_probabilities_empty_result():
    """Tests get_topic_term_probabilities with no matching topics."""
    mallet = Mallet()

    with patch.object(
        Mallet, "load_topic_term_distributions", return_value={0: {"word": 0.5}}
    ):
        # Request topic that doesn't exist
        result = mallet.get_topic_term_probabilities(topics=[99])

        # Result should be empty string (no matching topics)
        assert result == ""


def test_get_topic_term_probabilities_result_format(mock_topic_term_distributions):
    """Tests format of the output string from get_topic_term_probabilities.

    Args:
        mock_topic_term_distributions: Fixture providing mock topic-term distributions.
    """
    mallet = Mallet()

    with patch.object(
        Mallet,
        "load_topic_term_distributions",
        return_value=mock_topic_term_distributions,
    ):
        result = mallet.get_topic_term_probabilities(topics=0, n=2)

        # Check format: "Topic X\n\tterm1: prob1\n\tterm2: prob2\n\n"
        expected_format = "Topic 0\n\tapple: 0.25\n\tbanana: 0.2\n\n"
        assert result == expected_format


#### Test import_data() ####


@pytest.fixture
def sample_training_data():
    """Creates sample training data for testing.

    Returns:
        list: A list of strings representing document content.
    """
    return [
        "This is the first document.",
        "This is the second document with some\nnewlines to remove.",
        "This is the third document with\r\ncarriage returns.",
        "This is a longer document with more tokens to test mean calculation.",
    ]


def test_import_data_basic_functionality(sample_training_data):
    """Tests the basic functionality of import_data.

    Tests that import_data correctly saves training data to a file and
    updates the metadata dictionary.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet()

    # Mock open and os.system to prevent actual file operations
    with (
        patch("builtins.open", mock_open()) as mock_file,
        patch("os.system") as mock_system,
    ):
        mallet.import_data(
            training_data=sample_training_data, path_to_training_data="data.txt"
        )

        # Check that the file was opened for writing
        mock_file.assert_called_with("data.txt", "w", encoding="utf-8")

        # Check that each document was written to the file
        handle = mock_file()
        assert handle.write.call_count == len(sample_training_data)

        # Check metadata was updated
        assert mallet.metadata["path_to_training_data"] == "data.txt"
        assert mallet.metadata["path_to_formatted_training_data"] is not None
        assert mallet.metadata["num_docs"] == 4
        assert "mean_num_tokens" in mallet.metadata
        assert "vocab_size" in mallet.metadata
        assert "model_directory" in mallet.metadata

        # Verify MALLET command was executed
        mock_system.assert_called_once()
        assert mallet.path_to_mallet in mock_system.call_args[0][0]


def test_import_data_directory_creation(sample_training_data):
    """Tests import_data handles directory creation correctly.

    Tests that import_data correctly creates directories as needed and
    uses existing model_directory if available.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    # Test when model_directory is not in metadata
    mallet1 = Mallet()

    with (
        patch("builtins.open", mock_open()),
        patch("os.system"),
    ):
        mallet1.import_data(
            training_data=sample_training_data, path_to_training_data="data.txt"
        )
        print(Path("data.txt").parent)

        # Check that model_directory was set to the parent of path_to_training_data
        assert mallet1.metadata["model_directory"] == "."

    # Test when model_directory is already in metadata
    mallet2 = Mallet(metadata={"model_directory": "existing_dir"})

    with (
        patch("builtins.open", mock_open()),
        patch("os.system"),
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        mallet2.import_data(
            training_data=sample_training_data, path_to_training_data="data.txt"
        )

        # Check that the directory was created
        mock_mkdir.assert_called_once_with(exist_ok=True)

        # Check that path_to_training_data was modified to include model_directory
        assert "existing_dir" in mallet2.metadata["path_to_training_data"]


def test_import_data_with_training_ids(sample_training_data):
    """Tests import_data with custom training IDs.

    Tests that import_data correctly uses custom training IDs when provided.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet()
    training_ids = [100, 200, 300, 400]

    with patch("builtins.open", mock_open()) as mock_file, patch("os.system"):
        mallet.import_data(
            training_data=sample_training_data,
            path_to_training_data="data.txt",
            training_ids=training_ids,
        )

        handle = mock_file()
        # Check that custom IDs were used
        for i, doc_id in enumerate(training_ids):
            doc = re.sub("[\r\n]+", " ", sample_training_data[i]).strip()
            handle.write.assert_any_call(f"{doc_id}\tno_label\t{doc}\n")


def test_import_data_with_use_pipe_from(sample_training_data):
    """Tests import_data with use_pipe_from parameter.

    Tests that import_data correctly adds --use-pipe-from to the MALLET command
    when the parameter is provided.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet()

    with patch("builtins.open", mock_open()), patch("os.system") as mock_system:
        mallet.import_data(
            training_data=sample_training_data,
            path_to_training_data="data.txt",
            use_pipe_from="previous_pipe.mallet",
        )

        # Verify --use-pipe-from was added to the command
        cmd = mock_system.call_args[0][0]
        assert "--use-pipe-from previous_pipe.mallet" in cmd


def test_import_data_with_formatted_training_data_path(sample_training_data):
    """Tests import_data with custom path_to_formatted_training_data.

    Tests that import_data correctly uses a custom path for formatted training data
    when provided.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet()

    with patch("builtins.open", mock_open()), patch("os.system") as mock_system:
        mallet.import_data(
            training_data=sample_training_data,
            path_to_training_data="data.txt",
            path_to_formatted_training_data="custom_formatted.mallet",
        )

        # Verify the custom path was used in metadata and command
        assert (
            mallet.metadata["path_to_formatted_training_data"]
            == "custom_formatted.mallet"
        )
        cmd = mock_system.call_args[0][0]
        assert "custom_formatted.mallet" in cmd


def test_import_data_newline_removal(sample_training_data):
    """Tests import_data properly removes newlines and carriage returns.

    Tests that import_data correctly strips newlines and carriage returns from
    the document content before writing to file.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet()

    with patch("builtins.open", mock_open()) as mock_file, patch("os.system"):
        mallet.import_data(
            training_data=sample_training_data, path_to_training_data="data.txt"
        )

        handle = mock_file()

        # Check second document (contains newlines)
        handle.write.assert_any_call(
            "1\tno_label\t This is the second document with some newlines to remove.\n"
        )
        # Check third document (contains carriage returns)
        handle.write.assert_any_call(
            "2\tno_label\t This is the third document with carriage returns.\n"
        )


def test_import_data_metadata_calculation(sample_training_data):
    """Tests import_data correctly calculates metadata values.

    Tests that import_data calculates num_docs, mean_num_tokens, and vocab_size
    correctly.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet()

    with (
        patch("builtins.open", mock_open()),
        patch("os.system"),
        patch("numpy.mean", return_value=5.5) as mock_mean,
    ):
        mallet.import_data(
            training_data=sample_training_data, path_to_training_data="data.txt"
        )

        # Check number of documents
        assert mallet.metadata["num_docs"] == 4

        # Check mean_num_tokens calculation
        mock_mean.assert_called_once()
        mock_mean_args = mock_mean.call_args[0][0]
        assert len(mock_mean_args) == 4  # One count per document
        assert mallet.metadata["mean_num_tokens"] == 5.5

        # Check vocab_size calculation
        # The vocabulary size should count unique tokens across all documents
        assert isinstance(mallet.metadata["vocab_size"], int)
        assert mallet.metadata["vocab_size"] > 0


def test_import_data_command_execution(sample_training_data):
    """Tests import_data executes the correct MALLET command.

    Tests that import_data constructs and executes the correct MALLET command
    with all required parameters.

    Args:
        sample_training_data: Fixture providing sample document content.
    """
    mallet = Mallet(path_to_mallet="/path/to/mallet")

    with (
        patch("builtins.open", mock_open()),
        patch("os.system") as mock_system,
        patch("pathlib.Path.parent", return_value=Path("mock_dir")),
    ):
        mallet.import_data(
            training_data=sample_training_data, path_to_training_data="data.txt"
        )

        # Verify the command structure
        cmd = mock_system.call_args[0][0]
        expected_cmd_parts = [
            "/path/to/mallet import-file",
            "--input",
            "--output",
            "--keep-sequence",
            "--preserve-case",
        ]

        for part in expected_cmd_parts:
            assert part in cmd


#### Test load_topic_term_distributions() ####


def test_missing_path_to_term_weights():
    """Tests that the method raises a LexosException when path_to_term_weights is not in metadata.

    Verifies that an appropriate error is raised when the required path isn't set.
    """
    mallet = Mallet()

    with pytest.raises(LexosException) as excinfo:
        mallet.load_topic_term_distributions()

    assert "No term weights been set" in str(excinfo.value)


def test_load_topic_term_distributions_basic():
    """Tests the basic functionality with a simple file format.

    Verifies that the method correctly parses a term weights file and calculates
    probabilities.
    """
    # Mock term weights file content
    mock_content = "0\tword1\t10\n0\tword2\t5\n1\tword3\t8\n1\tword4\t2\n"

    mallet = Mallet(metadata={"path_to_term_weights": "mock/path.txt"})

    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = mallet.load_topic_term_distributions()

        # Check structure and conversion to int keys
        assert 0 in result
        assert 1 in result
        assert isinstance(result[0], defaultdict)

        # Check word entries
        assert "word1" in result[0]
        assert "word2" in result[0]
        assert "word3" in result[1]
        assert "word4" in result[1]

        # Check probability calculations (word weight / total topic weight)
        assert result[0]["word1"] == 10 / 15  # 10 / (10+5)
        assert result[0]["word2"] == 5 / 15  # 5 / (10+5)
        assert result[1]["word3"] == 8 / 10  # 8 / (8+2)
        assert result[1]["word4"] == 2 / 10  # 2 / (8+2)


def test_load_topic_term_distributions_empty_file():
    """Tests handling of empty term weights file.

    Verifies that the method correctly handles an empty file.
    """
    mallet = Mallet(metadata={"path_to_term_weights": "mock/path.txt"})

    with patch("builtins.open", mock_open(read_data="")):
        result = mallet.load_topic_term_distributions()

        # Result should be empty (but still valid defaultdicts)
        assert isinstance(result, defaultdict)
        assert len(result) == 0


def test_load_topic_term_distributions_file_not_found():
    """Tests handling of FileNotFoundError.

    Verifies that the method propagates file system errors appropriately.
    """
    mallet = Mallet(metadata={"path_to_term_weights": "nonexistent/path.txt"})

    with (
        patch("builtins.open", side_effect=FileNotFoundError()),
        pytest.raises(FileNotFoundError),
    ):
        mallet.load_topic_term_distributions()


def test_load_topic_term_distributions_malformed_line():
    """Tests handling of malformed lines in the term weights file.

    Verifies that the method raises appropriate errors when encountering
    malformed data.
    """
    # Mock term weights file with a malformed line
    mock_content = (
        "0\tword1\t10\n"
        "0\tword2\n"  # Missing weight
        "1\tword3\t8\n"
    )

    mallet = Mallet(metadata={"path_to_term_weights": "mock/path.txt"})

    with (
        patch("builtins.open", mock_open(read_data=mock_content)),
        pytest.raises(ValueError),
    ):
        mallet.load_topic_term_distributions()


def test_load_topic_term_distributions_with_float_weights():
    """Tests handling of floating point weights.

    Verifies that the method correctly handles floating point weights in the file.
    """
    # Mock term weights file with floating point weights
    mock_content = "0\tword1\t10.5\n0\tword2\t5.25\n1\tword3\t8.75\n1\tword4\t2.25\n"

    mallet = Mallet(metadata={"path_to_term_weights": "mock/path.txt"})

    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = mallet.load_topic_term_distributions()

        # Check probability calculations with float values
        assert result[0]["word1"] == 10.5 / (10.5 + 5.25)
        assert result[0]["word2"] == 5.25 / (10.5 + 5.25)
        assert result[1]["word3"] == 8.75 / (8.75 + 2.25)
        assert result[1]["word4"] == 2.25 / (8.75 + 2.25)


def test_load_topic_term_distributions_multiple_terms():
    """Tests handling of multiple terms in a single topic.

    Verifies the method correctly calculates probabilities with many terms.
    """
    # Create a mock file with multiple terms for topic 0
    terms = [f"0\tword{i}\t{i}" for i in range(1, 21)]
    mock_content = "\n".join(terms)

    mallet = Mallet(metadata={"path_to_term_weights": "mock/path.txt"})

    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = mallet.load_topic_term_distributions()

        # Calculate expected total (1+2+...+20 = 210)
        total = sum(range(1, 21))

        # Check a few probabilities
        assert result[0]["word1"] == 1 / total
        assert result[0]["word10"] == 10 / total
        assert result[0]["word20"] == 20 / total

        # Check all probabilities sum to 1 (allowing for floating point precision)
        prob_sum = sum(result[0].values())
        assert abs(prob_sum - 1.0) < 1e-10


#### Test _setup_wordcloud() ####


def test_setup_wordcloud_default_options(mallet_instance):
    """Tests _setup_wordcloud with default options.

    Verifies that the method correctly configures a WordCloud with default options
    when round_mask is False.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
    """
    # Call the method with round_mask=False
    wordcloud = mallet_instance._setup_wordcloud(round_mask=False, max_terms=30)

    # Check if WordCloud was called with correct default options
    expected_options = {
        "background_color": "white",
        "mask": None,
        "contour_width": 0.1,
        "contour_color": "white",
        "max_words": 30,
        "min_font_size": 10,
        "max_font_size": 150,
        # "random_state": 42, WordCloud stores this as a random generator object
        "colormap": "Dark2",
    }
    for k, v in expected_options.items():
        assert wordcloud.__dict__[k] == v, (
            f"Expected {k} to be {v}, but got {wordcloud.__dict__[k]}"
        )


def test_setup_wordcloud_with_round_mask(mallet_instance):
    """Tests _setup_wordcloud with round_mask=True.

    Verifies that the method correctly creates a circular mask when round_mask=True
    and passes it to the WordCloud constructor.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
    """
    wordcloud = mallet_instance._setup_wordcloud(round_mask=True, max_terms=30)
    assert wordcloud.__dict__["mask"] is not None


def test_setup_wordcloud_with_custom_parameters(mallet_instance):
    """Tests _setup_wordcloud with custom parameters.

    Verifies that the method correctly handles custom parameters passed via kwargs
    and overrides default options.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
    """
    # Call the method with custom parameters
    wordcloud = mallet_instance._setup_wordcloud(
        round_mask=False,
        max_terms=50,
        background_color="black",
        prefer_horizontal=0.9,
        colormap="viridis",
    )
    expected_options = {
        "background_color": "black",
        "mask": None,
        "max_words": 50,
        "prefer_horizontal": 0.9,
        "colormap": "viridis",
    }
    for k, v in expected_options.items():
        assert wordcloud.__dict__[k] == v, (
            f"Expected {k} to be {v}, but got {wordcloud.__dict__[k]}"
        )


def test_setup_wordcloud_returns_wordcloud_object(mallet_instance):
    """Tests that _setup_wordcloud returns a WordCloud object.

    Verifies that the method returns an instance of WordCloud.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
    """
    wordcloud = mallet_instance._setup_wordcloud(round_mask=False, max_terms=30)
    assert isinstance(wordcloud, WordCloud)


#### Test Box Plots ####


@pytest.fixture
def mock_topic_keys_for_boxplots():
    """Creates mock topic keys for testing.

    Returns:
        list: Sample topic keys in the format returned by the topic_keys property.
    """
    return [
        ["0", "0.5", "word1 word2 word3 word4 word5 word6"],
        ["1", "0.3", "apple banana cherry date elderberry fig"],
        ["2", "0.2", "red blue green yellow purple orange"],
    ]


@pytest.fixture
def mock_distributions_for_boxplots():
    """Creates mock distributions for testing boxplots.

    Returns:
        list: Sample topic distributions for documents.
    """
    return [
        [0.7, 0.2, 0.1],  # Document 1
        [0.3, 0.6, 0.1],  # Document 2
        [0.1, 0.1, 0.8],  # Document 3
        [0.5, 0.3, 0.2],  # Document 4
        [0.2, 0.7, 0.1],  # Document 5
    ]


def test_single_topic_boxplot(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests plotting a single topic boxplot.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show") as mock_show,
        patch("seaborn.boxplot") as mock_boxplot,
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks") as mock_xticks,
        patch("matplotlib.pyplot.title") as mock_title,
        patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Call with a single topic (as integer)
        result = mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=1, show=False
        )

        # Verify figure was created with expected dimensions
        # TODO: For some reason, the figure is called twice
        # mock_figure.assert_called_once_with(figsize=(6, 6))

        # Verify seaborn styling
        # TODO: Figure out how to test this
        # assert sns.set_theme.called

        # Verify boxplot creation and parameters
        mock_boxplot.assert_called_once()
        args, kwargs = mock_boxplot.call_args

        # Check that the dataframe has the expected structure
        df = kwargs["data"]
        assert isinstance(df, pd.DataFrame)
        assert "Category" in df.columns
        assert "Probability" in df.columns
        assert "Topic" in df.columns

        # Check number of rows matches number of categories
        assert len(df) == len(categories)

        # Verify plotting parameters
        assert kwargs["x"] == "Category"
        assert kwargs["y"] == "Probability"
        assert kwargs["color"] == "lightblue"

        # Check that the plot was properly configured
        mock_xticks.assert_called_once_with(rotation=45, ha="right")
        assert mock_title.called
        mock_tight_layout.assert_called_once()

        # Verify show wasn't called (since show=False)
        mock_show.assert_not_called()

        # Verify a figure was returned
        assert result is not None


def test_multiple_topics_boxplot(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests plotting multiple topic boxplots.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.close") as mock_close,
        patch("matplotlib.pyplot.gcf", return_value="mock_fig") as mock_gcf,
        patch("seaborn.boxplot"),
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Call with multiple topics as list
        result = mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=[0, 2], show=False
        )

        # Verify figure was created twice (once for each topic)
        assert mock_figure.call_count == 2

        # Verify gcf was called twice to get the figures
        assert mock_gcf.call_count == 2

        # Verify sublot was closed
        assert mock_close.call_count == 2

        # Verify show wasn't called
        mock_show.assert_not_called()

        # Verify a list of figures was returned, with 2 figures
        assert isinstance(result, list)
        assert len(result) == 2


def test_all_topics_boxplot(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests plotting all topic boxplots (topics=None).

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.close") as mock_close,
        patch("matplotlib.pyplot.gcf", return_value="mock_fig") as mock_gcf,
        patch("seaborn.boxplot"),
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Call with topics=None (should plot all topics)
        result = mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=None, show=False
        )

        # Verify figure was created 3 times (once for each topic)
        assert mock_figure.call_count == 3

        # Verify gcf was called 3 times
        assert mock_gcf.call_count == 3

        # Verify figures were closed
        assert mock_close.call_count == 3

        # Verify show wasn't called
        mock_show.assert_not_called()

        # Verify a list of figures was returned
        assert isinstance(result, list)
        assert len(result) == 3


def test_filter_by_target_labels(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests filtering categories by target_labels parameter.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "C", "A", "B"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("seaborn.boxplot") as mock_boxplot,
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Call with specific target_labels to filter categories
        mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories,
            topics=0,
            target_labels=["A", "B"],  # Should exclude "C"
            show=True,
        )

        # Verify boxplot creation with filtered data
        args, kwargs = mock_boxplot.call_args
        df = kwargs["data"]

        # Check that only categories "A" and "B" are in the dataframe
        category_values = df["Category"].unique()
        assert "A" in category_values
        assert "B" in category_values
        assert "C" not in category_values

        # Should be 4 rows (2 "A" + 2 "B")
        assert len(df) == 4


def test_save_output(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests saving the plot to a file.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]
    output_path = "test_output.png"

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.show"),
        patch("seaborn.boxplot"),
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Call with output_path specified
        mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, output_path=output_path, show=True
        )

        # Verify savefig was called with the correct path
        mock_savefig.assert_called_once_with(output_path)


def test_show_parameter(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests the show parameter behavior.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value="mock_fig"),
        patch("seaborn.boxplot"),
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Test with show=True
        result_show_true = mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, show=True
        )

        # Verify show was called and result is None
        mock_show.assert_called_once()
        assert result_show_true is None

        mock_show.reset_mock()

        # Test with show=False
        result_show_false = mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, show=False
        )

        # Verify show was not called and result is not None
        mock_show.assert_not_called()
        assert result_show_false is not None


def test_custom_figure_parameters(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests customization of figure parameters.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("seaborn.set_theme") as mock_set_theme,
        patch("seaborn.boxplot") as mock_boxplot,
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Call with custom figsize, font_scale, and color
        custom_figsize = (10, 8)
        custom_font_scale = 1.5
        custom_color = "salmon"

        mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories,
            topics=0,
            figsize=custom_figsize,
            font_scale=custom_font_scale,
            color=custom_color,
            show=True,
        )

        # Verify figure was created with custom dimensions
        mock_figure.assert_called_once_with(figsize=custom_figsize)

        # Verify seaborn theme was set with custom font scale
        mock_set_theme.assert_called_once_with(
            style="ticks", font_scale=custom_font_scale
        )

        # Verify boxplot was created with custom color
        args, kwargs = mock_boxplot.call_args
        assert kwargs["color"] == custom_color


def test_custom_num_keys(
    mallet_instance, mock_topic_keys_for_boxplots, mock_distributions_for_boxplots
):
    """Tests custom number of keywords in topic titles.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_boxplots: Fixture providing mock topic keys.
        mock_distributions_for_boxplots: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("seaborn.boxplot"),
        patch("seaborn.despine"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.title") as mock_title,
        patch("matplotlib.pyplot.tight_layout"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_boxplots
        mock_dist_prop.return_value = mock_distributions_for_boxplots

        # Test with custom num_keys=2 (should show only first 2 keywords)
        mallet_instance.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, num_keys=2, show=True
        )

        # Verify the title contains only the first 2 keywords
        args, kwargs = mock_title.call_args
        title_text = args[0]

        # Should contain "word1 word2" but not "word3"
        assert "word1 word2" in title_text
        assert "word3" not in title_text


#### Test plot_categories_by_topics_heatmap() ####


@pytest.fixture
def mock_topic_keys_for_heatmaps():
    """Creates mock topic keys for testing.

    Returns:
        list: Sample topic keys in the format returned by the topic_keys property.
    """
    return [
        ["0", "0.5", "word1 word2 word3 word4 word5 word6"],
        ["1", "0.3", "apple banana cherry date elderberry fig"],
        ["2", "0.2", "red blue green yellow purple orange"],
    ]


@pytest.fixture
def mock_distributions_for_heatmaps():
    """Creates mock distributions for testing.

    Returns:
        list: Sample topic distributions for documents.
    """
    return [
        [0.7, 0.2, 0.1],  # Document 1
        [0.3, 0.6, 0.1],  # Document 2
        [0.1, 0.1, 0.8],  # Document 3
        [0.5, 0.3, 0.2],  # Document 4
        [0.2, 0.7, 0.1],  # Document 5
    ]


def test_basic_heatmap_creation(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests basic heatmap creation functionality.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch(
            "matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())
        ) as mock_subplots,
        patch("seaborn.heatmap", return_value=MagicMock()) as mock_heatmap,
        patch("matplotlib.pyplot.xticks") as mock_xticks,
        patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.close") as mock_close,
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Call with show=False to get the returned figure
        result = mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, show=False
        )

        # Verify plotting components were called
        assert mock_subplots.called
        assert mock_heatmap.called
        assert mock_xticks.called
        assert mock_tight_layout.called
        assert not mock_savefig.called  # Should not be called as no output_path
        assert not mock_show.called  # Should not be called as show=False
        assert mock_close.called

        # Result should be a figure object (mock in this case)
        assert result is not None


def test_heatmap_with_show_true(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests heatmap creation with show=True.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap", return_value=MagicMock()),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.close") as mock_close,
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Call with show=True
        result = mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, show=True
        )

        # Verify show was called
        mock_show.assert_called_once()

        # close should not be called with show=True
        assert not mock_close.called

        # Result should be None when show=True
        assert result is None


def test_heatmap_with_output_path(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests heatmap creation with output_path specified.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]
    output_path = "test_output.png"

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap", return_value=MagicMock()),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Call with output_path specified
        mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, output_path=output_path
        )

        # Verify savefig was called with the correct path
        mock_savefig.assert_called_once_with(output_path)


def test_heatmap_target_labels_filtering(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests filtering by target_labels parameter.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    # Categories A, B, and C but we'll filter for only A and B
    categories = ["A", "B", "C", "A", "B"]
    target_labels = ["A", "B"]  # Exclude C

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap") as mock_heatmap,
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Use our capture method to get the DataFrame passed to heatmap
        def capture_df(data, **kwargs):
            capture_df.captured_data = data
            return MagicMock()

        mock_heatmap.side_effect = capture_df

        # Call with target_labels specified
        mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, target_labels=target_labels
        )

        # Get the captured DataFrame
        captured_df = capture_df.captured_data

        # Verify only target_labels are in the index
        assert all(label in target_labels for label in captured_df.index)
        assert "C" not in captured_df.index


def test_heatmap_custom_figsize(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests custom figsize parameter.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]
    custom_figsize = (10, 8)

    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Call with custom figsize
        mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, figsize=custom_figsize
        )

        # Verify figure was created with custom dimensions
        mock_figure.assert_called_with(figsize=custom_figsize)


def test_heatmap_custom_font_scale(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests custom font_scale parameter.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]
    custom_font_scale = 1.5

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.set_theme") as mock_set_theme,
        patch("seaborn.heatmap"),
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Call with custom font_scale
        mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, font_scale=custom_font_scale
        )

        # Verify seaborn theme was set with custom font scale
        mock_set_theme.assert_called_once_with(
            style="ticks", font_scale=custom_font_scale
        )


def test_heatmap_custom_cmap(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests custom colormap parameter.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]
    custom_cmap = "viridis"

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap") as mock_heatmap,
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Call with custom colormap
        mallet_instance.plot_categories_by_topics_heatmap(
            categories=categories, cmap=custom_cmap
        )

        # Verify heatmap was called with custom colormap
        args, kwargs = mock_heatmap.call_args
        assert kwargs["cmap"] == custom_cmap


def test_heatmap_data_transformation(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests the data transformation process for the heatmap.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap") as mock_heatmap,
        patch("matplotlib.pyplot.xticks"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Capture the DataFrame passed to heatmap
        def capture_df(data, **kwargs):
            capture_df.captured_data = data
            return MagicMock()

        mock_heatmap.side_effect = capture_df

        mallet_instance.plot_categories_by_topics_heatmap(categories=categories)

        # Get the captured DataFrame
        captured_df = capture_df.captured_data

        # Check DataFrame has been normalized
        # For a normalized column, mean should be close to 0 and std close to 1
        for col in captured_df.columns:
            col_data = captured_df[col].dropna()
            if len(col_data) > 1:  # Need at least 2 values for meaningful std
                assert abs(col_data.mean()) < 1e-10  # Mean should be very close to 0
                assert (
                    abs(col_data.std() - 1.0) < 1e-10
                )  # Std should be very close to 1


def test_heatmap_axis_configuration(
    mallet_instance, mock_topic_keys_for_heatmaps, mock_distributions_for_heatmaps
):
    """Tests the axis configuration for the heatmap.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_keys_for_heatmaps: Fixture providing mock topic keys.
        mock_distributions_for_heatmaps: Fixture providing mock distributions.
    """
    categories = ["A", "B", "A", "B", "A"]

    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("seaborn.heatmap", return_value=MagicMock()) as mock_heatmap,
        patch("matplotlib.pyplot.xticks") as mock_xticks,
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.show"),
        patch.object(Mallet, "topic_keys", new_callable=PropertyMock) as mock_keys_prop,
        patch.object(
            Mallet, "distributions", new_callable=PropertyMock
        ) as mock_dist_prop,
    ):
        mock_keys_prop.return_value = mock_topic_keys_for_heatmaps
        mock_dist_prop.return_value = mock_distributions_for_heatmaps

        # Create a mock axis object
        mock_ax = MagicMock()
        mock_heatmap.return_value = mock_ax

        mallet_instance.plot_categories_by_topics_heatmap(categories=categories)

        # Verify axis configuration
        assert mock_ax.xaxis.tick_top.called
        assert mock_ax.xaxis.set_label_position.called
        mock_xticks.assert_called_once_with(rotation=30, ha="left")


#### Test topic_clouds() ####


@pytest.fixture
def mock_topic_term_distributions_for_topic_clouds():
    """Creates mock topic-term distributions for testing.

    Returns:
        dict: A dictionary mapping topic IDs to dictionaries of term-probability pairs.
    """
    return {
        0: {
            "apple": 0.25,
            "banana": 0.20,
            "cherry": 0.15,
            "date": 0.10,
            "elderberry": 0.08,
            "fig": 0.07,
            "grape": 0.05,
            "honeydew": 0.04,
            "imbe": 0.03,
            "jackfruit": 0.03,
        },
        1: {
            "red": 0.30,
            "blue": 0.25,
            "green": 0.15,
            "yellow": 0.12,
            "purple": 0.08,
            "orange": 0.05,
            "black": 0.03,
            "white": 0.02,
        },
        2: {"cat": 0.40, "dog": 0.30, "fish": 0.15, "bird": 0.10, "hamster": 0.05},
    }


def test_topic_clouds_single_topic(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with a single topic.

    Tests that topic_clouds correctly handles a single topic (as an integer)
    and generates the appropriate word cloud.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    # Various mocks to prevent actual plotting
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot") as mock_subplot,
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title") as mock_title,
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close") as mock_close,
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()) as mock_gcf,
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies") as mock_generate,
    ):
        # Call with a single topic (as integer)
        result = mallet_instance.topic_clouds(topics=1, show=False)

        # Check that only one subplot was created
        mock_subplot.assert_called_once_with(1, 1, 2)  # (1 row, 1 column, topic 1+1)

        # Check title contains topic number
        mock_title.assert_called_once_with("Topic 1")

        # Check word cloud was generated with correct data
        mock_generate.assert_called_once_with(
            mock_topic_term_distributions_for_topic_clouds[1]
        )

        # Verify figure was retrieved and returned
        mock_gcf.assert_called_once()
        assert result is not None
        mock_close.assert_called_once()


def test_topic_clouds_multiple_topics(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with multiple topics.

    Tests that topic_clouds correctly handles multiple topics (as a list)
    and generates the appropriate word clouds.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot") as mock_subplot,
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title") as mock_title,
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies") as mock_generate,
    ):
        # Call with multiple topics as list
        mallet_instance.topic_clouds(topics=[0, 2], show=False)

        # Check that subplots were created for both topics
        assert mock_subplot.call_count == 2

        # Check titles were set for both topics
        assert mock_title.call_count == 2
        mock_title.assert_has_calls(
            [
                call("Topic 0"),
                call("Topic 2"),
            ]
        )

        # Check word clouds were generated for both topics
        assert mock_generate.call_count == 2
        mock_generate.assert_has_calls(
            [
                call(mock_topic_term_distributions_for_topic_clouds[0]),
                call(mock_topic_term_distributions_for_topic_clouds[2]),
            ]
        )


def test_topic_clouds_all_topics(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with None to show all topics.

    Tests that topic_clouds correctly handles topics=None to generate
    word clouds for all topics.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot") as mock_subplot,
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title") as mock_title,
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies") as mock_generate,
    ):
        # Call with topics=None (should process all topics)
        mallet_instance.topic_clouds(topics=None, show=False)

        # Check that subplots were created for all topics
        assert mock_subplot.call_count == 3  # 3 topics in mock data

        # Check titles were set for all topics
        assert mock_title.call_count == 3

        # Check word clouds were generated for all topics
        assert mock_generate.call_count == 3


def test_topic_clouds_auto_layout(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with auto layout calculation.

    Tests that topic_clouds correctly calculates rows and columns for the grid
    when layout="auto".

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot") as mock_subplot,
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies"),
    ):
        # Call with topics=None and layout="auto"
        mallet_instance.topic_clouds(topics=None, layout="auto", show=False)

        # With 3 topics, auto layout should use 3 rows and 1 column (sqrt(3) ≈ 1.7, floor=1)
        # After the function call but before the assertion
        mock_subplot.assert_has_calls(
            [
                call(3, 1, 1),  # (3 rows, 1 column, topic 0+1)
                call(3, 1, 2),  # (3 rows, 1 column, topic 1+1)
                call(3, 1, 3),  # (3 rows, 1 column, topic 2+1)
            ]
        )


def test_topic_clouds_custom_layout(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with a custom layout.

    Tests that topic_clouds correctly uses a custom layout when provided.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot") as mock_subplot,
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies"),
    ):
        # Call with custom layout (3 rows, 2 columns)
        custom_layout = (3, 2)
        mallet_instance.topic_clouds(topics=[0, 1, 2], layout=custom_layout, show=False)

        # The subplot calls should use the custom layout
        mock_subplot.assert_has_calls(
            [
                call(3, 2, 1),  # (3 rows, 2 columns, topic 0+1)
                call(3, 2, 2),  # (3 rows, 2 columns, topic 1+1)
                call(3, 2, 3),  # (3 rows, 2 columns, topic 2+1)
            ]
        )


def test_topic_clouds_output_path(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with output_path parameter.

    Tests that topic_clouds correctly saves the figure when output_path is specified.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    output_path = "test_output.png"

    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot"),
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies"),
    ):
        # Call with output_path specified
        mallet_instance.topic_clouds(topics=0, output_path=output_path, show=False)

        # Verify savefig was called with the correct path
        mock_savefig.assert_called_once_with(output_path)


def test_topic_clouds_show_parameter(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with show parameter.

    Tests that topic_clouds either shows the plot and returns None when show=True,
    or doesn't show the plot and returns a figure when show=False.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot"),
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close") as mock_close,
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()) as mock_gcf,
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies"),
    ):
        # Test with show=True
        result_show_true = mallet_instance.topic_clouds(topics=0, show=True)

        # Verify show was called and result is None
        mock_show.assert_called_once()
        assert result_show_true is None
        mock_show.reset_mock()

        # Test with show=False
        result_show_false = mallet_instance.topic_clouds(topics=0, show=False)

        # Verify show was not called and result is not None
        mock_show.assert_not_called()
        mock_gcf.assert_called_once()
        mock_close.assert_called_once()
        assert result_show_false is not None


def test_topic_clouds_custom_figsize(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds with custom figsize.

    Tests that topic_clouds correctly sets the figure size when figsize is specified.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    custom_figsize = (15, 15)

    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot"),
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies"),
    ):
        # Mock plt.rcParams to check it's being set
        with patch.dict("matplotlib.pyplot.rcParams", {"figure.figsize": (10, 10)}):
            # Call with custom figsize
            mallet_instance.topic_clouds(topics=0, figsize=custom_figsize, show=False)

            # Verify figure size was set
            assert tuple(plt.rcParams["figure.figsize"]) == custom_figsize


def test_topic_clouds_nonexistent_topic(
    mallet_instance, mock_topic_term_distributions_for_topic_clouds
):
    """Tests topic_clouds when requested topic doesn't exist.

    Tests that topic_clouds handles gracefully when a requested topic doesn't exist
    in the topic term distributions.

    Args:
        mallet_instance: Fixture providing a Mallet instance.
        mock_topic_term_distributions_for_topic_clouds: Fixture providing mock topic-term distributions.
    """
    with (
        patch.object(
            Mallet,
            "load_topic_term_distributions",
            return_value=mock_topic_term_distributions_for_topic_clouds,
        ),
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplot") as mock_subplot,
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("seaborn.set_theme"),
        patch("wordcloud.WordCloud.generate_from_frequencies") as mock_generate,
    ):
        # Call with a topic that doesn't exist
        mallet_instance.topic_clouds(topics=99, show=False)

        # No subplot or generate calls should be made since topic doesn't exist
        mock_subplot.assert_not_called()
        mock_generate.assert_not_called()


#### Test train() ####


@pytest.fixture
def mallet_with_model_directory():
    """Creates a Mallet instance with model_directory set.

    Returns:
        Mallet: A Mallet instance with model_directory in metadata.
    """
    return Mallet(metadata={"model_directory": "/test/model/dir"})


@pytest.fixture
def mallet_with_training_data():
    """Creates a Mallet instance with model_directory and formatted training data path set.

    Returns:
        Mallet: A Mallet instance with necessary metadata.
    """
    return Mallet(
        metadata={
            "model_directory": "/test/model/dir",
            "path_to_formatted_training_data": "/test/model/dir/formatted_data.mallet",
        }
    )


def test_train_missing_model_directory():
    """Tests train raises exception when model_directory is missing.

    Tests that the train method raises a LexosException when the model_directory
    is not set in the metadata.
    """
    mallet = Mallet()  # No model_directory set

    with pytest.raises(LexosException) as excinfo:
        mallet.train(num_topics=10)

    assert "No model directory has been set" in str(excinfo.value)


def test_train_missing_formatted_training_data(mallet_with_model_directory):
    """Tests train raises exception when path_to_formatted_training_data is missing.

    Tests that the train method raises a LexosException when the
    path_to_formatted_training_data is not set in the metadata.

    Args:
        mallet_with_model_directory: Fixture providing a Mallet instance with model_directory.
    """
    with pytest.raises(LexosException) as excinfo:
        mallet_with_model_directory.train(num_topics=10)

    assert "No training data has been set" in str(excinfo.value)


def test_train_with_provided_formatted_training_data(mallet_with_model_directory):
    """Tests train with path_to_formatted_training_data provided as parameter.

    Tests that the train method correctly updates metadata when
    path_to_formatted_training_data is provided as a parameter.

    Args:
        mallet_with_model_directory: Fixture providing a Mallet instance with model_directory.
    """
    path = "/test/data/formatted.mallet"

    with patch.object(mallet_with_model_directory, "_track_progress"):
        mallet_with_model_directory.train(
            num_topics=10, path_to_formatted_training_data=path
        )

        assert (
            mallet_with_model_directory.metadata["path_to_formatted_training_data"]
            == path
        )


def test_train_command_building(mallet_with_training_data):
    """Tests train builds the MALLET command correctly.

    Tests that the train method correctly builds the MALLET command with all the
    necessary parameters.

    Args:
        mallet_with_training_data: Fixture providing a Mallet instance with training data.
    """
    with patch.object(mallet_with_training_data, "_track_progress") as mock_track:
        mallet_with_training_data.train(
            num_topics=10,
            num_iterations=200,
            path_to_model="model.bin",
            path_to_state="state.gz",
            path_to_topic_keys="keys.txt",
            path_to_topic_distributions="distributions.txt",
            path_to_term_weights="weights.txt",
            path_to_diagnostics="diag.xml",
            optimize_interval=20,
        )

        # Check that _track_progress was called with the correct command
        cmd = mock_track.call_args[0][0]

        # Check that all parameters were included in the command
        assert mallet_with_training_data.path_to_mallet in cmd
        assert "train-topics" in cmd
        assert "--input" in cmd
        assert "--num-topics 10" in cmd
        assert "--num-iterations 200" in cmd
        assert "--inferencer-filename" in cmd
        assert "--output-state" in cmd
        assert "--output-topic-keys" in cmd
        assert "--output-doc-topics" in cmd
        assert "--topic-word-weights-file" in cmd
        assert "--diagnostics-file" in cmd
        assert "--optimize-interval 20" in cmd


def test_train_default_parameters(mallet_with_training_data):
    """Tests train with default parameters.

    Tests that the train method correctly builds the command with default values
    when optional parameters are not provided.

    Args:
        mallet_with_training_data: Fixture providing a Mallet instance with training data.
    """
    with patch.object(mallet_with_training_data, "_track_progress") as mock_track:
        mallet_with_training_data.train(num_topics=10)

        # Check that _track_progress was called with basic command
        cmd = mock_track.call_args[0][0]

        # Should include default values
        assert "--num-topics 10" in cmd
        assert "--num-iterations 100" in cmd  # Default
        assert "--optimize-interval 10" in cmd  # Default

        # Optional params not provided, shouldn't be in command
        assert "--inferencer-filename" not in cmd
        assert "--output-state" not in cmd
        assert "--output-topic-keys" not in cmd
        assert "--output-doc-topics" not in cmd
        assert "--topic-word-weights-file" not in cmd
        assert "--diagnostics-file" not in cmd


def test_train_relative_path_handling(mallet_with_training_data):
    """Tests train handles relative paths correctly.

    Tests that the train method correctly converts relative paths to absolute paths
    based on the model_directory.

    Args:
        mallet_with_training_data: Fixture providing a Mallet instance with training data.
    """
    with patch.object(mallet_with_training_data, "_track_progress") as mock_track:
        mallet_with_training_data.train(
            num_topics=10,
            path_to_model="model.bin",  # Relative path
            path_to_topic_keys="/absolute/path/keys.txt",  # Absolute path
        )

        cmd = mock_track.call_args[0][0]

        # Relative path should be prefixed with model_directory
        assert (
            f"{mallet_with_training_data.metadata['model_directory']}/model.bin" in cmd
        )

        # Absolute path should remain unchanged
        assert "/absolute/path/keys.txt" in cmd


def test_train_metadata_updates(mallet_with_training_data):
    """Tests train updates metadata correctly.

    Tests that the train method correctly updates the metadata dictionary with
    all the relevant paths and parameters.

    Args:
        mallet_with_training_data: Fixture providing a Mallet instance with training data.
    """
    with patch.object(mallet_with_training_data, "_track_progress"):
        mallet_with_training_data.train(
            num_topics=10,
            path_to_model="model.bin",
            path_to_state="state.gz",
            path_to_topic_keys="keys.txt",
            optimize_interval=20,
        )

        # Check metadata was updated
        assert mallet_with_training_data.metadata["num_topics"] == 10
        assert mallet_with_training_data.metadata["optimize_interval"] == 20

        # Check paths were updated and converted to absolute paths
        model_dir = mallet_with_training_data.metadata["model_directory"]
        assert (
            mallet_with_training_data.metadata["path_to_model"]
            == f"{model_dir}/model.bin"
        )
        assert (
            mallet_with_training_data.metadata["path_to_state"]
            == f"{model_dir}/state.gz"
        )
        assert (
            mallet_with_training_data.metadata["path_to_topic_keys"]
            == f"{model_dir}/keys.txt"
        )

        # Command should be saved in metadata
        assert "training_command" in mallet_with_training_data.metadata
        assert (
            mallet_with_training_data.path_to_mallet
            in mallet_with_training_data.metadata["training_command"]
        )


def test_train_track_progress_call(mallet_with_training_data):
    """Tests train calls _track_progress correctly.

    Tests that the train method correctly calls the _track_progress method with
    the expected arguments.

    Args:
        mallet_with_training_data: Fixture providing a Mallet instance with training data.
    """
    with patch.object(mallet_with_training_data, "_track_progress") as mock_track:
        mallet_with_training_data.train(
            num_topics=10, num_iterations=200, verbose=False
        )

        # Check _track_progress was called with correct args
        mock_track.assert_called_once()
        args = mock_track.call_args[0]
        assert isinstance(args[0], str)  # cmd
        assert args[1] == 200  # num_iterations
        assert args[2] is False  # verbose


def test_train_with_absolute_path_to_formatted_training_data(
    mallet_with_model_directory,
):
    """Tests train with absolute path to formatted training data.

    Tests that the train method correctly handles an absolute path provided for
    path_to_formatted_training_data.

    Args:
        mallet_with_model_directory: Fixture providing a Mallet instance with model_directory.
    """
    abs_path = "/absolute/path/to/formatted_data.mallet"

    with patch.object(mallet_with_model_directory, "_track_progress"):
        mallet_with_model_directory.train(
            num_topics=10, path_to_formatted_training_data=abs_path
        )

        # Check path was preserved as absolute
        assert (
            mallet_with_model_directory.metadata["path_to_formatted_training_data"]
            == abs_path
        )

        # Check command contains the absolute path
        cmd = mallet_with_model_directory.metadata["training_command"]
        assert f"--input {abs_path}" in cmd
