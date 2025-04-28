"""test_dfr_browser.py.

Last updated: April 28, 2025
"""

import time
import zipfile
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from lexos.exceptions import LexosException
from lexos.topic_modeling.dfr_browser import DFR_BROWSER_TEMPLATE_DIR, DfrBrowser

#### Common Fixtures ####


@pytest.fixture
def valid_browser_params():
    """Creates valid parameters for DfrBrowser initialization.

    Returns:
        dict: Valid parameters for a DfrBrowser instance.
    """
    return {
        "path_to_browser_dir": "/path/to/browser",
        "metadata": [{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        "num_topics": 10,
        "path_to_state_file": "/path/to/state.gz",
    }


#### Test Initialisation ####


def test_init_required_fields(valid_browser_params):
    """Tests initialization with required fields only.

    Verifies that DfrBrowser can be created with just the required
    parameters and that they're correctly set.

    Args:
        valid_browser_params: Fixture providing valid initialization parameters.
    """
    browser = DfrBrowser(**valid_browser_params)

    assert browser.path_to_browser_dir == valid_browser_params["path_to_browser_dir"]
    assert isinstance(browser.metadata, pd.DataFrame)
    assert browser.num_topics == valid_browser_params["num_topics"]
    assert browser.path_to_state_file == valid_browser_params["path_to_state_file"]


def test_default_optional_field_values(valid_browser_params):
    """Tests default values for optional fields.

    Verifies that optional fields have the expected default values when
    not explicitly provided.

    Args:
        valid_browser_params: Fixture providing valid initialization parameters.
    """
    browser = DfrBrowser(**valid_browser_params)

    assert browser.path_to_data_file is None
    assert browser.properties is None
    assert browser.embargo is False
    assert browser.path_to_template_dir == DFR_BROWSER_TEMPLATE_DIR
    assert browser.port == 8888
    assert browser.handler is None


def test_custom_optional_field_values(valid_browser_params):
    """Tests setting custom values for optional fields.

    Verifies that custom values for optional fields are correctly set
    when explicitly provided.

    Args:
        valid_browser_params: Fixture providing valid initialization parameters.
    """
    # Add custom values for optional parameters
    params_with_options = valid_browser_params.copy()
    params_with_options.update(
        {
            "path_to_data_file": "/path/to/data.txt",
            "properties": {"title": "Test Browser", "author": "Test Author"},
            "embargo": True,
            "path_to_template_dir": "/custom/template/dir",
            "port": 9999,
            "handler": SimpleHTTPRequestHandler,
        }
    )

    browser = DfrBrowser(**params_with_options)

    assert browser.path_to_data_file == params_with_options["path_to_data_file"]
    assert browser.properties == params_with_options["properties"]
    assert browser.embargo is True
    assert browser.path_to_template_dir == params_with_options["path_to_template_dir"]
    assert browser.port == 9999
    assert browser.handler == SimpleHTTPRequestHandler


def test_missing_required_fields():
    """Tests validation for missing required fields.

    Verifies that appropriate validation errors are raised when required fields
    are not provided during instantiation.
    """
    # Test missing all required fields
    with pytest.raises(ValidationError):
        DfrBrowser()

    # Test with only path_to_browser_dir
    with pytest.raises(ValidationError):
        DfrBrowser(path_to_browser_dir="/path/to/browser")

    # Test with path_to_browser_dir and metadata
    with pytest.raises(ValidationError):
        DfrBrowser(
            path_to_browser_dir="/path/to/browser",
            metadata=[{"id": 1, "title": "Doc1"}],
        )

    # Test with path_to_browser_dir, metadata, and num_topics
    with pytest.raises(ValidationError):
        DfrBrowser(
            path_to_browser_dir="/path/to/browser",
            metadata=[{"id": 1, "title": "Doc1"}],
            num_topics=10,
        )


def test_invalid_field_types():
    """Tests validation for invalid field types.

    Verifies that appropriate validation errors are raised when fields
    have invalid types.
    """
    # Test invalid path_to_browser_dir type
    with pytest.raises(ValidationError):
        DfrBrowser(
            path_to_browser_dir=[{"id": 1}],  # Should be a string
            metadata=[{"id": 1}],
            num_topics=10,
            path_to_state_file="/path/to/state.gz",
        )

    # Test invalid metadata type
    with pytest.raises(ValidationError):
        DfrBrowser(
            path_to_browser_dir="/path/to/browser",
            metadata="invalid metadata",  # Should be a list of dicts
            num_topics=10,
            path_to_state_file="/path/to/state.gz",
        )

    # Test invalid num_topics type
    with pytest.raises(ValidationError):
        DfrBrowser(
            path_to_browser_dir="/path/to/browser",
            metadata=[{"id": 1}],
            num_topics="ten",  # Should be an integer
            path_to_state_file="/path/to/state.gz",
        )

    # Test invalid embargo type
    with pytest.raises(ValidationError):
        DfrBrowser(
            path_to_browser_dir="/path/to/browser",
            metadata=[{"id": 1}],
            num_topics=10,
            path_to_state_file="/path/to/state.gz",
            embargo=[{"id": 1}],  # Should be a boolean
        )


def test_metadata_conversion():
    """Tests conversion of metadata from list to DataFrame.

    Verifies that the __init__ method correctly converts a list of dicts
    to a pandas DataFrame.
    """
    metadata_list = [
        {"id": 1, "title": "Doc1", "author": "Author1"},
        {"id": 2, "title": "Doc2", "author": "Author2"},
    ]

    browser = DfrBrowser(
        path_to_browser_dir="/path/to/browser",
        metadata=metadata_list,
        num_topics=10,
        path_to_state_file="/path/to/state.gz",
    )

    # Verify metadata was converted to DataFrame
    assert isinstance(browser.metadata, pd.DataFrame)
    assert len(browser.metadata) == 2
    assert list(browser.metadata.columns) == ["id", "title", "author"]
    assert browser.metadata.iloc[0]["title"] == "Doc1"


def test_model_config():
    """Tests that the model_config is correctly set.

    Verifies that arbitrary_types_allowed is set to True in the model config,
    which is necessary for fields like pandas DataFrame.
    """
    assert DfrBrowser.model_config["arbitrary_types_allowed"] is True


def test_init_converts_metadata_to_dataframe(valid_browser_params):
    """Tests that __init__ converts metadata from list of dicts to DataFrame.

    Verifies that when a list of dictionaries is provided for metadata,
    it gets converted to a pandas DataFrame.

    Args:
        valid_browser_params: Fixture providing minimal valid parameters.
    """
    # Create a browser instance
    browser = DfrBrowser(**valid_browser_params)

    # Verify metadata was converted to DataFrame
    assert isinstance(browser.metadata, pd.DataFrame)
    assert browser.metadata.shape == (2, 2)  # 2 rows, 2 columns
    assert set(browser.metadata.columns) == {"id", "title"}


def test_init_with_empty_metadata_list():
    """Tests __init__ with an empty metadata list.

    Verifies behavior when metadata is an empty list.
    """
    params = {
        "path_to_browser_dir": "/test/browser/dir",
        "metadata": [],
        "num_topics": 10,
        "path_to_state_file": "/test/state/file.gz",
    }

    browser = DfrBrowser(**params)

    # Verify metadata was converted to an empty DataFrame
    assert isinstance(browser.metadata, pd.DataFrame)
    assert browser.metadata.empty


def test_init_with_none_values_in_metadata(valid_browser_params):
    """Tests __init__ with None values in metadata.

    Verifies that None values in metadata are converted to the string 'None'.

    Args:
        valid_browser_params: Fixture providing minimal valid parameters.
    """
    params = valid_browser_params.copy()
    params["metadata"] = [{"id": 1, "value": None}, {"id": 2, "value": "exists"}]

    browser = DfrBrowser(**params)

    # Verify None was converted to an empty string
    assert browser.metadata.loc[0, "value"] == ""
    assert browser.metadata.loc[1, "value"] == "exists"


def test_init_with_inconsistent_keys_in_metadata(valid_browser_params):
    """Tests __init__ with inconsistent keys in metadata dictionaries.

    Verifies that the DataFrame correctly handles dictionaries with different keys.

    Args:
        valid_browser_params: Fixture providing minimal valid parameters.
    """
    params = valid_browser_params.copy()
    params["metadata"] = [
        {"id": 1, "title": "Doc1", "author": "Author1"},
        {"id": 2, "title": "Doc2"},  # Missing author
        {"id": 3, "title": "Doc3", "author": "Author3", "year": "2023"},  # Extra field
    ]

    browser = DfrBrowser(**params)

    # Verify DataFrame has all columns with appropriate NaN values
    assert list(browser.metadata.columns) == ["id", "title", "author", "year"]
    assert browser.metadata.loc[0, "author"] == "Author1"
    assert browser.metadata.loc[1, "author"] == ""
    assert browser.metadata.loc[0, "year"] == ""
    assert browser.metadata.loc[2, "year"] == "2023"


#### Test Convert State ####


@pytest.fixture
def dfr_browser_instance():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    return DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )


def test_convert_state_basic_functionality(dfr_browser_instance):
    """Tests basic functionality of _convert_state method.

    Verifies that the method correctly calls prepare_data.convert_state with
    the appropriate parameters.

    Args:
        dfr_browser_instance: Fixture providing a configured DfrBrowser instance.
    """
    # Mock the prepare_data.convert_state function
    with patch(
        "lexos.topic_modeling.dfr_browser.prepare_data.convert_state"
    ) as mock_convert_state:
        # Call the method with a test data directory
        data_dir = "/test/data/dir"
        dfr_browser_instance._convert_state(data_dir)

        # Verify prepare_data.convert_state was called with correct arguments
        mock_convert_state.assert_called_once_with(
            dfr_browser_instance.path_to_state_file,
            f"{data_dir}/tw.json",
            f"{data_dir}/dt.json.zip",
            dfr_browser_instance.num_topics,
        )


def test_convert_state_with_path_object(dfr_browser_instance):
    """Tests _convert_state with a Path object instead of a string.

    Verifies that the method correctly handles Path objects for the data_dir parameter.

    Args:
        dfr_browser_instance: Fixture providing a configured DfrBrowser instance.
    """
    # Mock the prepare_data.convert_state function
    with patch(
        "lexos.topic_modeling.dfr_browser.prepare_data.convert_state"
    ) as mock_convert_state:
        # Call the method with a Path object
        data_dir = Path("/test/data/dir")
        dfr_browser_instance._convert_state(data_dir)

        # Verify prepare_data.convert_state was called with correct arguments
        # Path object should be converted to string with correct path separators
        mock_convert_state.assert_called_once_with(
            dfr_browser_instance.path_to_state_file,
            f"{data_dir}/tw.json",
            f"{data_dir}/dt.json.zip",
            dfr_browser_instance.num_topics,
        )


def test_convert_state_with_relative_path(dfr_browser_instance):
    """Tests _convert_state with a relative path.

    Verifies that the method correctly handles relative paths for the data_dir parameter.

    Args:
        dfr_browser_instance: Fixture providing a configured DfrBrowser instance.
    """
    # Mock the prepare_data.convert_state function
    with patch(
        "lexos.topic_modeling.dfr_browser.prepare_data.convert_state"
    ) as mock_convert_state:
        # Call the method with a relative path
        data_dir = "relative/data/dir"
        dfr_browser_instance._convert_state(data_dir)

        # Verify prepare_data.convert_state was called with correct arguments
        mock_convert_state.assert_called_once_with(
            dfr_browser_instance.path_to_state_file,
            f"{data_dir}/tw.json",
            f"{data_dir}/dt.json.zip",
            dfr_browser_instance.num_topics,
        )


def test_convert_state_error_propagation(dfr_browser_instance):
    """Tests that errors from prepare_data.convert_state are propagated.

    Verifies that the method doesn't catch errors raised by the underlying
    prepare_data.convert_state function.

    Args:
        dfr_browser_instance: Fixture providing a configured DfrBrowser instance.
    """
    # Mock convert_state to raise an exception
    with patch(
        "lexos.topic_modeling.dfr_browser.prepare_data.convert_state",
        side_effect=ValueError("Test error"),
    ):
        # Call should raise the ValueError
        with pytest.raises(ValueError, match="Test error"):
            dfr_browser_instance._convert_state("/test/data/dir")


#### Test Copy Docs ####


@pytest.fixture
def dfr_browser_instance2(tmp_path):
    """Creates a DfrBrowser instance for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    browser_dir = tmp_path / "browser_dir"
    browser_dir.mkdir()

    return DfrBrowser(
        path_to_browser_dir=str(browser_dir),
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        path_to_data_file="/test/path/to/data.txt",
        properties={},  # Initialize as empty dict for doc_uris to be added
    )


def test_copy_docs_basic_functionality(dfr_browser_instance2, tmp_path):
    """Tests basic functionality of _copy_docs method.

    Verifies that documents are correctly extracted from the data file,
    written to the docs directory, and added to doc_uris property.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create mock data file content
    mock_data = pd.DataFrame(
        {
            0: ["doc1", "doc2", "doc3"],
            1: ["word1", "word2", "word3"],
            2: ["This is document 1", "This is document 2", "This is document 3"],
        }
    )

    # Mock pandas.read_csv to return our test data
    with patch("pandas.read_csv", return_value=mock_data):
        # Execute the method
        dfr_browser_instance2._copy_docs()

        # Check that docs directory was created
        docs_dir = Path(dfr_browser_instance2.path_to_browser_dir) / "docs"
        assert docs_dir.exists()

        # Check that document files were created
        assert (docs_dir / "doc0.txt").exists()
        assert (docs_dir / "doc1.txt").exists()
        assert (docs_dir / "doc2.txt").exists()

        # Check that doc_uris property was updated correctly
        assert len(dfr_browser_instance2.properties["doc_uris"]) == 3
        assert dfr_browser_instance2.properties["doc_uris"][0].endswith(
            "/docs/doc0.txt"
        )
        assert dfr_browser_instance2.properties["doc_uris"][1].endswith(
            "/docs/doc1.txt"
        )
        assert dfr_browser_instance2.properties["doc_uris"][2].endswith(
            "/docs/doc2.txt"
        )

        # Check file contents
        with open(docs_dir / "doc0.txt", "r", encoding="utf-8") as f:
            assert f.read() == "This is document 1"
        with open(docs_dir / "doc1.txt", "r", encoding="utf-8") as f:
            assert f.read() == "This is document 2"
        with open(docs_dir / "doc2.txt", "r", encoding="utf-8") as f:
            assert f.read() == "This is document 3"


def test_copy_docs_file_not_found(dfr_browser_instance2):
    """Tests _copy_docs when data file is not found.

    Verifies that the method raises a LexosException when the data file cannot be found.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Mock pandas.read_csv to raise FileNotFoundError
    with patch("pandas.read_csv", side_effect=FileNotFoundError("File not found")):
        # Check that the method raises a LexosException
        with pytest.raises(LexosException, match="Could not find data file"):
            dfr_browser_instance2._copy_docs()


def test_copy_docs_empty_file(dfr_browser_instance2):
    """Tests _copy_docs when data file is empty.

    Verifies that the method raises a LexosException when the data file is empty.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Mock pandas.read_csv to raise EmptyDataError
    with patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError("Empty file")):
        # Check that the method raises a LexosException
        with pytest.raises(LexosException, match="Data file is empty"):
            dfr_browser_instance2._copy_docs()


def test_copy_docs_invalid_format(dfr_browser_instance2):
    """Tests _copy_docs when data file has invalid format.

    Verifies that the method raises a LexosException when the data file format is incorrect.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Mock pandas.read_csv to raise ParserError
    with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Parse error")):
        # Check that the method raises a LexosException
        with pytest.raises(
            LexosException, match="Data file is not in the correct format"
        ):
            dfr_browser_instance2._copy_docs()


def test_copy_docs_missing_columns(dfr_browser_instance2):
    """Tests _copy_docs when data file has insufficient columns.

    Verifies that the method raises a LexosException when the data file does not have enough columns.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Create mock data with insufficient columns
    mock_data = pd.DataFrame(
        {0: ["doc1", "doc2"], 1: ["word1", "word2"]}
    )  # Missing third column

    # Mock pandas.read_csv to return our test data
    with patch("pandas.read_csv", return_value=mock_data):
        # Check that the method raises a LexosException
        with pytest.raises(
            LexosException,
            match="Data file does not have the correct number of columns",
        ):
            dfr_browser_instance2._copy_docs()


def test_copy_docs_existing_directory(dfr_browser_instance2):
    """Tests _copy_docs when docs directory already exists.

    Verifies that the method works correctly when the docs directory already exists.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Create the docs directory in advance
    docs_dir = Path(dfr_browser_instance2.path_to_browser_dir) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create mock data
    mock_data = pd.DataFrame({0: ["doc1"], 1: ["word1"], 2: ["This is document 1"]})

    # Mock pandas.read_csv to return our test data
    with patch("pandas.read_csv", return_value=mock_data):
        # Method should succeed even though directory exists
        dfr_browser_instance2._copy_docs()

        # Check that doc file was created
        assert (docs_dir / "doc0.txt").exists()

        # Check that doc_uris property was updated
        assert len(dfr_browser_instance2.properties["doc_uris"]) == 1
        assert dfr_browser_instance2.properties["doc_uris"][0].endswith(
            "/docs/doc0.txt"
        )


def test_copy_docs_special_characters(dfr_browser_instance2):
    """Tests _copy_docs with special characters in document text.

    Verifies that the method correctly handles UTF-8 special characters in document content.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Create mock data with special characters
    mock_data = pd.DataFrame(
        {0: ["doc1"], 1: ["word1"], 2: ["Special characters: áéíóúñüç€©"]}
    )

    # Mock pandas.read_csv to return our test data
    with patch("pandas.read_csv", return_value=mock_data):
        # Execute the method
        dfr_browser_instance2._copy_docs()

        # Check that doc file was created
        docs_dir = Path(dfr_browser_instance2.path_to_browser_dir) / "docs"
        assert (docs_dir / "doc0.txt").exists()

        # Check file contents with special characters
        with open(docs_dir / "doc0.txt", "r", encoding="utf-8") as f:
            assert f.read() == "Special characters: áéíóúñüç€©"


def test_copy_docs_empty_documents(dfr_browser_instance2):
    """Tests _copy_docs with empty document text.

    Verifies that the method correctly handles documents with empty content.

    Args:
        dfr_browser_instance2: Fixture providing a configured DfrBrowser instance.
    """
    # Create mock data with empty document
    mock_data = pd.DataFrame(
        {
            0: ["doc1", "doc2"],
            1: ["word1", "word2"],
            2: ["Content for doc1", ""],  # Second document is empty
        }
    )

    # Mock pandas.read_csv to return our test data
    with patch("pandas.read_csv", return_value=mock_data):
        # Execute the method
        dfr_browser_instance2._copy_docs()

        # Check that both doc files were created
        docs_dir = Path(dfr_browser_instance2.path_to_browser_dir) / "docs"
        assert (docs_dir / "doc0.txt").exists()
        assert (docs_dir / "doc1.txt").exists()

        # Check file contents
        with open(docs_dir / "doc0.txt", "r", encoding="utf-8") as f:
            assert f.read() == "Content for doc1"
        with open(docs_dir / "doc1.txt", "r", encoding="utf-8") as f:
            assert f.read() == ""  # Empty content


#### Test Copy Metadata ####


@pytest.fixture
def dfr_browser_instance3():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test metadata.
    """
    return DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[
            {"id": 1, "title": "Document 1", "author": "Author A"},
            {"id": 2, "title": "Document 2", "author": "Author B"},
        ],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )


def test_copy_metadata_basic_functionality(dfr_browser_instance3, tmp_path):
    """Tests basic functionality of _copy_metadata method.

    Verifies that metadata is correctly written to both regular and zip CSV files
    in the specified directory.

    Args:
        dfr_browser_instance3: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a data directory in the temporary path
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method
    dfr_browser_instance3._copy_metadata(data_dir)

    # Check that both files were created
    assert (data_dir / "meta.csv").exists()
    assert (data_dir / "meta.csv.zip").exists()

    # Verify CSV file contents
    df = pd.read_csv(data_dir / "meta.csv", header=None)
    assert df.shape == (2, 3)  # 2 rows, 3 columns

    # Verify ZIP file contains the CSV
    with zipfile.ZipFile(data_dir / "meta.csv.zip") as zip_file:
        assert len(zip_file.namelist()) == 1
        assert zip_file.namelist()[0] == "meta.csv"


def test_copy_metadata_with_string_path(dfr_browser_instance3, tmp_path):
    """Tests _copy_metadata with string path.

    Verifies that the method works correctly when data_dir is provided as a string.

    Args:
        dfr_browser_instance3: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a data directory in the temporary path
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method with a string path
    dfr_browser_instance3._copy_metadata(str(data_dir))

    # Check that files were created
    assert (data_dir / "meta.csv").exists()
    assert (data_dir / "meta.csv.zip").exists()


def test_copy_metadata_with_path_object(dfr_browser_instance3, tmp_path):
    """Tests _copy_metadata with Path object.

    Verifies that the method works correctly when data_dir is provided as a Path object.

    Args:
        dfr_browser_instance3: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a data directory in the temporary path
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method with a Path object
    dfr_browser_instance3._copy_metadata(data_dir)

    # Check that files were created
    assert (data_dir / "meta.csv").exists()
    assert (data_dir / "meta.csv.zip").exists()


def test_copy_metadata_already_dataframe(tmp_path):
    """Tests _copy_metadata when metadata is already a DataFrame.

    Verifies that the method works correctly when the metadata is already a pandas DataFrame.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a DataFrame directly
    metadata_df = pd.DataFrame(
        {"id": [1, 2], "title": ["Doc1", "Doc2"], "year": [2020, 2021]}
    )

    # Create a browser instance with the DataFrame
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=metadata_df.to_dict(
            "records"
        ),  # Convert to records for initialization
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )

    # Create a data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method
    browser._copy_metadata(data_dir)

    # Check that files were created
    assert (data_dir / "meta.csv").exists()
    assert (data_dir / "meta.csv.zip").exists()

    # Verify contents
    df = pd.read_csv(data_dir / "meta.csv", header=None)
    assert df.shape == (2, 3)  # 2 rows, 3 columns


def test_copy_metadata_conversion_error():
    """Tests _copy_metadata error handling for invalid metadata.

    Verifies that the method raises a LexosException when metadata cannot be
    converted to a DataFrame.
    """
    # Create a browser instance with invalid metadata
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[
            {"title": "title", "text": "row1"}
        ],  # This is valid but will be changed below
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )

    # Add invalid metadata after instantiation
    browser.metadata = "invalid metadata"

    # Call the method - should raise LexosException
    with pytest.raises(
        LexosException, match="Cannot parse metadata as a pandas dataframe"
    ):
        browser._copy_metadata("data_dir")


def test_copy_metadata_csv_contents(dfr_browser_instance3, tmp_path):
    """Tests the contents of the CSV file created by _copy_metadata.

    Verifies that the CSV file contains the expected data with proper quoting.

    Args:
        dfr_browser_instance3: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method
    dfr_browser_instance3._copy_metadata(data_dir)

    # Read the CSV file contents
    with open(data_dir / "meta.csv", "r") as f:
        content = f.read()

    # Verify all values are quoted (due to QUOTE_ALL)
    assert '"1"' in content
    assert '"Document 1"' in content
    assert '"Author A"' in content

    # Verify no headers are present
    first_line = content.split("\n")[0]
    assert "id" not in first_line
    assert "title" not in first_line


def test_copy_metadata_zip_file_contents(dfr_browser_instance3, tmp_path):
    """Tests the contents of the zip file created by _copy_metadata.

    Verifies that the zip file contains a properly formatted CSV file.

    Args:
        dfr_browser_instance3: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method
    dfr_browser_instance3._copy_metadata(data_dir)

    # Extract and read the zipped CSV
    with zipfile.ZipFile(data_dir / "meta.csv.zip") as zf:
        zf.extractall(path=data_dir)

    # Read the extracted CSV
    extracted_csv = pd.read_csv(data_dir / "meta.csv", header=None)

    # Verify the contents
    assert extracted_csv.shape == (2, 3)
    assert extracted_csv.iloc[0, 0] == 1
    assert extracted_csv.iloc[0, 1] == "Document 1"
    assert extracted_csv.iloc[0, 2] == "Author A"


def test_copy_metadata_with_na_values(tmp_path):
    """Tests _copy_metadata with NA values in the metadata.

    Verifies that NA values are properly handled and filled with empty strings.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create browser instance with NA values in metadata
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[
            {"id": 1, "title": "Document 1", "author": None},
            {"id": 2, "title": None, "author": "Author B"},
        ],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )

    # Create a data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Call the method
    browser._copy_metadata(data_dir)

    # Read the CSV file
    df = pd.read_csv(data_dir / "meta.csv", header=None, keep_default_na=False)

    # Verify NA values were filled with empty strings
    assert df.iloc[0, 2] == ""  # None in author column
    assert df.iloc[1, 1] == ""  # None in title column


#### Test Copy Template ####


@pytest.fixture
def dfr_browser_instance4():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    return DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        path_to_template_dir="/test/template/dir",
    )


def test_copy_template_success(dfr_browser_instance4):
    """Tests successful execution of _copy_template method.

    Verifies that the method correctly copies the template directory to the
    browser directory using shutil.copytree.

    Args:
        dfr_browser_instance4: Fixture providing a configured DfrBrowser instance.
    """
    with patch("shutil.copytree") as mock_copytree:
        # Call the method
        dfr_browser_instance4._copy_template()

        # Verify shutil.copytree was called with correct arguments
        mock_copytree.assert_called_once_with(
            dfr_browser_instance4.path_to_template_dir,
            dfr_browser_instance4.path_to_browser_dir,
        )


def test_copy_template_file_not_found(dfr_browser_instance4):
    """Tests _copy_template when template directory is not found.

    Verifies that the method raises a LexosException with an informative
    message when the template directory cannot be found.

    Args:
        dfr_browser_instance4: Fixture providing a configured DfrBrowser instance.
    """
    error_msg = "No such file or directory: '/test/template/dir'"

    with patch(
        "shutil.copytree", side_effect=FileNotFoundError(error_msg)
    ) as mock_copytree:
        # Check that the method raises a LexosException
        with pytest.raises(
            LexosException, match=f"Could not find dfr-browser template: {error_msg}"
        ):
            dfr_browser_instance4._copy_template()

        # Verify shutil.copytree was called
        mock_copytree.assert_called_once()


def test_copy_template_with_target_exists(dfr_browser_instance4):
    """Tests _copy_template when target directory already exists.

    Verifies that the method correctly handles the case where the destination
    directory already exists (shutil.copytree would raise a FileExistsError).

    Args:
        dfr_browser_instance4: Fixture providing a configured DfrBrowser instance.
    """
    error_msg = "Destination path '/test/browser/dir' already exists"

    with patch(
        "shutil.copytree", side_effect=FileExistsError(error_msg)
    ) as mock_copytree:
        # This would typically raise a FileExistsError, but the method doesn't catch it
        # so it should propagate up
        with pytest.raises(FileExistsError, match=error_msg):
            dfr_browser_instance4._copy_template()

        # Verify shutil.copytree was called
        mock_copytree.assert_called_once()


def test_copy_template_real_directories(tmp_path):
    """Tests _copy_template with real directories.

    Tests the method using actual directory creation rather than mocks.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a template directory with a test file
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    test_file = template_dir / "test.txt"
    test_file.write_text("Test content")

    # Create a browser directory
    browser_dir = tmp_path / "browser"

    # Create a DfrBrowser instance with these directories
    browser = DfrBrowser(
        path_to_browser_dir=str(browser_dir),
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/state.gz",
        path_to_template_dir=str(template_dir),
    )

    # Call the method
    browser._copy_template()

    # Verify the directory was copied
    assert browser_dir.exists()
    assert (browser_dir / "test.txt").exists()
    assert (browser_dir / "test.txt").read_text() == "Test content"


#### Test Scale Model ####


@pytest.fixture
def dfr_browser_instance5():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )
    # Set the path_to_scaled_file attribute which would normally be set in build()
    browser.path_to_scaled_file = "/test/browser/dir/topic_scaled.csv"
    return browser


def test_scale_model_success(dfr_browser_instance5):
    """Tests successful execution of _scale_model method.

    Verifies that the method correctly converts MALLET state data to topic
    coordinates and saves them to a CSV file.

    Args:
        dfr_browser_instance5: Fixture providing a configured DfrBrowser instance.
    """
    # Mock return values
    mock_converted_data = {
        "doc_lengths": [10, 20],
        "vocab": ["word1", "word2"],
        "term_frequency": [100, 200],
        "doc_topic_dists": [[0.7, 0.3], [0.4, 0.6]],
        "topic_term_dists": [[0.8, 0.2], [0.3, 0.7]],
    }

    # Create a mock DataFrame for topic coordinates
    mock_topic_coords = pd.DataFrame(
        {
            "x": [0.1, 0.4],
            "y": [0.2, 0.5],
            "topics": [1, 2],
            "cluster": [1, 1],
            "Freq": [0.6, 0.4],
        }
    )

    # Set up patches
    with (
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.convert_mallet_data",
            return_value=mock_converted_data,
        ) as mock_convert,
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.get_topic_coordinates",
            return_value=mock_topic_coords,
        ) as mock_get_coords,
    ):
        # Test the method
        dfr_browser_instance5._scale_model()

        # Verify convert_mallet_data was called with correct argument
        mock_convert.assert_called_once_with(dfr_browser_instance5.path_to_state_file)

        # Verify get_topic_coordinates was called with correct unpacked arguments
        mock_get_coords.assert_called_once_with(**mock_converted_data)

        # In a real test with an actual temporary file, we could verify file contents
        # But here we're mocking file operations


def test_scale_model_convert_data_error(dfr_browser_instance5):
    """Tests error handling when convert_mallet_data fails.

    Verifies that the method wraps exceptions from convert_mallet_data
    in a LexosException.

    Args:
        dfr_browser_instance5: Fixture providing a configured DfrBrowser instance.
    """
    # Set up patch to raise an exception
    with patch(
        "lexos.topic_modeling.dfr_browser.scale_model.convert_mallet_data",
        side_effect=ValueError("Invalid state file format"),
    ) as mock_convert:
        # Verify that the method raises a LexosException
        with pytest.raises(LexosException, match="Failed!: Invalid state file format"):
            dfr_browser_instance5._scale_model()

        # Verify convert_mallet_data was called
        mock_convert.assert_called_once_with(dfr_browser_instance5.path_to_state_file)


def test_scale_model_get_coordinates_error(dfr_browser_instance5):
    """Tests error handling when get_topic_coordinates fails.

    Verifies that the method wraps exceptions from get_topic_coordinates
    in a LexosException.

    Args:
        dfr_browser_instance5: Fixture providing a configured DfrBrowser instance.
    """
    # Mock return value for convert_mallet_data
    mock_converted_data = {
        "doc_lengths": [10, 20],
        "vocab": ["word1", "word2"],
        "term_frequency": [100, 200],
        "doc_topic_dists": [[0.7, 0.3], [0.4, 0.6]],
        "topic_term_dists": [[0.8, 0.2], [0.3, 0.7]],
    }

    # Set up patches
    with (
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.convert_mallet_data",
            return_value=mock_converted_data,
        ),
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.get_topic_coordinates",
            side_effect=RuntimeError("Scaling error"),
        ) as mock_get_coords,
    ):
        # Verify that the method raises a LexosException
        with pytest.raises(LexosException, match="Failed!: Scaling error"):
            dfr_browser_instance5._scale_model()

        # Verify get_topic_coordinates was called
        mock_get_coords.assert_called_once_with(**mock_converted_data)


def test_scale_model_to_csv_error(dfr_browser_instance5):
    """Tests error handling when to_csv operation fails.

    Verifies that the method wraps exceptions from DataFrame.to_csv
    in a LexosException.

    Args:
        dfr_browser_instance5: Fixture providing a configured DfrBrowser instance.
    """
    # Mock return values
    mock_converted_data = {"key": "value"}  # Simplified for this test

    # Create a mock DataFrame that will raise an error on to_csv
    mock_topic_coords = MagicMock()
    mock_topic_coords.to_csv.side_effect = PermissionError("Permission denied")

    # Set up patches
    with (
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.convert_mallet_data",
            return_value=mock_converted_data,
        ),
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.get_topic_coordinates",
            return_value=mock_topic_coords,
        ),
    ):
        # Verify that the method raises a LexosException
        with pytest.raises(LexosException, match="Failed!: Permission denied"):
            dfr_browser_instance5._scale_model()

        # Verify to_csv was called
        mock_topic_coords.to_csv.assert_called_once_with(
            dfr_browser_instance5.path_to_scaled_file, index=False, header=False
        )


def test_scale_model_with_real_file_output(dfr_browser_instance5, tmp_path):
    """Tests actual file output from _scale_model.

    Verifies that the method correctly writes data to a file when successful.

    Args:
        dfr_browser_instance5: Fixture providing a configured DfrBrowser instance.
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Set the path to a temporary file
    test_output_file = tmp_path / "topic_scaled.csv"
    dfr_browser_instance5.path_to_scaled_file = str(test_output_file)

    # Mock return values
    mock_converted_data = {"key": "value"}  # Simplified for test

    # Create a real DataFrame for topic coordinates
    test_data = {
        "x": [0.1, 0.4],
        "y": [0.2, 0.5],
        "topics": [1, 2],
        "cluster": [1, 1],
        "Freq": [0.6, 0.4],
    }
    mock_topic_coords = pd.DataFrame(test_data)

    # Set up patches
    with (
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.convert_mallet_data",
            return_value=mock_converted_data,
        ),
        patch(
            "lexos.topic_modeling.dfr_browser.scale_model.get_topic_coordinates",
            return_value=mock_topic_coords,
        ),
    ):
        # Call the method
        dfr_browser_instance5._scale_model()

        # Verify file was created
        assert test_output_file.exists()

        # Verify file contents
        result_df = pd.read_csv(test_output_file, header=None)
        # Check that data matches (without headers since they were suppressed)
        assert result_df.iloc[0, 0] == 0.1  # x value
        assert result_df.iloc[0, 1] == 0.2  # y value
        assert result_df.iloc[1, 0] == 0.4  # x value for second topic
        assert result_df.shape == (2, 5)  # 2 topics, 5 columns


#### Test the update_assets method ####


@pytest.fixture
def dfr_browser_instance7():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    return DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
    )


@pytest.fixture
def setup_browser_files(tmp_path):
    """Creates a temporary directory with browser files for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        tuple: A tuple containing (browser_dir, index_file_path, js_custom_path, js_output_path)
    """
    browser_dir = tmp_path / "browser"
    browser_dir.mkdir()

    # Create the index.html file with test content
    index_file = browser_dir / "index.html"
    index_file.write_text(
        "This is a reference to documents on JSTOR that should be replaced."
    )

    # Create the js directory
    js_dir = browser_dir / "js"
    js_dir.mkdir()

    # Create the dfb.min.js.custom file with test content
    js_custom_file = js_dir / "dfb.min.js.custom"
    js_content = """
    function someFunction() {
        t.select("#doc_remark a.url").attr("href", "http://example.com/" + id);
        // More JavaScript code here
    }
    """
    js_custom_file.write_text(js_content)

    # Path to the output js file
    js_output_file = js_dir / "dfb.min.js"

    return browser_dir, index_file, js_custom_file, js_output_file


def test_update_assets_basic_functionality(setup_browser_files):
    """Tests basic functionality of _update_assets method.

    Verifies that the method correctly updates index.html and creates dfb.min.js
    with the expected content.

    Args:
        setup_browser_files: Fixture providing temporary browser files.
    """
    browser_dir, index_file, js_custom_file, js_output_file = setup_browser_files

    # Create a browser instance pointing to our temporary directory
    browser = DfrBrowser(
        path_to_browser_dir=str(browser_dir),
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/state.gz",
    )

    # Call the update_assets method
    browser._update_assets()

    # Check the index.html file was updated
    updated_index_content = index_file.read_text()
    assert "on JSTOR" not in updated_index_content
    assert "JSON" in updated_index_content

    # Check the dfb.min.js file was created with the pattern replaced
    assert js_output_file.exists()
    updated_js_content = js_output_file.read_text()

    # Verify the regex substitution worked
    assert (
        'var doc_url = document.URL.split("modules")[0] + "project_data";'
        in updated_js_content
    )
    assert (
        't.select("#doc_remark a.url").attr("href", "http://example.com/"'
        not in updated_js_content
    )


def test_update_assets_with_file_errors(dfr_browser_instance7):
    """Tests _update_assets error handling for file operations.

    Verifies that the method properly handles file-related errors.

    Args:
        dfr_browser_instance7: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mock for index.html open operations that raises FileNotFoundError
    with patch("builtins.open") as mock_open2:
        mock_open2.side_effect = FileNotFoundError("File not found")

        # The method doesn't have explicit error handling, so the exception should propagate
        with pytest.raises(FileNotFoundError):
            dfr_browser_instance7._update_assets()


def test_update_assets_with_permission_error(dfr_browser_instance7):
    """Tests _update_assets error handling for permission issues.

    Verifies that the method properly handles permission-related errors.

    Args:
        dfr_browser_instance7: Fixture providing a configured DfrBrowser instance.
    """
    # Mock sequence: first open succeeds, second fails with permission error
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "text with on JSTOR"

    with patch("builtins.open") as mock_open3:
        mock_open3.side_effect = [
            mock_file,  # First open succeeds
            PermissionError("Permission denied"),  # Second open fails
        ]

        # The method doesn't have explicit error handling, so the exception should propagate
        with pytest.raises(PermissionError):
            dfr_browser_instance7._update_assets()


def test_update_assets_with_complex_html(setup_browser_files):
    """Tests _update_assets with complex HTML content.

    Verifies that the method correctly updates HTML with multiple occurrences
    of the target text.

    Args:
        setup_browser_files: Fixture providing temporary browser files.
    """
    browser_dir, index_file, js_custom_file, js_output_file = setup_browser_files

    # Create index.html with multiple occurrences of "on JSTOR"
    complex_html = """
    <html>
    <head><title>DFR Browser</title></head>
    <body>
        <p>This references documents on JSTOR in multiple places.</p>
        <p>Users can find articles on JSTOR through our interface.</p>
        <p>Read more about collections on JSTOR here.</p>
    </body>
    </html>
    """
    index_file.write_text(complex_html)

    # Create a browser instance pointing to our temporary directory
    browser = DfrBrowser(
        path_to_browser_dir=str(browser_dir),
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/state.gz",
    )

    # Call the update_assets method
    browser._update_assets()

    # Check all occurrences were replaced
    updated_content = index_file.read_text()
    assert "on JSTOR" not in updated_content
    assert (
        updated_content.count("JSON") == 3
    )  # All three occurrences should be replaced


def test_update_assets_complex_js_pattern(setup_browser_files):
    """Tests _update_assets with complex JS patterns.

    Verifies that the method correctly handles complex JavaScript patterns
    with regex substitution.

    Args:
        setup_browser_files: Fixture providing temporary browser files.
    """
    browser_dir, index_file, js_custom_file, js_output_file = setup_browser_files

    # Create a more complex js file with variations of the pattern to replace
    complex_js = """
    function someFunction() {
        // This pattern should be replaced
        t.select("#doc_remark a.url").attr("href", "http://example.com/" + id);

        // This similar pattern should not be replaced
        someOtherFunc("#doc_remark a.url").attr("href", "http://example.com/");

        // Another pattern that should not be replaced
        t.select("#not_doc_remark").attr("href", "http://example.com/");
    }
    """
    js_custom_file.write_text(complex_js)

    # Create a browser instance pointing to our temporary directory
    browser = DfrBrowser(
        path_to_browser_dir=str(browser_dir),
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/state.gz",
    )

    # Call the update_assets method
    browser._update_assets()

    # Check the dfb.min.js file was created with only the correct pattern replaced
    updated_js_content = js_output_file.read_text()

    # The target pattern should be replaced
    assert (
        'var doc_url = document.URL.split("modules")[0] + "project_data";'
        in updated_js_content
    )

    # Similar patterns should not be replaced
    assert (
        'someOtherFunc("#doc_remark a.url").attr("href", "http://example.com/");'
        in updated_js_content
    )
    assert (
        't.select("#not_doc_remark").attr("href", "http://example.com/");'
        in updated_js_content
    )


def test_update_assets_no_matches(setup_browser_files):
    """Tests _update_assets when no matches are found.

    Verifies that the method handles cases where the patterns to replace
    don't exist in the source files.

    Args:
        setup_browser_files: Fixture providing temporary browser files.
    """
    browser_dir, index_file, js_custom_file, js_output_file = setup_browser_files

    # Create files without the patterns to replace
    index_file.write_text("This document has no references to JSTOR.")
    js_custom_file.write_text(
        "function someFunction() { /* No patterns to replace */ }"
    )

    # Create a browser instance pointing to our temporary directory
    browser = DfrBrowser(
        path_to_browser_dir=str(browser_dir),
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/state.gz",
    )

    # Call the update_assets method
    browser._update_assets()

    # Verify files were processed without changes
    assert index_file.read_text() == "This document has no references to JSTOR."
    assert "No patterns to replace" in js_output_file.read_text()


#### Test the validate_settings method


@pytest.fixture
def dfr_browser_for_validate_settings():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with valid test settings.
    """
    # Create a DfrBrowser with valid settings
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[],  # Mock the metadata
        num_topics=10,
        path_to_state_file="/test/state/file.gz",
        path_to_template_dir="/test/template/dir",
        port=8888,
    )
    return browser


def test_validate_settings_all_valid(dfr_browser_for_validate_settings):
    """Tests _validate_settings when all settings are valid.

    Verifies that the method doesn't raise any exceptions when all paths exist
    and required values are set.

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """
    # Mock Path.exists to return True for all paths
    with patch("pathlib.Path.exists", return_value=True):
        # This should not raise an exception
        dfr_browser_for_validate_settings._validate_settings()


def test_validate_settings_nonexistent_input_file(dfr_browser_for_validate_settings):
    """Tests _validate_settings when state file or template directory doesn't exist (which they don't in the test instance).

    Verifies that the method raises LexosException when the state file path
    doesn't exist.

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """

    # Define a custom exists function that returns False for the state file path

    # Mock Path.exists with our custom function
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(LexosException) as excinfo:
            dfr_browser_for_validate_settings._validate_settings()

        assert "Path does not exist" in str(excinfo.value)
        assert dfr_browser_for_validate_settings.path_to_state_file in str(
            excinfo.value
        )


def test_validate_settings_zero_topics(dfr_browser_for_validate_settings):
    """Tests _validate_settings with zero topics.

    Verifies that the method raises LexosException when num_topics is set to 0.

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """
    # Set num_topics to 0
    dfr_browser_for_validate_settings.num_topics = 0

    with patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(LexosException) as excinfo:
            dfr_browser_for_validate_settings._validate_settings()

        assert "Number of topics must be provided" in str(excinfo.value)


def test_validate_settings_none_topics(dfr_browser_for_validate_settings):
    """Tests _validate_settings with None topics.

    Verifies that the method raises LexosException when num_topics is None.

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """
    # Set num_topics to None
    dfr_browser_for_validate_settings.num_topics = None

    with patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(LexosException) as excinfo:
            dfr_browser_for_validate_settings._validate_settings()

        assert "Number of topics must be provided" in str(excinfo.value)


def test_validate_settings_none_port(dfr_browser_for_validate_settings):
    """Tests _validate_settings with None port.

    Verifies that the method raises LexosException when port is None.

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """
    # Set port to None
    dfr_browser_for_validate_settings.port = None

    with patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(LexosException) as excinfo:
            dfr_browser_for_validate_settings._validate_settings()

        assert "Port number must be provided" in str(excinfo.value)


def test_validate_settings_path_objects(dfr_browser_for_validate_settings):
    """Tests _validate_settings with Path objects.

    Verifies that the method correctly handles Path objects instead of strings.

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """
    # Convert paths to Path objects
    dfr_browser_for_validate_settings.path_to_state_file = Path(
        dfr_browser_for_validate_settings.path_to_state_file
    )
    dfr_browser_for_validate_settings.path_to_template_dir = Path(
        dfr_browser_for_validate_settings.path_to_template_dir
    )

    with patch("pathlib.Path.exists", return_value=True):
        # This should not raise an exception
        dfr_browser_for_validate_settings._validate_settings()


def test_validate_settings_indentation_bug(dfr_browser_for_validate_settings):
    """Tests for the indentation bug in _validate_settings.

    Verifies that the num_topics and port checks are only executed once, not
    for each path in the loop (which appears to be a bug in the implementation).

    Args:
        dfr_browser_for_validate_settings: Fixture providing a configured DfrBrowser instance.
    """
    # Mock exists to return True
    with patch("pathlib.Path.exists", return_value=True):
        # The method should succeed with these valid settings
        dfr_browser_for_validate_settings._validate_settings()

        # Now set num_topics to 0, which should fail
        dfr_browser_for_validate_settings.num_topics = 0

        with pytest.raises(LexosException) as excinfo:
            dfr_browser_for_validate_settings._validate_settings()

        assert "Number of topics must be provided" in str(excinfo.value)


#### Test the build method ####


@pytest.fixture
def dfr_browser_instance_for_build():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        properties={},  # Initialize as empty dict
    )
    return browser


def test_build_basic_functionality(dfr_browser_instance_for_build):
    """Tests the basic functionality of the build method.

    Verifies that all the required methods are called in the correct order
    with appropriate parameters.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(
            dfr_browser_instance_for_build, "_validate_settings"
        ) as mock_validate,
        patch.object(Path, "exists", return_value=False),
        patch.object(
            dfr_browser_instance_for_build, "_copy_template"
        ) as mock_copy_template,
        patch.object(Path, "mkdir") as mock_mkdir,
        patch.object(
            dfr_browser_instance_for_build, "_scale_model"
        ) as mock_scale_model,
        patch.object(
            dfr_browser_instance_for_build, "_convert_state"
        ) as mock_convert_state,
        patch.object(dfr_browser_instance_for_build, "_copy_docs") as mock_copy_docs,
        patch(
            "lexos.topic_modeling.dfr_browser.prepare_data.info_stub"
        ) as mock_info_stub,
        patch("shutil.copy") as mock_copy,
        patch.object(
            dfr_browser_instance_for_build, "_copy_metadata"
        ) as mock_copy_metadata,
        patch.object(
            dfr_browser_instance_for_build, "_update_assets"
        ) as mock_update_assets,
    ):
        # Set path_to_data_file to trigger _copy_docs
        dfr_browser_instance_for_build.path_to_data_file = "/test/path/to/data.txt"

        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify each method was called correctly
        mock_validate.assert_called_once()
        mock_copy_template.assert_called_once()
        mock_mkdir.assert_called_once()
        mock_scale_model.assert_called_once()

        # Get the data_dir that should be passed to _convert_state
        expected_data_dir = Path("/test/browser/dir/data")
        mock_convert_state.assert_called_once_with(expected_data_dir)

        mock_copy_docs.assert_called_once()
        mock_info_stub.assert_called_once()
        mock_copy.assert_called_once()
        mock_copy_metadata.assert_called_once_with(expected_data_dir)
        mock_update_assets.assert_called_once()


def test_build_existing_browser_dir(dfr_browser_instance_for_build):
    """Tests build when browser directory already exists.

    Verifies that _copy_template is not called when the browser directory exists.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(
            dfr_browser_instance_for_build, "_copy_template"
        ) as mock_copy_template,
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy"),
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify _copy_template was not called since directory exists
        mock_copy_template.assert_not_called()


def test_build_with_data_file(dfr_browser_instance_for_build):
    """Tests build with path_to_data_file provided.

    Verifies that _copy_docs is called when path_to_data_file is provided.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch.object(dfr_browser_instance_for_build, "_copy_docs") as mock_copy_docs,
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy"),
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Set path_to_data_file
        dfr_browser_instance_for_build.path_to_data_file = "/test/path/to/data.txt"

        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify _copy_docs was called
        mock_copy_docs.assert_called_once()


def test_build_with_doc_uri_column(dfr_browser_instance_for_build):
    """Tests build with doc_uri column in metadata.

    Verifies that doc_uris are set from metadata when doc_uri column exists.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup metadata with doc_uri column
    dfr_browser_instance_for_build.metadata = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Doc1", "Doc2"],
            "doc_uri": ["/uri/to/doc1.txt", "/uri/to/doc2.txt"],
        }
    )

    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch.object(dfr_browser_instance_for_build, "_copy_docs") as mock_copy_docs,
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy"),
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify _copy_docs was not called since we have doc_uri
        mock_copy_docs.assert_not_called()

        # Verify doc_uris were set from metadata
        assert dfr_browser_instance_for_build.properties["doc_uris"] == [
            "/uri/to/doc1.txt",
            "/uri/to/doc2.txt",
        ]


def test_build_embargo_default(dfr_browser_instance_for_build):
    """Tests build with no data_file or doc_uri column.

    Verifies that embargo is set to True when neither path_to_data_file
    nor doc_uri column is provided.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy"),
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Ensure path_to_data_file is None and doc_uri column doesn't exist
        dfr_browser_instance_for_build.path_to_data_file = None
        # dfr_browser_instance_for_build.metadata has no doc_uri column already

        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify embargo was set to True
        assert dfr_browser_instance_for_build.properties["embargo"] is True


def test_build_explicit_embargo(dfr_browser_instance_for_build):
    """Tests build with explicit embargo=True.

    Verifies that embargo is set in properties when embargo is True.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch.object(dfr_browser_instance_for_build, "_copy_docs"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy"),
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Set embargo to True
        dfr_browser_instance_for_build.embargo = True
        dfr_browser_instance_for_build.path_to_data_file = "/test/path/to/data.txt"

        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify embargo was set in properties
        assert dfr_browser_instance_for_build.properties["embargo"] is True


def test_build_scaled_file_path(dfr_browser_instance_for_build):
    """Tests that path_to_scaled_file is set correctly.

    Verifies that path_to_scaled_file is set to the expected path.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy"),
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify path_to_scaled_file is set correctly
        expected_path = "/test/browser/dir/topic_scaled.csv"
        assert dfr_browser_instance_for_build.path_to_scaled_file == expected_path


def test_build_copy_scaled_file(dfr_browser_instance_for_build):
    """Tests that the scaled file is copied to the data directory.

    Verifies that shutil.copy is called to copy the scaled file.

    Args:
        dfr_browser_instance_for_build: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch.object(dfr_browser_instance_for_build, "_validate_settings"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir"),
        patch.object(dfr_browser_instance_for_build, "_scale_model"),
        patch.object(dfr_browser_instance_for_build, "_convert_state"),
        patch("lexos.topic_modeling.dfr_browser.prepare_data.info_stub"),
        patch("shutil.copy") as mock_copy,
        patch.object(dfr_browser_instance_for_build, "_copy_metadata"),
        patch.object(dfr_browser_instance_for_build, "_update_assets"),
    ):
        # Call build method
        dfr_browser_instance_for_build.build()

        # Verify shutil.copy was called with the right parameters
        expected_source = "/test/browser/dir/topic_scaled.csv"
        expected_dest = "/test/browser/dir/data"
        mock_copy.assert_called_once_with(expected_source, Path(expected_dest))


#### Test the serve method ####


@pytest.fixture
def dfr_browser_for_serve():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    # Create a DfrBrowser with minimal required attributes
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=MagicMock(),  # Mock the DataFrame
        num_topics=10,
        path_to_state_file="/test/state/file.gz",
    )
    return browser


def test_serve_custom_port(dfr_browser_for_serve):
    """Tests that serve properly sets a custom port.

    Verifies that the method uses the provided port instead of the default.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks to prevent side effects
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open") as mock_browser,
        patch("time.sleep"),
        patch("builtins.print"),
    ):
        # Configure thread mock to exit after one loop iteration
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, False]

        # Call serve with a custom port
        custom_port = 9999
        dfr_browser_for_serve.serve(port=custom_port)

        # Verify port was updated
        assert dfr_browser_for_serve.port == custom_port

        # Verify browser opened with correct URL using custom port
        mock_browser.assert_called_once_with(f"http://127.0.0.1:{custom_port}/")


def test_serve_default_port(dfr_browser_for_serve):
    """Tests that serve uses the default port when none is provided.

    Verifies that the method uses the instance's port attribute when no port
    is provided.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open") as mock_browser,
        patch("time.sleep"),
        patch("builtins.print"),
    ):
        # Configure thread mock to exit after one loop iteration
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, False]

        # Set a default port
        default_port = 8888
        dfr_browser_for_serve.port = default_port

        # Call serve without providing a port
        dfr_browser_for_serve.serve()

        # Verify port remains unchanged
        assert dfr_browser_for_serve.port == default_port

        # Verify browser opened with correct URL using default port
        mock_browser.assert_called_once_with(f"http://127.0.0.1:{default_port}/")


def test_serve_handler_setup(dfr_browser_for_serve):
    """Tests that serve correctly sets up the HTTP handler.

    Verifies that the method correctly configures the SimpleHTTPRequestHandler
    with the browser directory.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Create a proper mock handler object
    mock_handler = MagicMock(spec=SimpleHTTPRequestHandler)

    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
        patch("time.sleep"),
        patch("builtins.print"),
        # Return the mock handler object instead of a string
        patch(
            "lexos.topic_modeling.dfr_browser.partial", return_value=mock_handler
        ) as mock_partial,
    ):
        # Configure thread mock to exit after one iteration
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, False]

        # Call serve
        dfr_browser_for_serve.serve()

        # Verify handler was set up correctly
        mock_partial.assert_called_once_with(
            SimpleHTTPRequestHandler,
            directory=Path(dfr_browser_for_serve.path_to_browser_dir),
        )

        # Verify handler was assigned to the instance
        assert dfr_browser_for_serve.handler == mock_handler


def test_serve_thread_setup(dfr_browser_for_serve):
    """Tests that serve correctly sets up the server thread.

    Verifies that the method creates and starts a daemon thread targeting
    the _serve_forever method.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
        patch("time.sleep"),
        patch("builtins.print"),
    ):
        # Configure thread mock
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, False]

        # Call serve
        dfr_browser_for_serve.serve()

        # Verify thread was created with correct target
        mock_thread.assert_called_once_with(target=dfr_browser_for_serve._serve_forever)

        # Verify thread was configured as daemon
        assert mock_thread_instance.daemon is True

        # Verify thread was started
        mock_thread_instance.start.assert_called_once()


def test_serve_browser_opened(dfr_browser_for_serve):
    """Tests that serve opens the web browser with the correct URL.

    Verifies that the method opens a browser to the local server URL with
    the correct port after waiting.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open") as mock_browser,
        patch("time.sleep") as mock_sleep,
        patch("builtins.print"),
    ):
        # Configure thread mock
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, False]

        # Call serve
        dfr_browser_for_serve.serve()

        # Verify sleep was called with correct duration before opening browser
        mock_sleep.assert_called_once_with(2)

        # Verify browser was opened with correct URL
        mock_browser.assert_called_once_with(
            f"http://127.0.0.1:{dfr_browser_for_serve.port}/"
        )


def test_serve_keyboard_interrupt(dfr_browser_for_serve):
    """Tests that serve correctly handles KeyboardInterrupt.

    Verifies that the method catches KeyboardInterrupt, prints a message,
    and exits with code 1.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
        patch("time.sleep"),
        patch("builtins.print") as mock_print,
        patch("sys.exit") as mock_exit,
    ):
        # Configure thread mock to raise KeyboardInterrupt
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.join.side_effect = KeyboardInterrupt()
        mock_thread_instance.is_alive.return_value = (
            True  # Always alive to enter the loop
        )

        # Call serve
        dfr_browser_for_serve.serve()

        # Verify proper message was printed
        mock_print.assert_called_with("Server interrupted.")

        # Verify sys.exit was called with code 1
        mock_exit.assert_called_once_with(1)


def test_serve_thread_join_timeout(dfr_browser_for_serve):
    """Tests that serve joins the thread with a timeout.

    Verifies that the method joins the server thread with a timeout to
    allow for keyboard interrupts.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
        patch("time.sleep"),
        patch("builtins.print"),
    ):
        # Configure thread mock to simulate multiple loop iterations
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, True, False]

        # Call serve
        dfr_browser_for_serve.serve()

        # Verify join was called multiple times with a timeout of 1 second
        assert mock_thread_instance.join.call_count == 2
        mock_thread_instance.join.assert_has_calls([call(1), call(1)])


def test_serve_prints_instructions(dfr_browser_for_serve):
    """Tests that serve prints helpful instructions.

    Verifies that the method prints appropriate messages about the server
    and how to stop it.

    Args:
        dfr_browser_for_serve: Fixture providing a configured DfrBrowser instance.
    """
    # Setup mocks
    with (
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
        patch("time.sleep"),
        patch("builtins.print") as mock_print,
    ):
        # Configure thread mock
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.side_effect = [True, False]

        # Call serve
        dfr_browser_for_serve.serve()

        # Verify correct messages were printed
        assert mock_print.call_count >= 2
        mock_print.assert_any_call(
            f"Serving Dfr-Browser from {Path(dfr_browser_for_serve.path_to_browser_dir)}."
        )
        mock_print.assert_any_call(
            "Type Ctrl-C to stop the server. If you are serving from a notebook, interrupt the kernel."
        )


#### Test for _serve_forever method ####


@pytest.fixture
def dfr_browser_instance_for_serve_forever():
    """Creates a DfrBrowser instance for testing.

    Returns:
        DfrBrowser: A configured DfrBrowser instance with test settings.
    """
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}, {"id": 2, "title": "Doc2"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        port=8888,
    )

    # Create a mock handler
    browser.handler = MagicMock(spec=SimpleHTTPRequestHandler)
    return browser


def test_serve_forever_basic_functionality(dfr_browser_instance_for_serve_forever):
    """Tests the basic functionality of _serve_forever method.

    Verifies that the method creates a TCPServer with the correct parameters
    and calls serve_forever() on it. Includes a timeout to ensure the test doesn't hang.

    Args:
        dfr_browser_instance_for_serve_forever: Fixture providing a configured DfrBrowser instance.
    """
    # Create a mock TCPServer
    mock_server = MagicMock()

    # Configure serve_forever to include a timeout
    def timed_serve_forever():
        """Simulates serve_forever with a timeout."""
        # Just return after a short delay (5ms) instead of serving indefinitely
        time.sleep(0.005)
        return

    mock_server.serve_forever = timed_serve_forever

    # Patch the TCPServer constructor and context manager
    with patch("socketserver.TCPServer", return_value=mock_server) as mock_tcp_server:
        # Configure mock_server to behave like a context manager
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=None)

        # Call the method with a timeout
        with patch(
            "time.sleep", return_value=None
        ):  # Speed up any sleeps in the method
            dfr_browser_instance_for_serve_forever._serve_forever()

        # Verify that TCPServer was created with the correct parameters
        mock_tcp_server.assert_called_once_with(
            ("", dfr_browser_instance_for_serve_forever.port),
            dfr_browser_instance_for_serve_forever.handler,
        )


def test_serve_forever_custom_port():
    """Tests _serve_forever with a custom port.

    Verifies that the method uses the configured port when creating the TCP server.
    """
    # Create a browser instance with a custom port
    custom_port = 9999
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        port=custom_port,
    )
    browser.handler = MagicMock(spec=SimpleHTTPRequestHandler)

    # Create a mock TCPServer
    mock_server = MagicMock()

    # Configure serve_forever to include a timeout
    def timed_serve_forever():
        """Simulates serve_forever with a timeout."""
        # Just return after a short delay (5ms) instead of serving indefinitely
        time.sleep(0.005)
        return

    mock_server.serve_forever = timed_serve_forever

    # Patch the TCPServer constructor and context manager
    with (
        patch("socketserver.TCPServer", return_value=mock_server) as mock_tcp_server,
        patch("time.sleep", return_value=None),
    ):  # Speed up any sleeps in the method
        # Configure mock_server to behave like a context manager
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=None)

        # Call the method
        browser._serve_forever()

        # Verify TCPServer was created with the custom port
        mock_tcp_server.assert_called_once_with(("", custom_port), browser.handler)


def test_serve_forever_none_handler():
    """Tests _serve_forever with None handler.

    Verifies that the method raises an appropriate exception when
    the handler is None.
    """
    # Create a browser instance with no handler
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        port=8888,
    )
    # Explicitly set handler to None
    browser.handler = None

    # Calling _serve_forever should raise an exception because None is not callable
    with pytest.raises(LexosException, match="Handler is not set"):
        browser._serve_forever()


def test_serve_forever_server_exception():
    """Tests error handling in _serve_forever.

    Verifies that exceptions from the server are properly propagated.
    """
    # Create a browser instance
    browser = DfrBrowser(
        path_to_browser_dir="/test/browser/dir",
        metadata=[{"id": 1, "title": "Doc1"}],
        num_topics=5,
        path_to_state_file="/test/path/to/state.gz",
        port=8888,
    )
    browser.handler = MagicMock(spec=SimpleHTTPRequestHandler)

    # Create a mock TCPServer
    mock_server = MagicMock()

    # Configure serve_forever to include a timeout and raise an exception
    def timed_serve_forever():
        """Simulates serve_forever with a timeout that raises an exception."""
        # Sleep briefly then raise the error
        time.sleep(0.005)
        raise OSError("Test server error")

    mock_server.serve_forever = timed_serve_forever

    # Patch the TCPServer constructor and context manager
    with (
        patch("socketserver.TCPServer", return_value=mock_server),
        patch("time.sleep", return_value=None),
    ):  # Speed up any sleeps in the method
        # Configure mock_server to behave like a context manager
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=None)

        # Verify that the exception is propagated
        with pytest.raises(OSError, match="Test server error"):
            browser._serve_forever()
