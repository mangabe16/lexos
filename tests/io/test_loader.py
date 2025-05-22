"""test_loader.py.

Last Update: 2025-01-14

Note: Some tests use mocked data, but real files in temporary directories are
created where functions have context managers (since those are hard to mock).
"""

import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, mock_open, patch

import pandas as pd
import pytest
from lexos.exceptions import LexosException
from lexos.io.loader import (
    DOCX_TYPES,
    PDF_TYPES,
    TEXT_TYPES,
    ZIP_TYPES,
    DataLoader,
    Loader,
)

# Fixtures


@pytest.fixture
def sample_paths():
    """Fixture to create sample paths."""
    return ["test1.txt", "test2.txt", "test3.txt"]


@pytest.fixture
def sample_mime_types():
    """Fixture to create sample mime_types."""
    return ["text/plain", "text/plain", "text/plain"]


@pytest.fixture
def sample_names():
    """Fixture to create sample names."""
    return ["test1", "test2", "test3"]


@pytest.fixture
def sample_texts():
    """Fixture to create sample texts."""
    return ["test1", "test2", "test3"]


@pytest.fixture
def sample_errors():
    """Fixture to create sample errors."""
    return []


@pytest.fixture
def sample_dataframe():
    """Fixture to create a dataframe with sample data."""
    return pd.DataFrame(
        [
            {"name": "test1", "text": "test1"},
            {"name": "test2", "text": "test2"},
            {"name": "test3", "text": "test3"},
        ]
    )


@pytest.fixture
def sample_dataframe_invalid_data():
    """Fixture to create a dataframe with invalid sample data."""
    return pd.DataFrame(
        [
            {"title": "test1", "content": "test1"},
            {"title": "test2", "content": "test2"},
            {"title": "test3", "content": "test3"},
        ]
    )


@pytest.fixture
def dataframe():
    """Fixture to create a dataframe from sample data."""
    return []


@pytest.fixture
def loader():
    """Fixture to create an empty Loader instance."""
    return Loader()


@pytest.fixture
def loader_with_data(
    sample_paths, sample_mime_types, sample_names, sample_texts, sample_errors
):
    """Fixture to create a baseloader with data."""
    loader = Loader()
    loader.paths = sample_paths
    loader.mime_types = sample_mime_types
    loader.names = sample_names
    loader.texts = sample_texts
    loader.errors = sample_errors
    return loader


@pytest.fixture
def mock_mime():
    """Fixture to create a mock mime type."""
    mime = MagicMock()
    mime.from_buffer.side_effect = lambda x: "text/plain"
    return mime


@pytest.fixture
def mock_document():
    """Fixture to create a mock document."""
    doc = Mock()
    doc.paragraphs = [Mock(text="Paragraph 1"), Mock(text="Paragraph 2")]
    return doc


@pytest.fixture
def mock_pdf_multi_page():
    """Fixture to create a mock PDF reader with multiple pages."""
    mock_pages = [Mock(), Mock(), Mock()]
    for i, page in enumerate(mock_pages, 1):
        page.extract_text.return_value = f"Page {i} content"
    mock_reader = Mock()
    mock_reader.pages = mock_pages
    return mock_reader


@pytest.fixture
def mock_pdf_single_page():
    """Fixture to create a mock PDF reader with a single page."""
    mock_page = Mock()
    mock_page.extract_text.return_value = "Page 1 content"
    mock_reader = Mock()
    mock_reader.pages = [mock_page]
    return mock_reader


@pytest.fixture
def mock_text_file():
    """Fixture to create a mock text file."""
    mock_file = Mock()
    mock_file.open.return_value = "Text content"
    return mock_file


#########################################################################################################
# Tests


def test_loader_init(loader):
    """Test Loader initialization."""
    assert isinstance(loader, Loader)
    assert loader.paths == []
    assert loader.mime_types == []
    assert loader.names == []
    assert loader.texts == []
    assert loader.errors == []


def test_loader_data(
    loader_with_data,
    sample_paths,
    sample_mime_types,
    sample_names,
    sample_texts,
    sample_errors,
):
    """Test Loader data property."""
    assert loader_with_data.data == {
        "paths": sample_paths,
        "mime_types": sample_mime_types,
        "names": sample_names,
        "texts": sample_texts,
        "errors": sample_errors,
    }


def test_loader_records(
    loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test Loader records property."""
    assert loader_with_data.records == [
        {
            "name": sample_names[0],
            "path": sample_paths[0],
            "mime_type": sample_mime_types[0],
            "text": sample_texts[0],
        },
        {
            "name": sample_names[1],
            "path": sample_paths[1],
            "mime_type": sample_mime_types[1],
            "text": sample_texts[1],
        },
        {
            "name": sample_names[2],
            "path": sample_paths[2],
            "mime_type": sample_mime_types[2],
            "text": sample_texts[2],
        },
    ]


def test_loader_iter(
    loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test Loader records property."""
    assert isinstance(loader_with_data.__iter__(), Generator)
    for item in loader_with_data:
        assert item in [
            {
                "name": sample_names[0],
                "path": sample_paths[0],
                "mime_type": sample_mime_types[0],
                "text": sample_texts[0],
            },
            {
                "name": sample_names[1],
                "path": sample_paths[1],
                "mime_type": sample_mime_types[1],
                "text": sample_texts[1],
            },
            {
                "name": sample_names[2],
                "path": sample_paths[2],
                "mime_type": sample_mime_types[2],
                "text": sample_texts[2],
            },
        ]


def test_loader_df(
    loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test Loader df property."""
    assert isinstance(loader_with_data.df, pd.DataFrame)
    df = pd.DataFrame(
        [
            {
                "name": sample_names[0],
                "path": sample_paths[0],
                "mime_type": sample_mime_types[0],
                "text": sample_texts[0],
            },
            {
                "name": sample_names[1],
                "path": sample_paths[1],
                "mime_type": sample_mime_types[1],
                "text": sample_texts[1],
            },
            {
                "name": sample_names[2],
                "path": sample_paths[2],
                "mime_type": sample_mime_types[2],
                "text": sample_texts[2],
            },
        ]
    )
    assert loader_with_data.df.equals(df)


def test_loader_df_empty(loader):
    """Test Loader df property."""
    assert loader.df.empty


def test_loader_reset(loader_with_data):
    """Test Loader reset method."""
    loader_with_data.errors == [1]
    assert loader_with_data.paths != []
    assert loader_with_data.mime_types != []
    assert loader_with_data.names != []
    assert loader_with_data.texts != []
    loader_with_data.reset()
    assert loader_with_data.paths == []
    assert loader_with_data.mime_types == []
    assert loader_with_data.names == []
    assert loader_with_data.texts == []
    assert loader_with_data.errors == []


def test_data_loader_load_dataset(loader_with_data):
    """Test DataLoader load_dataset method."""
    loader = loader_with_data
    dataset = DataLoader()
    dataset.paths = ["test4.txt", "test5.txt", "test6.txt"]
    dataset.mime_types = ["text/plain", "text/plain", "text/plain"]
    dataset.names = ["test4", "test5", "test6"]
    dataset.texts = ["test4", "test5", "test6"]
    dataset.errors = []
    loader.load_dataset(dataset)
    assert loader.paths == [
        "test1.txt",
        "test2.txt",
        "test3.txt",
        "test4.txt",
        "test5.txt",
        "test6.txt",
    ]
    assert loader.mime_types == [
        "text/plain",
        "text/plain",
        "text/plain",
        "text/plain",
        "text/plain",
        "text/plain",
    ]
    assert loader.names == ["test1", "test2", "test3", "test4", "test5", "test6"]
    assert loader.texts == ["test1", "test2", "test3", "test4", "test5", "test6"]
    assert loader.errors == []


def test_data_loader_load_invalid_dataset(loader_with_data):
    """Test DataLoader load_dataset method with invalid data type."""
    invalid_data = ["test4.txt", "test5.txt", "test6.txt"]
    with pytest.raises(LexosException, match="Invalid dataset type."):
        loader_with_data.load_dataset(invalid_data)


def test_load_single_file(loader, tmp_path):
    """Test loading a single file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    with (
        patch("builtins.open", mock_open(read_data=b"test content")),
        patch.object(loader, "_load_text_file") as mock_load_text,
    ):
        loader.load([test_file])

        mock_load_text.assert_called_once_with(test_file, "text/plain")
        assert len(loader.errors) == 0


def test_load_directory(loader, tmp_path):
    """Test loading a directory with multiple files."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")

    with (
        patch("builtins.open", mock_open(read_data=b"test content")),
        patch.object(loader, "_load_text_file") as mock_load_text,
    ):
        loader.load([test_dir])

        assert mock_load_text.call_count == 2


def test_load_dispatches_to_pdf(loader, tmp_path):
    """Test loading a file with PDF MIME type."""
    path = tmp_path / "file.pdf"
    path.write_bytes(b"%PDF-1.4")

    with (
        patch.object(loader.mime, "from_buffer", return_value="application/pdf"),
        patch.object(loader, "_load_pdf_file") as mock_pdf,
    ):
        loader.load([path])
        mock_pdf.assert_called_once_with(path)


def test_load_dispatches_to_docx(loader, tmp_path):
    """Test loading a file with DOCX MIME type."""
    path = tmp_path / "file.docx"
    path.write_bytes(b"PK\x03\x04")

    with (
        patch.object(
            loader.mime,
            "from_buffer",
            return_value="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        patch.object(loader, "_load_docx_file") as mock_docx,
    ):
        loader.load([path])
        mock_docx.assert_called_once_with(path)


def test_load_dispatches_to_zip(loader, tmp_path):
    """Test loading a file with ZIP MIME type."""
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file.txt", "content")

    with (
        patch.object(loader.mime, "from_buffer", return_value="application/zip"),
        patch.object(loader, "_load_zip_file") as mock_zip,
    ):
        loader.load([zip_path])
        mock_zip.assert_called_once_with(zip_path)


def test_load_invalid_mime_type(loader, tmp_path):
    """Test loading a file with an invalid MIME type.

    Note: The `load()` method looks for a pickle file extension and sets the MIME type to
    "application/vnd.python.pickle" if it finds one. The main purpose of this procedure is
    to allow this test to run, although it does add some minimal security.
    """
    test_file = tmp_path / "test.pickle"
    test_file.write_text("test content")

    with (
        patch("builtins.open", mock_open(read_data=b"test content")),
        patch.object(loader, "_load_text_file"),
    ):
        loader.load([test_file])
        assert len(loader.errors) == 1
        assert "Invalid MIME type" in loader.errors[0]


def test_load_io_error(loader):
    """Test loading a file that raises an IOError."""
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = IOError("Test IO Error")

        loader.load(["nonexistent.txt"])

        assert len(loader.errors) == 1
        assert isinstance(loader.errors[0], IOError)


def test_load_docx_file_success(loader, mock_document, tmp_path):
    """Test loading a DOCX file."""
    test_file = tmp_path / "test.docx"
    test_file.touch()

    with patch("lexos.io.loader.Document", return_value=mock_document):
        loader._load_docx_file(test_file)

        assert len(loader.names) == 1
        assert loader.names[0] == "test.docx"
        assert len(loader.mime_types) == 1
        assert loader.mime_types[0] == "application/docx"
        assert len(loader.texts) == 1
        assert loader.texts[0] == "Paragraph 1\nParagraph 2"
        assert len(loader.errors) == 0


def test_load_docx_file_empty(loader, tmp_path):
    """Test loading an empty DOCX file."""
    mock_empty_doc = Mock()
    mock_empty_doc.paragraphs = []
    test_file = tmp_path / "empty.docx"
    test_file.touch()

    with patch("lexos.io.loader.Document", return_value=mock_empty_doc):
        loader._load_docx_file(test_file)

        assert len(loader.texts) == 1
        assert loader.texts[0] == ""
        assert len(loader.errors) == 0


def test_load_docx_file_not_found(loader):
    """Test loading a DOCX file that does not exist."""
    with patch(
        "lexos.io.loader.Document", side_effect=FileNotFoundError("File not found")
    ):
        loader._load_docx_file("nonexistent.docx")

        assert len(loader.errors) == 1
        assert isinstance(loader.errors[0], FileNotFoundError)


def test_load_docx_file_permission_error(loader, tmp_path):
    """Test loading a DOCX file with permission error."""
    test_file = tmp_path / "noperm.docx"
    test_file.touch()

    with patch(
        "lexos.io.loader.Document", side_effect=PermissionError("Permission denied")
    ):
        loader._load_docx_file(test_file)

        assert len(loader.errors) == 1
        assert isinstance(loader.errors[0], PermissionError)


def test_load_pdf_single_page(loader, mock_pdf_single_page, tmp_path):
    """Test loading a single-page PDF file."""
    test_file = tmp_path / "test.pdf"
    test_file.touch()

    with patch("lexos.io.loader.PdfReader", return_value=mock_pdf_single_page):
        loader._load_pdf_file(test_file)

        assert len(loader.names) == 1
        assert loader.names[0] == "test.pdf"
        assert len(loader.mime_types) == 1
        assert loader.mime_types[0] == "application/pdf"
        assert len(loader.texts) == 1
        assert loader.texts[0] == "Page 1 content"
        assert len(loader.errors) == 0


def test_load_pdf_multi_page(loader, mock_pdf_multi_page, tmp_path):
    """Test loading a multi-page PDF file."""
    test_file = tmp_path / "test.pdf"
    test_file.touch()

    with patch("lexos.io.loader.PdfReader", return_value=mock_pdf_multi_page):
        loader._load_pdf_file(test_file)

        assert len(loader.names) == 3
        assert all(name == "test.pdf" for name in loader.names)
        assert len(loader.mime_types) == 3
        assert all(mime == "application/pdf" for mime in loader.mime_types)
        assert len(loader.texts) == 3
        assert loader.texts == ["Page 1 content", "Page 2 content", "Page 3 content"]


def test_load_pdf_empty(loader, tmp_path):
    """Test loading an empty PDF file."""
    mock_empty_pdf = Mock()
    mock_empty_pdf.pages = []
    test_file = tmp_path / "empty.pdf"
    test_file.touch()

    with patch("lexos.io.loader.PdfReader", return_value=mock_empty_pdf):
        loader._load_pdf_file(test_file)
        assert len(loader.texts) == 0
        assert len(loader.errors) == 0


def test_load_pdf_error(loader, tmp_path):
    """Test loading a PDF file that raises an error."""
    test_file = tmp_path / "error.pdf"
    test_file.touch()

    with patch("lexos.io.loader.PdfReader", side_effect=Exception("PDF Error")):
        loader._load_pdf_file(test_file)
        assert len(loader.errors) == 1
        assert str(loader.errors[0]) == "PDF Error"


def create_text_file(temp_dir, mime_type="text/plain"):
    """Create a sample text file."""
    # Create zip file path
    if mime_type == "text/plain":
        file_path = Path(temp_dir) / "test.txt"
        content = "Some content"
    else:
        file_path = Path(temp_dir) / "test.html"
        content = "<p>Some content</p>"

    # Create sample files
    with open(file_path, "w") as f:
        f.write(content)

    return file_path


def test_load_text_file_success(loader):
    """Test successful text file loading."""
    temp_dir = tempfile.TemporaryDirectory()
    file_path = create_text_file(temp_dir.name)
    loader._load_text_file(file_path, "text/plain")
    temp_dir.cleanup()
    assert len(loader.paths) == 1
    assert len(loader.texts) == 1
    assert len(loader.texts) == 1
    assert len(loader.mime_types) == 1
    assert loader.paths[0] == "test.txt"
    assert loader.names[0] == "test"
    assert loader.mime_types[0] == "text/plain"
    assert loader.texts[0] == "Some content"
    assert len(loader.errors) == 0


def test_load_text_file_failed(loader):
    """Test failed text file loading."""
    loader._load_text_file("bad_file_path.txt", "text/plain")
    assert len(loader.paths) == 0
    assert len(loader.texts) == 0
    assert len(loader.texts) == 0
    assert len(loader.mime_types) == 0
    assert len(loader.errors) == 1


def test_load_text_file_different_mime_types(loader):
    """Test different mime types."""
    temp_dir = tempfile.TemporaryDirectory()
    file_path = create_text_file(temp_dir.name, mime_type="text/html")
    loader._load_text_file(file_path, "text/html")
    temp_dir.cleanup()
    assert len(loader.paths) == 1
    assert len(loader.texts) == 1
    assert len(loader.texts) == 1
    assert len(loader.mime_types) == 1
    assert loader.paths[0] == "test.html"
    assert loader.names[0] == "test"
    assert loader.mime_types[0] == "text/html"
    assert loader.texts[0] == "<p>Some content</p>"
    assert len(loader.errors) == 0


def create_zip_file(temp_dir, empty=False, invalid=False):
    """Create a sample zip file with text files."""
    # Create zip file path
    zip_path = Path(temp_dir) / "test.zip"

    # Create sample files
    if empty:
        files = {}
    elif invalid:
        files = {"file1.invalid": pickle.dumps({"invalid": "content"})}
    else:
        files = {
            "file1.txt": "This is file 1 content",
            "file2.txt": "This is file 2 content",
        }

    # Create zip archive
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return zip_path


def test_load_zip(loader):
    """Test loading from a zip archive."""
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = create_zip_file(temp_dir.name, empty=False)
    loader._load_zip_file(zip_path)
    temp_dir.cleanup()
    assert len(loader.names) == 2
    assert len(loader.mime_types) == 2
    assert len(loader.texts) == 2
    assert loader.names == ["file1", "file2"]
    assert loader.mime_types == ["text/plain", "text/plain"]
    assert loader.texts == ["This is file 1 content", "This is file 2 content"]


def test_load_zip_empty(loader):
    """Test loading from an empty zip archive."""
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = create_zip_file(temp_dir.name, empty=True)
    loader._load_zip_file(zip_path)
    temp_dir.cleanup()
    assert len(loader.names) == 0
    assert len(loader.mime_types) == 0
    assert len(loader.texts) == 0


def test_load_invalid_content(loader):
    """Test loading from a zip archive with invalid content."""
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = create_zip_file(temp_dir.name, invalid=True)
    loader._load_zip_file(zip_path)
    temp_dir.cleanup()
    assert len(loader.names) == 0
    assert len(loader.mime_types) == 0
    assert len(loader.texts) == 0
    assert len(loader.errors) == 1


def test_zip_file_with_decode_failure(loader, tmp_path):
    """Force decode error inside _load_zip_file using mocking."""
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("bad.txt", b"valid content")

    with patch(
        "lexos.io.loader.decode",
        side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "boom"),
    ):
        loader._load_zip_file(zip_path)

    assert len(loader.errors) == 1
    assert isinstance(loader.errors[0], UnicodeDecodeError)


def test_loads_with_default_names(loader):
    """Test the loads() method with default name generation."""
    texts = ["First text", "Second text", "Third text"]
    loader.loads(texts=texts, names=None, start=1, zero_pad="02")

    assert loader.names == ["text01", "text02", "text03"]
    assert loader.mime_types == ["text/plain"] * 3
    assert loader.texts == texts
    assert len(loader.errors) == 0
