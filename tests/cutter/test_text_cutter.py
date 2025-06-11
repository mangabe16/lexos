"""test_text_cutter.py.

Last updated: 2025-06-26
"""

from pathlib import Path

import pytest
from lexos.cutter.text_cutter import TextCutter
from lexos.exceptions import LexosException
from lexos.milestones.string_milestones import StringSpan
from pydantic import ValidationError

# Fixtures

@pytest.fixture
def cutter():
    """Create a new instance of TextCutter for general use."""
    return TextCutter()

@pytest.fixture
def sample_text():
    """Return a sample multiline string used for buffer-based tests."""
    return "Line1\nLine2\nLine3\nLine4\nLine5\n"

@pytest.fixture
def text_file(tmp_path, sample_text):
    """Create a temporary text file from sample_text and return the file path."""
    file_path = tmp_path / "test.txt"
    file_path.write_text(sample_text)
    return file_path

@pytest.fixture
def sample_milestones_text():
    """Return a single-line text string used for milestone tests."""
    return "The quick brown fox jumps over the lazy dog."

@pytest.fixture
def mock_milestones():
    """Provide a list of mock StringSpan milestone markers for testing."""
    return [
        StringSpan(start=4, end=9, text="quick"),
        StringSpan(start=26, end=30, text="over")
    ]

@pytest.fixture
def milestones_text_file(tmp_path, sample_milestones_text):
    """Create a temporary file containing sample milestone text."""
    file_path = tmp_path / "test.txt"
    file_path.write_text(sample_milestones_text)
    return file_path

@pytest.fixture
def sample_paths():
    """Return a list of dummy Path objects representing text files."""
    return [
        Path("test1.txt"),
        Path("test2.txt")
    ]

@pytest.fixture
def custom_names():
    """Provide a list of custom string-based document names."""
    return ["custom1", "custom2"]

@pytest.fixture
def sample_chunks():
    """Return a nested list representing two documents with two chunks each."""
    return [
        ["chunk1-1", "chunk1-2"],
        ["chunk2-1", "chunk2-2"]
    ]

@pytest.fixture
def output_dir(tmp_path):
    """Create and return a temporary directory to store output chunk files."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir

@pytest.fixture
def cutter_for_save():
    """Return a TextCutter instance preloaded with chunks for save() testing."""
    cutter = TextCutter()
    cutter.chunks = [
        ["First chunk.", "Second chunk."],
        ["Third chunk.", "Fourth chunk."]
    ]
    return cutter

# Tests

def test_initialization(cutter):
    """Test TextCutter initialization."""
    assert cutter.chunksize == 1_000_000
    assert cutter.newline is False
    assert cutter.merge_final is False
    assert cutter.delimiter == "_"
    assert cutter.pad == 3

def test_iter(cutter, sample_text):
    """Test class iterator and cut method."""
    cutter.split(sample_text, n=2, merge_threshold=0.0, file=False)
    chunks = cutter.chunks[0]
    assert len(chunks) == 2

def test_iter_method(cutter):
    """Test that __iter__ returns an iterator over the chunk groups."""
    cutter.chunks = [["a", "b"], ["c"]]
    collected_chunks = list(iter(cutter))
    assert collected_chunks == [["a", "b"], ["c"]]

def test_len_method(cutter):
    """Test that __len__ correctly returns the number of source chunk groups."""
    cutter.chunks = [["chunk1"], ["chunk2"], ["chunk3"]]
    assert len(cutter) == 3

def test_process_file(cutter, text_file):
    """Test processing file in chunks."""
    cutter.chunksize = 10
    chunks = cutter._process_file(text_file)
    assert len(chunks) == 3
    assert chunks[0] == "Line1\nLine"
    assert chunks[1] == "2\nLine3\nLi"
    assert chunks[2] == "ne4\nLine5\n"

def test_process_buffer(cutter, sample_text):
    """Test processing buffer in chunks."""
    cutter.chunksize = 10
    chunks = list(cutter._process_buffer(sample_text))
    assert len(chunks) == 3
    assert chunks[0] == "Line1\nLine"
    assert chunks[1] == "2\nLine3\nLi"
    assert chunks[2] == "ne4\nLine5\n"

def test_merge_final_chunks(cutter):
    """Test merging final chunks."""
    chunks = iter(["chunk1", "chunk2", "chunk3"])
    merged_chunks = list(cutter._merge_final_chunks(chunks))
    assert len(merged_chunks) == 2
    assert merged_chunks[0] == "chunk1"
    assert merged_chunks[1] == "chunk2chunk3"

def test_merge_final_single_chunk(cutter):
    """Test merging a generator with one final chunk."""
    chunks = iter(["lastchunk"])
    result = list(cutter._merge_final_chunks(chunks))
    assert result == ["lastchunk"]

def test_write_chunk(cutter, tmp_path):
    """Test writing chunk to output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    chunk = "Test chunk"
    cutter._write_chunk("test.txt", 1, chunk, output_dir)
    output_file = output_dir / "test_001.txt"
    assert output_file.exists()
    assert output_file.read_text() == chunk

def test_split_file(cutter, text_file, tmp_path):
    """Test cutting file into chunks."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cutter.split(sources=text_file, chunksize=10, file=True)
    chunks = cutter.chunks[0]
    assert len(chunks) == 3

def test_split_buffer(cutter, sample_text, tmp_path):
    """Test cutting buffer into chunks."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cutter.split(sources=sample_text, chunksize=10)
    chunks = cutter.chunks[0]
    assert len(chunks) == 3

def test_split_with_newline(cutter, text_file):
    """Test cutting file with newline option."""
    cutter.split(sources=text_file, chunksize=10, newline=True, file=True)
    chunks = list(cutter.chunks)[0]
    assert len(chunks) == 3
    assert chunks[0] == "Line1\nLine2\n"
    assert chunks[1] == "Line3\nLine4\n"
    assert chunks[2] == "Line5\n"

def test_split_with_merge_final(cutter, text_file):
    """Test cutting file with merge final option."""
    cutter.split(sources=text_file, chunksize=9, merge_final=False, file=True)
    chunks = cutter.chunks[0]
    assert len(chunks) == 3
    assert chunks[0] == "Line1\nLin"
    assert chunks[1] == "e2\nLine3\n"
    assert chunks[2] == "Line4\nLine5\n"

def test_split_with_custom_delimiter_pad(cutter, text_file, tmp_path):
    """Test cutting file with custom delimiter and padding."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cutter.split(sources=text_file, file=True, chunksize=10, output_dir=output_dir, delimiter="-", pad=4)
    chunks = list(cutter.chunks)[0]
    assert len(chunks) == 3
    output_files = list(output_dir.glob("*"))
    assert all("-" in file.name for file in output_files)
    assert all(len(file.stem.split("-")[1]) == 4 for file in output_files)

def test_split_sets_custom_path_names(cutter, text_file):
    """Test that split sets custom names when Path objects are passed."""
    cutter.split(sources=text_file, chunksize=10, file=True, names=("custom.txt"))
    assert "custom" in cutter.names[0]

def test_process_buffer_with_n(cutter, sample_text):
    """Test processing buffer with n chunks."""
    cutter.n = 3
    cutter.split(sources=sample_text, n=2, merge_threshold=0.0, file=False)
    chunks = list(cutter.chunks)[0]
    assert len(chunks) == 2
    assert chunks[0] == "Line1\nLine2\nLin"
    assert chunks[1] == "e3\nLine4\nLine5\n"

def test_process_file_with_n(cutter, text_file):
    """Test processing file with n chunks."""
    cutter.split(sources=text_file, n=2, merge_threshold=0.0, file=True)
    chunks = cutter.chunks[0]
    assert len(chunks) == 2
    assert chunks[0] == "Line1\nLine2\nLine3"
    assert chunks[1] == "\nLine4\nLine5\n"

def test_process_file_triggers_early_break(cutter, tmp_path):
    """Test that _process_file triggers early break on empty chunk."""
    test_file = tmp_path / "early_break.txt"
    test_file.write_text("Short\n")
    cutter.n = 10
    cutter.newline = False
    cutter.chunksize = 1000
    chunks = cutter._process_file(test_file, n=True)
    assert isinstance(chunks, list)
    assert len(chunks) < 10

def test_process_buffer_with_n_newline(cutter, text_file):
    """Test cutting file with newline option."""
    cutter.split(sources=text_file, file=True, n=2, merge_threshold=0.0, newline=True)
    chunks = list(cutter.chunks)[0]
    assert len(chunks) == 2
    assert chunks[0] == "Line1\nLine2\nLine3\n"
    assert chunks[1] == "Line4\nLine5\n"

def test_process_buffer_newline_chunk_empty(cutter):
    """Test _process_buffer with newline=True where chunk becomes empty."""
    cutter.n = 2
    cutter.newline = True
    short_text = "Line1\n"
    chunks = cutter._process_buffer(short_text, n=True)
    assert isinstance(chunks, list)
    assert len(chunks) <= 2

def test_split_file_basic(cutter, milestones_text_file, mock_milestones):
    """Test basic file splitting."""
    cutter.split_on_milestones(milestones_text_file, mock_milestones)
    assert len(cutter.chunks) == 1
    assert len(cutter.chunks[0]) == 3
    assert cutter.chunks[0] == ["The", "brown fox jumps", "the lazy dog."]

def test_split_on_milestones_sets_custom_path_names(cutter, milestones_text_file, mock_milestones):
    """Test split_on_milestones with custom Path-style names input."""
    text = milestones_text_file.read_text()
    cutter.split_on_milestones(text, mock_milestones, names=["named.txt"], file=False)
    assert cutter.names[0] == "named"

def test_split_on_milestones_writes_to_disk(cutter, milestones_text_file, mock_milestones, output_dir):
    """Test that split_on_milestones writes output files when output_dir is set."""
    cutter.split_on_milestones(
        sources=milestones_text_file,
        milestones=mock_milestones,
        output_dir=output_dir,
        file=True,
    )
    written_files = list(output_dir.glob("*.txt"))
    assert len(written_files) == 3
    for f in written_files:
        assert f.read_text()

def test_split_buffer_milestones(cutter, sample_milestones_text, mock_milestones):
    """Test splitting string buffer on milestones."""
    cutter.split_on_milestones(sample_milestones_text, mock_milestones, file=False)
    assert len(cutter.chunks) == 1
    assert len(cutter.chunks[0]) == 3

def test_keep_spans_following(cutter, sample_milestones_text, mock_milestones):
    """Test splitting with keep_spans='following'."""
    cutter.split_on_milestones(sample_milestones_text, mock_milestones,
                                  keep_spans="following", file=False)
    assert "quick" in cutter.chunks[0][1]
    assert "over" in cutter.chunks[0][2]

def test_keep_spans_preceding(cutter, sample_milestones_text, mock_milestones):
    """Test splitting with keep_spans='preceding'."""
    cutter.split_on_milestones(sample_milestones_text, mock_milestones,
                                  keep_spans="preceding", file=False)
    chunks = cutter.chunks[0]
    assert "quick" in chunks[0]
    assert "over" in chunks[1]

def test_strip_option(cutter, sample_milestones_text, mock_milestones):
    """Test strip option."""
    cutter.split_on_milestones(sample_milestones_text, mock_milestones,
                                  strip=True, file=False)
    assert all(not chunk.startswith(' ') for chunk in cutter.chunks[0])
    assert all(not chunk.endswith(' ') for chunk in cutter.chunks[0])

def test_file_not_found(cutter, mock_milestones):
    """Test handling of non-existent file."""
    with pytest.raises(LexosException):
        cutter.split_on_milestones("nonexistent.txt", mock_milestones)

def test_multiple_sources(cutter, tmp_path, mock_milestones):
    """Test splitting multiple sources."""
    files = []
    for i in range(2):
        path = tmp_path / f"test{i}.txt"
        path.write_text("Sample text")
        files.append(path)
    cutter.split_on_milestones(files, mock_milestones)
    assert len(cutter.chunks) == 2

def test_custom_milestone_flag(cutter, sample_milestones_text, mock_milestones):
    """Test custom milestone flag."""
    cutter.split_on_milestones(sample_milestones_text, mock_milestones,
                                  milestone_flag="###",
                                  keep_spans="following",
                                  file=False)
    assert len(cutter.chunks[0]) == 3

def test_to_dict_default_names(cutter, sample_chunks):
    """Test dictionary conversion with default source names."""
    cutter.chunks = sample_chunks
    result = cutter.to_dict()
    assert len(result) == 2
    assert "text001" in result
    assert "text002" in result
    assert result["text001"] == sample_chunks[0]
    assert result["text002"] == sample_chunks[1]

def test_to_dict_custom_names(cutter, sample_chunks):
    """Test dictionary conversion with custom source names."""
    cutter.chunks = sample_chunks
    names = ["source1", "source2"]
    result = cutter.to_dict(names=names)

    assert len(result) == 2
    assert "source1" in result
    assert "source2" in result
    assert result["source1"] == sample_chunks[0]

def test_to_dict_path_names(cutter, sample_chunks):
    """Test dictionary conversion with Path source names."""
    cutter.chunks = sample_chunks
    names = [Path("file1.txt"), Path("file2.txt")]
    with pytest.raises(ValidationError):
        _ = cutter.to_dict(names=names)

def test_to_dict_mixed_names(cutter, sample_chunks):
    """Test dictionary conversion with mixed source name types."""
    cutter.chunks = sample_chunks
    names = ["source1", Path("file2.txt")]
    with pytest.raises(ValidationError):
        _ = cutter.to_dict(names=names)

def test_to_dict_empty_chunks(cutter):
    """Test dictionary conversion with empty chunks list."""
    cutter.chunks = []
    result = cutter.to_dict()

    assert len(result) == 0
    assert isinstance(result, dict)

def test_to_dict_single_chunk(cutter):
    """Test dictionary conversion with single chunk."""
    cutter.chunks = [["single_chunk"]]
    result = cutter.to_dict()

    assert len(result) == 1
    assert "text001" in result
    assert result["text001"] == ["single_chunk"]

def test_name_custom_names(cutter, custom_names):
    """Test source name resolution with custom names."""
    cutter.names = custom_names
    result = cutter._get_name("source", 0)
    assert result == "custom1"

def test_name_instance_names(cutter):
    """Test source name resolution with instance names."""
    cutter.names = ["instance1", "instance2"]
    result = cutter._get_name("source", 0)
    assert result == "instance1"

def test_name_default(cutter):
    """Test default source name generation."""
    result = cutter._get_name("source", 1)
    assert result == "text001"

def test_get_name_from_path_object(cutter):
    """Test fallback to Path(source).stem in _get_name()."""
    cutter.names = []
    source = Path("document.txt")
    name = cutter._get_name(source, 0)
    assert name == "document"

def test_save_basic(cutter_for_save, output_dir):
    """Test basic save functionality."""
    cutter_for_save.save(output_dir)
    files = list(output_dir.glob("*.txt"))
    assert len(files) == 4
    assert (output_dir / "doc001_001.txt").exists()

def test_save_custom_names(cutter_for_save, output_dir):
    """Test save with custom names."""
    names = ["custom1", "custom2"]
    cutter_for_save.save(output_dir, names=names)
    assert (output_dir / "custom1_001.txt").exists()
    assert (output_dir / "custom2_001.txt").exists()

def test_save_custom_delimiter_padding(cutter_for_save, output_dir):
    """Test save with custom delimiter and padding."""
    cutter_for_save.save(output_dir, delimiter="-", pad=4)
    assert (output_dir / "doc0001-0001.txt").exists()

def test_save_strip_chunks(cutter_for_save, output_dir):
    """Test save with strip_chunks option."""
    cutter_for_save.chunks = [["  Text with spaces  "]]
    cutter_for_save.save(output_dir, strip_chunks=True)
    content = (output_dir / "doc001_001.txt").read_text()
    assert content == "Text with spaces"

def test_save_no_chunks(cutter_for_save, output_dir):
    """Test error when no chunks to save."""
    cutter_for_save.chunks = []
    with pytest.raises(LexosException, match="No chunks to save."):
        cutter_for_save.save(output_dir)

def test_save_mismatched_names_chunks(cutter_for_save, output_dir):
    """Test error when names length doesn't match chunks length."""
    with pytest.raises(LexosException, match="must equal the number of docs in `chunks`"):
        cutter_for_save.save(output_dir, names=["single_name"])

def test_save_invalid_output_dir(cutter_for_save):
    """Test error with invalid output directory."""
    with pytest.raises(Exception):
        cutter_for_save.save("/invalid/path/here")

def test_merge_basic(cutter):
    """Test basic merge functionality."""
    chunks = ["First chunk.", "Second chunk."]
    result = cutter.merge(chunks)
    assert result == "First chunk. Second chunk."

def test_merge_custom_separator(cutter):
    """Test merge with custom separator."""
    chunks = ["First chunk", "Second chunk"]
    result = cutter.merge(chunks, sep=", ")
    assert result == "First chunk, Second chunk"

def test_merge_empty_chunks(cutter):
    """Test merge with empty chunks list."""
    chunks = []
    with pytest.raises(LexosException, match="No chunks to merge."):
        cutter.merge(chunks)

def test_merge_single_chunk(cutter):
    """Test merge with single chunk."""
    chunks = ["Single chunk"]
    result = cutter.merge(chunks)
    assert result == "Single chunk"

def test_merge_multiple_chunks(cutter):
    """Test merge with multiple chunks."""
    chunks = ["First", "Second", "Third"]
    result = cutter.merge(chunks)
    assert result == "First Second Third"