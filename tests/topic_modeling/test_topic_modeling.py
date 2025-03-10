"""test_mallet.py.

Last Update: 9 March, 2025
"""

from pathlib import Path
from subprocess import CalledProcessError, Popen
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import spacy
from spacy.tokens import Token
from wasabi import Printer

from lexos.exceptions import LexosException
from lexos.topic_modeling.mallet import Mallet

# Fixtures


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Path: Temporary directory path
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def nlp():
    """Create spaCy language model.

    Returns:
        Language: Small English language model
    """
    return spacy.load("en_core_web_sm")


@pytest.fixture
def mallet_instance(tmp_path):
    """Create Mallet instance with test configuration.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Mallet: Configured Mallet instance
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return Mallet(model_dir=str(model_dir))


@pytest.fixture
def setup_token_extension():
    """Set up and tear down spaCy token extension."""
    if not Token.has_extension("is_allowed"):
        Token.set_extension("is_allowed", default=True)
    yield
    if Token.has_extension("is_allowed"):
        Token.remove_extension("is_allowed")


@pytest.fixture(autouse=True)
def cleanup_extensions():
    """Clean up token extensions after each test."""
    yield
    if Token.has_extension("is_allowed"):
        Token.remove_extension("is_allowed")


@pytest.fixture
def sample_text_files(tmp_path):
    """Create sample text files for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Path: Directory containing sample text files
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create sample text files
    files = [
        ("doc1.txt", "Test document one"),
        ("doc2.txt", "Test document two"),
        ("doc3.txt", "Test document three"),
    ]

    for filename, content in files:
        with open(input_dir / filename, "w") as f:
            f.write(content)

    return input_dir


@pytest.fixture
def sample_docs(nlp):
    """Create sample spaCy documents and strings.

    Args:
        nlp: spaCy language model

    Returns:
        list: Mixed list of docs and strings
    """
    docs = [
        nlp("This is document one."),
        "Plain text document two",
        nlp("Document three with stopwords and punctuation!"),
    ]
    return docs


@pytest.fixture
def sample_input_file(tmp_path):
    """Create sample input file for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Path: Path to sample input file
    """
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text content")
    return input_file


@pytest.fixture
def mock_training_options():
    """Create mock training options.

    Returns:
        dict: Mock training options
    """
    return {
        "input": "input.mallet",
        "num-topics": 20,
        "num-iterations": 1000,
        "optimize-interval": 10,
        "random-seed": 42,
        "output-state": "state.gz",
    }


@pytest.fixture
def mock_converted_data():
    """Create mock data for scale_model functions.

    Returns:
        dict: Mock converted MALLET data
    """
    return {
        "topic_term_dists": pd.DataFrame([[0.1, 0.2], [0.3, 0.4]]),
        "doc_topic_dists": pd.DataFrame([[0.5, 0.5], [0.6, 0.4]]),
        "term_frequency": pd.Series([100, 200]),
        "vocab": ["word1", "word2"],
        "doc_lengths": [10, 20],
    }


@pytest.fixture
def mock_topic_coordinates():
    """Create mock topic coordinates DataFrame.

    Returns:
        DataFrame: Mock coordinate data
    """
    return pd.DataFrame(
        {"x": [0.1, 0.2], "y": [0.3, 0.4], "topics": [1, 2], "cluster": [0, 1]}
    )


# Tests


def test_mallet_init_minimal(temp_model_dir):
    """Test minimal Mallet initialization."""
    mallet = Mallet(model_dir=str(temp_model_dir))

    assert mallet.model_dir == str(temp_model_dir).replace("\\", "/")
    assert mallet.data_file == f"{mallet.model_dir}/data.txt"
    assert isinstance(mallet.msg, Printer)


def test_mallet_init_full(temp_model_dir):
    """Test full Mallet initialization with all parameters."""
    params = {
        "model_dir": str(temp_model_dir),
        "data_file": "test.txt",
        "mallet_import_file": "import.mallet",
        "mallet_bin": "mallet/bin",
    }

    mallet = Mallet(**params)

    for key, value in params.items():
        assert getattr(mallet, key) == Path(value).as_posix()


def test_mallet_init_no_model_dir():
    """Test Mallet initialization without model_dir."""
    with pytest.raises(ValueError):
        Mallet()


def test_mallet_init_posix_conversion(temp_model_dir):
    """Test path conversion to POSIX format."""
    windows_path = "C:\\test\\path"
    mallet = Mallet(model_dir=str(temp_model_dir), mallet_bin=windows_path)

    assert "\\" not in mallet.mallet_bin
    assert "/" in mallet.mallet_bin


def test_mallet_printer_initialization(temp_model_dir):
    """Test message printer initialization."""
    mallet = Mallet(model_dir=str(temp_model_dir))

    assert hasattr(mallet, "msg")
    assert isinstance(mallet.msg, Printer)


@pytest.mark.parametrize(
    "model_dir", ["relative/path", "/absolute/path", "C:\\windows\\path"]
)
def test_mallet_path_handling(model_dir, tmp_path):
    """Test handling of different path formats.

    Args:
        model_dir: Path to test
        tmp_path: Pytest temporary directory
    """
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    mallet = Mallet(model_dir=str(test_dir))

    assert "/" in mallet.model_dir
    assert "\\" not in mallet.model_dir
    assert mallet.data_file.endswith("/data.txt")


def test_basic_token_bag(mallet_instance, nlp, setup_token_extension):
    """Test basic token bag generation."""
    doc = nlp("This is a test document")
    result = mallet_instance._get_token_bag(doc)
    assert isinstance(result, str)
    assert "test" in result
    assert "document" in result


def test_token_bag_lemmatization(mallet_instance, nlp, setup_token_extension):
    """Test token bag with lemmatization."""
    doc = nlp("running runs ran")
    result = mallet_instance._get_token_bag(doc, use_lemmas=True)
    assert "run" in result
    assert "running" not in result
    assert "runs" not in result
    assert "ran" not in result


def test_token_bag_stop_words(mallet_instance, nlp, setup_token_extension):
    """Test token bag with stop word removal."""
    doc = nlp("this is a test")
    result = mallet_instance._get_token_bag(doc, remove_stops=True)
    assert "test" in result
    assert "is" not in result
    assert "this" not in result


def test_token_bag_punctuation(mallet_instance, nlp, setup_token_extension):
    """Test token bag with punctuation removal."""
    doc = nlp("Hello, world! This is a test.")
    result = mallet_instance._get_token_bag(doc, remove_punct=True)
    assert "," not in result
    assert "!" not in result
    assert "." not in result
    assert "Hello" in result


def test_token_bag_frequency(mallet_instance, nlp, setup_token_extension):
    """Test token frequency preservation in bag."""
    doc = nlp("test test test other")
    result = mallet_instance._get_token_bag(doc)
    assert result.count("test") == 3
    assert result.count("other") == 1


def test_token_bag_with_span(mallet_instance, nlp, setup_token_extension):
    """Test token bag generation from spaCy Span."""
    doc = nlp("This is a test document")
    span = doc[3:5]  # "test document"
    result = mallet_instance._get_token_bag(span)
    assert "test" in result
    assert "document" in result
    assert "is" not in result


def test_token_bag_allowed_tokens(mallet_instance, nlp):
    """Test token bag with allowed token filtering."""
    doc = nlp("This is a test document")
    Token.set_extension("is_allowed", default=False, force=True)
    doc[3]._.is_allowed = True  # Only allow "test"

    result = mallet_instance._get_token_bag(doc)
    assert result == "test"
    Token.remove_extension("is_allowed")


@pytest.mark.parametrize(
    "options",
    [
        {"use_lemmas": True, "remove_stops": True},
        {"use_lemmas": True, "remove_punct": True},
        {"remove_stops": True, "remove_punct": True},
        {"use_lemmas": True, "remove_stops": True, "remove_punct": True},
    ],
)
def test_token_bag_combined_options(
    mallet_instance, nlp, setup_token_extension, options
):
    """Test token bag with combinations of options.

    Args:
        options: Dictionary of token bag options to test
    """
    doc = nlp("Running, the test! It is running.")
    result = mallet_instance._get_token_bag(doc, **options)

    if options.get("use_lemmas"):
        assert "running" not in result
        assert "run" in result

    if options.get("remove_stops"):
        assert "it" not in result
        assert "is" not in result

    if options.get("remove_punct"):
        assert "," not in result
        assert "!" not in result


def test_import_data_basic(tmp_path, capsys):
    """Test basic data import with minimal parameters."""
    model = Mallet(model_dir=str(tmp_path))

    # Create a small data file
    data = ["This is a test document", "This is another test document"]
    with open(f"{model.model_dir}/data.txt", "w") as file:
        file.write("\n".join(data))

    # Run import
    model._import_data(
        input=f"{model.model_dir}/data.txt", output=f"{model.model_dir}/import.mallet"
    )

    captured = capsys.readouterr().out
    assert "import-file" in captured
    assert "--input" in captured
    assert "--output" in captured
    assert "Import complete" in captured
    # Check that the import.mallet file has been created
    assert Path(f"{model.model_dir}/import.mallet").exists()


def test_import_data_custom_options(tmp_path, capsys):
    """Test import with custom options."""
    model = Mallet(model_dir=str(tmp_path))

    # Create a small data file
    data = ["This is a test document", "This is another test document"]
    with open(f"{model.model_dir}/data.txt", "w") as file:
        file.write("\n".join(data))

    # Run import
    model._import_data(
        input=f"{model.model_dir}/data.txt",
        output=f"{model.model_dir}/import.mallet",
        keep_sequence=True,
        preserve_case=True,
        token_regex="custom_regex",
    )

    captured = capsys.readouterr().out
    assert "--keep-sequence True" in captured
    assert "--preserve-case True" in captured
    assert "--token-regex 'custom_regex'" in captured


def test_import_data_missing_input(tmp_path, capsys):
    """Test error handling for missing input file."""
    model = Mallet(model_dir=str(tmp_path))
    model.data_file = None

    # Run import
    model._import_data(input=None, output=f"{model.model_dir}/import.mallet")

    captured = capsys.readouterr().out
    assert "Please provide an input file." in captured


def test_import_data_default_output(tmp_path):
    """Test default output file path generation."""
    model = Mallet(model_dir=str(tmp_path))

    # Create a small data file
    data = ["This is a test document", "This is another test document"]
    with open(f"{model.model_dir}/data.txt", "w") as file:
        file.write("\n".join(data))

    model._import_data(input=f"{model.model_dir}/data.txt", output=None)
    assert model.mallet_import_file == f"{model.model_dir}/import.mallet"


@pytest.mark.skip(
    reason="Disabled because it is hard to trigger a CallledProcessError from check_output."
)
def test_import_data_command_error(tmp_path):
    """Test error handling for MALLET command failure."""
    model = Mallet(model_dir=str(tmp_path))

    # Create a small data file
    data = ["This is a test document", "This is another test document"]
    with open(f"{model.model_dir}/data.txt", "w") as file:
        file.write("\n".join(data))

    # Run a bad import command
    with pytest.raises(CalledProcessError):
        model.mallet_bin = "invalid_mallet_bin"
        model._import_data(
            input=f"{model.model_dir}/data.txt",
            output=f"{model.model_dir}/import.mallet",
        )


def test_import_data_path_conversion(tmp_path, capsys):
    """Test path conversion to POSIX format."""
    model = Mallet(model_dir=str(tmp_path))

    # Create a small data file
    data = ["This is a test document", "This is another test document"]
    with open(f"{model.model_dir}/data.txt", "w") as file:
        file.write("\n".join(data))

    model_dir = model.model_dir.replace("/", "\\")
    output_dir = f"{model.model_dir}/import.mallet".replace("/", "\\")
    model._import_data(input=f"{model_dir}\\data.txt", output=output_dir)

    captured = capsys.readouterr().out
    assert "import-file" in captured
    assert "--input" in captured
    assert "--output" in captured
    assert "Import complete" in captured
    # Check that the import.mallet file has been created
    assert Path(f"{model.model_dir}/import.mallet").exists()


@pytest.mark.parametrize(
    "token_regex",
    [
        r"\p{L}[\p{L}\p{P}]+\p{L}",
        r"[A-Za-z]+",
        r"\w+",
    ],
)
def test_import_data_token_regex(tmp_path, capsys, token_regex):
    """Test different token regex patterns.

    Args:
        token_regex: Token regex pattern to test
    """
    model = Mallet(model_dir=str(tmp_path))

    # Create a small data file
    data = ["This is a test document", "This is another test document"]
    with open(f"{model.model_dir}/data.txt", "w") as file:
        file.write("\n".join(data))

    model._import_data(
        input=f"{model.model_dir}/data.txt",
        output=f"{model.model_dir}/import.mallet",
        token_regex=token_regex,
    )

    captured = capsys.readouterr().out
    print(captured)
    assert token_regex in captured
    # Check that the import.mallet file has been created
    assert Path(f"{model.model_dir}/import.mallet").exists()


def test_set_training_options_basic(mallet_instance):
    """Test basic training options setup."""
    options = mallet_instance._set_training_options(
        import_file="test.mallet",
        num_topics=20,
        num_iterations=1000,
        optimize_interval=10,
        random_seed=42,
    )

    assert options["input"] == "test.mallet"
    assert options["num-topics"] == 20
    assert options["num-iterations"] == 1000
    assert options["optimize-interval"] == 10
    assert options["random-seed"] == 42


def test_set_training_options_output_paths(mallet_instance):
    """Test output file path generation."""
    options = mallet_instance._set_training_options(
        import_file="test.mallet",
        num_topics=20,
        num_iterations=1000,
        optimize_interval=10,
        random_seed=42,
    )

    assert options["output-state"].endswith("/state.gz")
    assert options["output-topic-keys"].endswith("/keys.txt")
    assert options["output-doc-topics"].endswith("/composition.txt")
    assert options["word-topic-counts-file"].endswith("/counts.txt")
    assert options["output-topic-docs"].endswith("/topic-docs.txt")
    assert options["diagnostics-file"].endswith("/diagnostics.xml")


def test_set_training_options_custom_kwargs(mallet_instance):
    """Test handling of additional keyword arguments."""
    custom_opts = {
        "alpha": 0.5,
        "beta": 0.1,
        "num_threads": 4,
    }

    options = mallet_instance._set_training_options(
        import_file="test.mallet",
        num_topics=20,
        num_iterations=1000,
        optimize_interval=10,
        random_seed=42,
        **custom_opts,
    )

    assert options["alpha"] == 0.5
    assert options["beta"] == 0.1
    assert options["num-threads"] == 4


def test_set_training_options_path_conversion(mallet_instance):
    """Test path conversion for file paths in options."""
    windows_paths = {
        "output-state": "C:\\test\\state.gz",
        "output-topic-keys": "C:\\test\\keys.txt",
    }

    options = mallet_instance._set_training_options(
        import_file="test.mallet",
        num_topics=20,
        num_iterations=1000,
        optimize_interval=10,
        random_seed=42,
        **windows_paths,
    )

    assert "\\" not in options["output-state"]
    assert "\\" not in options["output-topic-keys"]
    assert "/" in options["output-state"]
    assert "/" in options["output-topic-keys"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("diagnostics-file", "test.xml"),
        ("input", "test.mallet"),
        ("output-doc-topics", "topics.txt"),
        ("output-state", "state.gz"),
        ("output-topic-docs", "docs.txt"),
        ("output-topic-keys", "keys.txt"),
        ("word-topic-counts-file", "counts.txt"),
    ],
)
def test_set_training_options_path_keys(mallet_instance, key, value):
    """Test path conversion for specific option keys.

    Args:
        key: Option key to test
        value: Option value to test
    """
    options = mallet_instance._set_training_options(
        import_file="test.mallet",
        num_topics=20,
        num_iterations=1000,
        optimize_interval=10,
        random_seed=42,
        **{key: value},
    )

    assert Path(options[key]).as_posix() == value


@pytest.fixture
def mock_process():
    """Create mock process with simulated MALLET output.

    Returns:
        Mock: Mocked process with configured stdout
    """
    process = Mock(spec=Popen)
    process.stdout = Mock()
    process.stdout.readline.side_effect = [
        b"<100> LL/topic: -8.23456\n",
        b"<200> LL/topic: -7.34567\n",
        b"<300> LL/topic: -6.45678\n",
        b"",
    ]
    process.poll.side_effect = [None, None, None, -1]
    return process


@pytest.mark.skip(
    reason="Disabled because I can't figure out how to mock a subprocess."
)
def test_track_progress_basic(mallet_instance, mock_process, capsys):
    """Test basic progress tracking functionality."""
    with patch("subprocess.Popen", return_value=mock_process):
        mallet_instance._track_progress("test_cmd", 1000)
        captured = capsys.readouterr()

        # Verify output contains expected progress messages
        assert "LL/topic: -8.23456" in captured.out
        assert "Modeling progress: 10%" in captured.out
        assert "Modeling progress: 20%" in captured.out
        assert "Modeling progress: 30%" in captured.out


@pytest.mark.skip(
    reason="Disabled because I can't figure out how to mock a subprocess."
)
def test_track_progress_ll_tracking(mallet_instance):
    """Test log likelihood value tracking."""
    process = Mock(spec=Popen)
    process.stdout = Mock()
    process.stdout.readline.side_effect = [
        b"<100> LL/topic: -8.23456\n",
        b"<200> LL/topic: -7.34567\n",
        b"",
    ]
    process.poll.side_effect = [None, None, -1]

    with patch("subprocess.Popen", return_value=process):
        mallet_instance._track_progress("test_cmd", 1000)
        assert hasattr(mallet_instance, "ll")


@pytest.mark.skip(
    reason="Disabled because I can't figure out how to mock a subprocess."
)
def test_track_progress_invalid_output(mallet_instance):
    """Test handling of invalid MALLET output."""
    process = Mock(spec=Popen)
    process.stdout = Mock()
    process.stdout.readline.side_effect = [
        b"Invalid line\n",
        b"Another invalid line\n",
        b"",
    ]
    process.poll.side_effect = [None, None, -1]

    with patch("subprocess.Popen", return_value=process):
        # Should not raise exceptions for invalid output
        mallet_instance._track_progress("test_cmd", 1000)


@pytest.mark.skip(
    reason="Disabled because I can't figure out how to mock a subprocess."
)
@pytest.mark.parametrize(
    "iterations,expected_progress",
    [(100, [10, 20, 30]), (1000, [10, 20, 30]), (2000, [5, 10, 15])],
)
def test_track_progress_different_iterations(
    mallet_instance, iterations, expected_progress
):
    """Test progress calculation with different iteration counts.

    Args:
        iterations: Number of iterations to test
        expected_progress: Expected progress percentages
    """
    process = Mock(spec=Popen)
    process.stdout = Mock()
    process.stdout.readline.side_effect = [
        b"<100> LL/topic: -8.23456\n",
        b"<200> LL/topic: -7.34567\n",
        b"<300> LL/topic: -6.45678\n",
        b"",
    ]
    process.poll.side_effect = [None, None, None, -1]

    with patch("subprocess.Popen", return_value=process):
        with patch("builtins.print") as mock_print:
            mallet_instance._track_progress("test_cmd", iterations)

            progress_calls = [
                call
                for call in mock_print.call_args_list
                if "Modeling progress" in str(call)
            ]
            for i, progress in enumerate(expected_progress):
                assert f"{progress}%" in str(progress_calls[i])


def test_import_dir_basic(tmp_path, capsys):
    """Test basic directory import functionality."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create sample text files
    files = [
        ("doc1.txt", "Test document one"),
        ("doc2.txt", "Test document two"),
        ("doc3.txt", "Test document three"),
    ]

    for filename, content in files:
        with open(input_dir / filename, "w") as f:
            f.write(content)

    # Create model and import files
    model = Mallet(model_dir=str(tmp_path))
    model.import_dir(
        input=str(input_dir),
        output=f"{model.model_dir}/import.mallet",
    )
    # Check that the data file contains the content
    with open(model.data_file) as f:
        content = f.read()
        assert "Test document one" in content
        assert "Test document two" in content
        assert "Test document three" in content


def test_import_dir_nonexistent(tmp_path):
    """Test handling of nonexistent directory."""
    model = Mallet(model_dir=str(tmp_path))
    with pytest.raises(LexosException):
        model.import_dir(input="nonexistent_dir")


def test_import_dir_empty(tmp_path):
    """Test handling of empty directory."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    model = Mallet(model_dir=str(tmp_path))
    with pytest.raises(LexosException):
        model.import_dir(input=str(input_dir))


def test_import_dir_custom_output(mallet_instance, sample_text_files):
    """Test import with custom output path."""
    output_file = "custom_output.mallet"

    with patch.object(mallet_instance, "_import_data") as mock_import:
        mallet_instance.import_dir(str(sample_text_files), output=output_file)

        mock_import.assert_called_with(
            mallet_instance.data_file,
            output_file,
            False,
            False,
            r"\p{L}[\p{L}\p{P}]+\p{L}",
        )


@pytest.mark.parametrize(
    "params",
    [
        {"keep_sequence": True},
        # {"preserve_case": True},
        # {"token_regex": r"[A-Za-z]+"},
        # {"keep_sequence": True, "preserve_case": True, "token_regex": r"[A-Za-z]+"}
    ],
)
def test_import_dir_options(tmp_path, params, capsys):
    """Test import with different option combinations.

    Args:
        params: Dictionary of import options to test
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create sample text files
    files = [
        ("doc1.txt", "Test document one"),
        ("doc2.txt", "Test document two"),
        ("doc3.txt", "Test document three"),
    ]

    for filename, content in files:
        with open(input_dir / filename, "w") as f:
            f.write(content)

    # Create model and import files
    model = Mallet(model_dir=str(tmp_path))
    model.import_dir(
        input=str(input_dir), output=f"{model.model_dir}/import.mallet", **params
    )

    captured = capsys.readouterr().out
    for k, v in params.items():
        k = k.replace("_", "-")
        assert f"--{k} {v}" in captured


def test_import_docs_basic(mallet_instance, sample_docs):
    """Test basic document import functionality."""
    with patch.object(mallet_instance, "_import_data"):
        mallet_instance.import_docs(sample_docs, output="output.mallet")

        # Verify data file creation and content
        assert Path(mallet_instance.data_file).exists()
        with open(mallet_instance.data_file) as f:
            content = f.readlines()
            assert len(content) == len(sample_docs)


def test_import_docs_string_handling(mallet_instance):
    """Test handling of string documents."""
    string_docs = ["Doc 1", "Doc 2", "Doc 3"]

    with patch.object(mallet_instance, "_import_data"):
        mallet_instance.import_docs(string_docs, output="output.mallet")

        with open(mallet_instance.data_file) as f:
            content = f.readlines()
            assert all(
                doc.strip() in line.strip() for doc, line in zip(string_docs, content)
            )


def test_import_docs_with_pos_filtering(mallet_instance, nlp):
    """Test document import with POS tag filtering."""
    docs = [nlp("The quick brown fox jumps.")]
    allowed_pos = ["NOUN", "VERB"]

    with patch.object(mallet_instance, "_import_data"):
        mallet_instance.import_docs(docs, output="output.mallet", allowed=allowed_pos)

        with open(mallet_instance.data_file) as f:
            content = f.read()
            assert "fox" in content  # NOUN
            assert "jumps" in content  # VERB
            assert "quick" not in content  # ADJ


@pytest.mark.parametrize(
    "options",
    [
        {"use_lemmas": True},
        {"remove_stops": True},
        {"remove_punct": True},
        {"use_lemmas": True, "remove_stops": True, "remove_punct": True},
    ],
)
def test_import_docs_processing_options(mallet_instance, nlp, options):
    """Test document import with different processing options.

    Args:
        options: Dictionary of processing options to test
    """
    docs = [nlp("Running quickly, the cats are jumping!")]

    with patch.object(mallet_instance, "_import_data"):
        mallet_instance.import_docs(docs, output="output.mallet", **options)

        with open(mallet_instance.data_file) as f:
            content = f.read()

            if options.get("use_lemmas"):
                assert "running" not in content
                assert "run" in content

            if options.get("remove_stops"):
                assert "the" not in content
                assert "are" not in content

            if options.get("remove_punct"):
                assert "," not in content
                assert "!" not in content


def test_import_docs_file_handling(mallet_instance, sample_docs):
    """Test handling of existing files."""
    # Create existing files
    Path(mallet_instance.data_file).parent.mkdir(exist_ok=True)
    Path(mallet_instance.data_file).touch()
    output_file = "output.mallet"
    Path(output_file).touch()

    with patch.object(mallet_instance, "_import_data"):
        mallet_instance.import_docs(sample_docs, output=output_file)

        # Verify files were handled correctly
        assert Path(mallet_instance.data_file).exists()
        assert not Path(output_file).exists()


def test_import_file_basic(tmp_path):
    """Test basic file import functionality."""
    model = Mallet(model_dir=str(tmp_path))
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text content")
    output_file = f"{model.model_dir}/import.mallet"

    with patch.object(model, "_import_data") as mock_import:
        model.import_file(
            input=str(input_file),
            output=output_file,
            keep_sequence=False,
            preserve_case=False,
        )

        mock_import.assert_called_once_with(
            str(input_file).replace("\\", "/"),
            output_file.replace("\\", "/"),
            False,
            False,
            r"\p{L}[\p{L}\p{P}]+\p{L}",
        )


def test_import_file_nonexistent(tmp_path, capsys):
    """Test handling of nonexistent input file."""
    model = Mallet(model_dir=str(tmp_path))
    output_file = f"{model.model_dir}/import.mallet"

    with patch("wasabi.Printer"):
        model.import_file(
            input="nonexistent.txt",
            output=output_file,
            keep_sequence=False,
            preserve_case=False,
        )
        captured = capsys.readouterr().out
        assert "File not found" in captured


def test_import_file_path_conversion(tmp_path, capsys):
    """Test path conversion to POSIX format."""
    model = Mallet(model_dir=str(tmp_path))
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text content")
    output_file = f"{model.model_dir}/import.mallet"

    windows_paths = {
        "input": str(input_file).replace("/", "\\"),
        "output": output_file.replace("/", "\\"),
    }

    with patch.object(model, "_import_data") as mock_import:
        model.import_file(**windows_paths, keep_sequence=False, preserve_case=False)

        input_path, output_path = mock_import.call_args[0][:2]
        assert "\\" not in input_path
        assert "\\" not in output_path
        assert "/" in input_path
        assert "/" in output_path


@pytest.mark.parametrize(
    "options",
    [
        {"keep_sequence": True, "preserve_case": False},
        {"keep_sequence": False, "preserve_case": True},
        {"keep_sequence": True, "preserve_case": True},
    ],
)
def test_import_file_options(mallet_instance, sample_input_file, options):
    """Test import with different options combinations.

    Args:
        options: Dictionary of import options to test
    """
    with patch.object(mallet_instance, "_import_data") as mock_import:
        mallet_instance.import_file(
            input=str(sample_input_file), output="output.mallet", **options
        )

        _, _, keep_sequence, preserve_case, _ = mock_import.call_args[0]
        assert keep_sequence == options["keep_sequence"]
        assert preserve_case == options["preserve_case"]


def test_import_file_custom_regex(mallet_instance, sample_input_file):
    """Test import with custom token regex pattern."""
    custom_regex = r"[A-Za-z]+"

    with patch.object(mallet_instance, "_import_data") as mock_import:
        mallet_instance.import_file(
            input=str(sample_input_file),
            output="output.mallet",
            keep_sequence=False,
            preserve_case=False,
            token_regex=custom_regex,
        )

        assert mock_import.call_args[0][4] == custom_regex


def test_train_basic(tmp_path, capsys):
    """Test basic model training with default parameters."""
    model = Mallet(model_dir=str(tmp_path))
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text content")
    output_file = f"{model.model_dir}/import.mallet"

    model.import_file(
        input=input_file, output=output_file, keep_sequence=False, preserve_case=False
    )
    model.train(verbose=False)

    captured = capsys.readouterr().out
    assert "Model trained successfully" in captured


def test_train_missing_import_file(tmp_path, capsys):
    """Test training without import file."""
    model = Mallet(model_dir=str(tmp_path))
    model.train(verbose=False)
    captured = capsys.readouterr().out
    assert "Please provide a `mallet_import_file`." in captured


def test_train_verbose_mode(mallet_instance):
    """Test training with verbose output."""
    mallet_instance.mallet_import_file = "test.mallet"

    with patch.object(mallet_instance, "_set_training_options") as mock_options:
        with patch.object(mallet_instance, "_track_progress") as mock_track:
            mock_options.return_value = {}
            mallet_instance.train(verbose=True)

            assert mock_track.called


def test_train_custom_parameters(mallet_instance):
    """Test training with custom parameters."""
    params = {
        "num_topics": 30,
        "num_iterations": 2000,
        "optimize_interval": 20,
        "random_seed": 42,
        "alpha": 0.1,
        "beta": 0.01,
    }

    with patch.object(mallet_instance, "_set_training_options") as mock_options:
        with patch("subprocess.check_output"):
            mallet_instance.train(mallet_import_file="test.mallet", **params)

            call_args = mock_options.call_args[0]
            assert call_args[1] == 30
            assert call_args[2] == 2000
            assert call_args[3] == 20
            assert call_args[4] == 42


@pytest.mark.skip(
    reason="Disabled because it is hard to trigger a CallledProcessError from check_output."
)
def test_train_command_error(mallet_instance):
    """Test error handling during training."""
    mallet_instance.mallet_import_file = "test.mallet"
    error_msg = "MALLET training failed"

    with patch.object(mallet_instance, "_set_training_options") as mock_options:
        with patch("subprocess.check_output") as mock_check:
            mock_options.return_value = {}
            mock_check.side_effect = CalledProcessError(1, "cmd", error_msg)

            with patch("wasabi.Printer") as mock_printer:
                mallet_instance.train(verbose=False)
                mock_printer.return_value.fail.assert_called()


@pytest.mark.parametrize(
    "verbose,expected_method", [(True, "_track_progress"), (False, "check_output")]
)
def test_train_output_modes(mallet_instance, verbose, expected_method, capsys):
    """Test different output modes.

    Args:
        verbose: Whether to use verbose output
        expected_method: Expected method to be called
    """
    mallet_instance.mallet_import_file = "test.mallet"

    with patch.object(mallet_instance, "_set_training_options") as mock_options:
        with patch.object(mallet_instance, "_track_progress") as mock_track:
            with patch("subprocess.check_output"):
                mock_options.return_value = {}
                mallet_instance.train(verbose=verbose)

                if expected_method == "_track_progress":
                    assert mock_track.called
                else:
                    captured = capsys.readouterr().out
                    assert "Running" in captured


def test_scale_default_output(
    mallet_instance, mock_converted_data, mock_topic_coordinates
):
    """Test scaling with default output path."""
    with patch(
        "lexos.topic_modeling.mallet.scale_model.convert_mallet_data"
    ) as mock_convert:
        with patch(
            "lexos.topic_modeling.mallet.scale_model.get_topic_coordinates"
        ) as mock_coords:
            mock_convert.return_value = mock_converted_data
            mock_coords.return_value = mock_topic_coordinates

            state_file = "test_state.gz"
            mallet_instance.scale(model_state_file=state_file)

            expected_output = Path(mallet_instance.model_dir) / "topic_scaled.csv"
            assert Path(expected_output).exists()
            mock_convert.assert_called_once_with(state_file)


def test_scale_custom_output(
    mallet_instance, mock_converted_data, mock_topic_coordinates
):
    """Test scaling with custom output path."""
    with patch(
        "lexos.topic_modeling.mallet.scale_model.convert_mallet_data"
    ) as mock_convert:
        with patch(
            "lexos.topic_modeling.mallet.scale_model.get_topic_coordinates"
        ) as mock_coords:
            mock_convert.return_value = mock_converted_data
            mock_coords.return_value = mock_topic_coordinates

            output = "custom_output.csv"
            mallet_instance.scale(model_state_file="test_state.gz", output=output)

            assert Path(output).exists()


def test_scale_conversion_error(mallet_instance):
    """Test error handling during data conversion."""
    with patch(
        "lexos.topic_modeling.mallet.scale_model.convert_mallet_data"
    ) as mock_convert:
        mock_convert.side_effect = ValueError("Conversion failed")

        with pytest.raises(LexosException) as exc_info:
            mallet_instance.scale(model_state_file="test_state.gz")

        assert "Failed!: Conversion failed" in str(exc_info.value)


def test_scale_coordinate_error(mallet_instance, mock_converted_data):
    """Test error handling during coordinate calculation."""
    with patch(
        "lexos.topic_modeling.mallet.scale_model.convert_mallet_data"
    ) as mock_convert:
        with patch(
            "lexos.topic_modeling.mallet.scale_model.get_topic_coordinates"
        ) as mock_coords:
            mock_convert.return_value = mock_converted_data
            mock_coords.side_effect = ValueError("Coordinate calculation failed")

            with pytest.raises(LexosException) as exc_info:
                mallet_instance.scale(model_state_file="test_state.gz")

            assert "Failed!: Coordinate calculation failed" in str(exc_info.value)


def test_scale_output_format(
    mallet_instance, mock_converted_data, mock_topic_coordinates
):
    """Test output file format and content."""
    with patch(
        "lexos.topic_modeling.mallet.scale_model.convert_mallet_data"
    ) as mock_convert:
        with patch(
            "lexos.topic_modeling.mallet.scale_model.get_topic_coordinates"
        ) as mock_coords:
            mock_convert.return_value = mock_converted_data
            mock_coords.return_value = mock_topic_coordinates

            output = "test_output.csv"
            mallet_instance.scale(model_state_file="test_state.gz", output=output)

            # Verify CSV content and format
            result = pd.read_csv(output, header=None)
            result.columns = (
                mock_topic_coordinates.columns.tolist()
            )  # ['x', 'y', 'topics', 'cluster']
            print(mock_topic_coordinates)
            # assert False
            pd.testing.assert_frame_equal(
                result, mock_topic_coordinates, check_names=False
            )
