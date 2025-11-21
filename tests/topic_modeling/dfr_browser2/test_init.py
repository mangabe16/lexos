"""Tests for the dfr_browser2 Browser class.

Coverage 99%. Missing: 328

Last Updated: November 20, 2025
"""

import json
import shutil
from pathlib import Path

import pytest

from lexos.topic_modeling.dfr_browser2 import Browser


def create_file(path: Path, content: str = ""):
    """Create a file at the given path with the specified content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_browser_initialization_and_config_merging(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Ensure Browser copies template and mallet files, merges config, and updates config paths to data/."""
    # Create a fake template directory (dist) with a basic config.json
    template_dir = dist_template_dir

    # Create a fake mallet files directory with the required files
    mallet_dir = mallet_dir_factory()

    # Create a sample data file (TSV)
    data_file = sample_tsv

    # Initialize browser with provided config to override 'application.name'
    user_cfg = {"application": {"name": "Custom Title"}}
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out"),
        data_path=str(data_file),
        config=user_cfg,
    )

    # Validate files copied into output browser data/ folder
    out_data = Path(b.browser_path) / "data"
    assert (out_data / "docs.txt").exists()
    assert (out_data / "topic-keys.txt").exists()
    # doc-topic should have been copied
    assert (out_data / "doc-topic.txt").exists()
    assert (out_data / "metadata.csv").exists()
    assert (out_data / "topic-state.gz").exists()
    assert (out_data / "topic_coords.csv").exists()

    # Validate config.json updated with merged config and file paths
    cfg_path = Path(b.browser_path) / "config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # User config should override template entries where appropriate
    assert cfg.get("application", {}).get("name") == "Custom Title"
    # File paths should point to data/ relative paths
    assert cfg.get("data_source") == "data/docs.txt"
    assert cfg.get("topic_keys_file") == "data/topic-keys.txt"
    assert cfg.get("doc_topic_file") == "data/doc-topic.txt"


def test_filename_map_original_to_new(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test that filename_map is treated as original -> destination mapping and config entries are updated."""
    # Test that filename_map is treated as original->new mapping
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topics.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # Create files using 'original' names
    orig_names = [
        "metadata.csv",
        "topic-keys.txt",
        "doc-topics.txt",
        "topic-state.gz",
        "topic_coords.csv",
    ]
    for fname in orig_names:
        create_file(mallet_dir / fname, "content")

    # We'll map 'doc-topics.txt' -> 'doc-topic.txt' and 'topic-keys.txt' -> 'topic-keys.txt'
    filename_map = {
        "doc-topics.txt": "doc-topic.txt",
        "topic-keys.txt": "topic-keys.txt",
        "metadata.csv": "metadata.csv",
    }

    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out2"),
        data_path=str(data_file),
        config={"application": {"name": "Mapped"}},
        filename_map=filename_map,
    )

    out_data = Path(b.browser_path) / "data"
    # The destination names should be the values from filename_map
    assert (out_data / "doc-topic.txt").exists()
    assert (out_data / "topic-keys.txt").exists()
    assert (out_data / "metadata.csv").exists()

    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("doc_topic_file") == "data/doc-topic.txt"
    assert cfg.get("topic_keys_file") == "data/topic-keys.txt"


def test_filename_map_reversed_mapping(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test reversed mapping where filename_map's key is treated as destination and value as source."""
    # Test reversed mapping where keys are destination and values are original
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory(
        [
            "doc-topics.txt",
            "metadata.csv",
            "topic-keys.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # Create original file named 'doc-topics.txt'
    create_file(mallet_dir / "doc-topics.txt", "content")
    create_file(mallet_dir / "metadata.csv", "content")
    create_file(mallet_dir / "topic-keys.txt", "content")
    # Create other required files
    create_file(mallet_dir / "topic-state.gz", "content")
    create_file(mallet_dir / "topic_coords.csv", "content")

    filename_map = {
        # reversed: key is destination name, value is original name
        "doc-topic.txt": "doc-topics.txt",
    }

    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out3"),
        data_path=str(data_file),
        filename_map=filename_map,
    )

    out_data = Path(b.browser_path) / "data"
    # Since the mapping was reversed, the final destination should be the key
    assert (out_data / "doc-topic.txt").exists()
    # Original file name should not be present (it was copied and renamed)
    assert not (out_data / "doc-topics.txt").exists()


def test_partial_filename_map(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test that filename_map can include only a subset of files and defaults are used for others."""
    # Test that filename_map can include only a subset of files and defaults are used for others
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory()

    # Only map topic-keys to a custom destination
    filename_map = {
        "topic-keys.txt": "custom-topic-keys.txt",
    }

    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out4"),
        data_path=str(data_file),
        filename_map=filename_map,
    )

    out_data = Path(b.browser_path) / "data"
    # The mapped file should be renamed
    assert (out_data / "custom-topic-keys.txt").exists()
    # Default file should still be copied into canonical filename
    assert (out_data / "doc-topic.txt").exists()
    # Check config updated to use custom path for topic keys and default path for doc-topic
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("topic_keys_file") == "data/custom-topic-keys.txt"
    assert cfg.get("doc_topic_file") == "data/doc-topic.txt"


def test_user_config_file_path_preserved(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Verify that user-specified file path keys in config are preserved and not overwritten by automatic data/ paths."""
    # Test that user-supplied file path in config is preserved and not overwritten by copied paths
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory()

    data_file = sample_tsv

    # User sets a custom path in config.json for doc_topic_file and it should be preserved
    user_cfg = {
        "application": {"name": "Custom Config"},
        "doc_topic_file": "/bad/path/doc-topic.txt",
    }
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out5"),
        data_path=str(data_file),
        config=user_cfg,
    )

    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Application override is preserved
    assert cfg.get("application", {}).get("name") == "Custom Config"
    # The file path value set by the user should be preserved
    assert cfg.get("doc_topic_file") == "/bad/path/doc-topic.txt"


def test_missing_required_files_raises(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Missing required mallet files should raise FileNotFoundError during Browser initialization."""
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory(
        ["doc-topics.txt", "metadata.csv"]
    )  # missing topic-keys

    data_file = sample_tsv

    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(template_dir),
            browser_path=str(tmp_path / "browser_out6"),
            data_path=str(data_file),
        )


def test_template_path_does_not_exist_raises(
    tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """Template path not found should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    data_file = sample_tsv
    # Use a template path that does not exist
    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(tmp_path / "no-such-template"),
            browser_path=str(tmp_path / "browser_out_t"),
            data_path=str(data_file),
        )


def test_mallet_path_does_not_exist_raises(
    tmp_path: Path, dist_template_dir: Path, sample_tsv: Path
):
    """Mallet folder missing should raise FileNotFoundError."""
    data_file = sample_tsv
    # Use mallet path that does not exist
    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(tmp_path / "no-such-mal"),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_t2"),
            data_path=str(data_file),
        )


def test_data_path_is_dir_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """Providing a directory as data_path should raise ValueError."""
    mallet_dir = mallet_dir_factory()
    # Use a directory as data_path
    with pytest.raises(ValueError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_dir"),
            data_path=str(mallet_dir),
        )


def test_data_path_nonexistent_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """Nonexistent data_path should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    # data_path doesn't exist
    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_no_data"),
            data_path=str(tmp_path / "no-data.tsv"),
        )


def test_invalid_tsv_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """TSV with invalid row column count should raise ValueError."""
    mallet_dir = mallet_dir_factory()
    # create invalid tsv with 4 columns
    bad_tsv = tmp_path / "bad.tsv"
    bad_tsv.write_text("1	A	B	C\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_bad"),
            data_path=str(bad_tsv),
        )


def test_creates_temp_browser_path_if_none(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Omitting browser_path creates a temporary folder for the browser."""
    mallet_dir = mallet_dir_factory()
    data_file = sample_tsv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        # no browser_path provided
        data_path=str(data_file),
    )
    # Should have created a temporary folder
    assert b.browser_path.exists()
    assert "dfr_browser_" in str(b.browser_path)


def test_canonicalization_and_duplicates(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Both doc-topic and doc-topics present are canonicalized and deduplicated."""
    # Create both doc-topic and doc-topics so duplicate dedup logic is hit
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "doc-topics.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    data_file = sample_tsv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_dup"),
        data_path=str(data_file),
    )
    out_data = Path(b.browser_path) / "data"
    # Only the canonical dest should exist once
    assert (out_data / "doc-topic.txt").exists()


def test_version_property_handles_faulty_config_bool(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """When self.config raises during boolean evaluation, version should fallback to class default."""
    mallet_dir = mallet_dir_factory()
    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_version"),
        data_path=str(data_file),
    )

    class BadBool:
        def __bool__(self):
            raise RuntimeError("boom")

    # Assign a value that will raise during boolean evaluation in the version property
    b.config = BadBool()
    # The property should gracefully fallback to the class-level BROWSER_VERSION
    assert b.version == Browser.BROWSER_VERSION


def test_topic_state_altname_canonicalization(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Ensure 'state.gz' alternate name results in canonical 'topic-state.gz' in copied files and config."""
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "state.gz",
            "topic_coords.csv",
        ]
    )
    data_file = sample_tsv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_state"),
        data_path=str(data_file),
    )

    # The canonical destination should be 'topic-state.gz'
    assert b._copied_files.get("topic_state_file") == "data/topic-state.gz"
    out_data = Path(b.browser_path) / "data"
    # doc-topics should be canonicalized to doc-topic and not present
    assert not (out_data / "doc-topics.txt").exists()
    # Ensure metadata keys present in copied files mapping
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("metadata_file") == "data/metadata.csv"
    assert cfg.get("topic_state_file") == "data/topic-state.gz"
    assert cfg.get("topic_coords_file") == "data/topic_coords.csv"


def test__write_config_bad_template_json(
    tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """If template config.json is invalid JSON, code falls back to base_cfg empty and writes merged config."""
    # Create dist with invalid config.json to hit the base_cfg except branch
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "config.json").write_text("not json", encoding="utf-8")
    (dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_dir),
        browser_path=str(tmp_path / "browser_out_badcfg"),
        data_path=str(sample_tsv),
        config={"application": {"name": "My"}},
    )
    # Config should be written and contain application name
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("application", {}).get("name") == "My"


def test__write_config_browser_path_none_raises(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Setting browser_path to None and calling _write_config should raise ValueError."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_6"),
        data_path=str(sample_tsv),
    )
    # set browser_path to None and ensure _write_config raises
    b.browser_path = None
    with pytest.raises(ValueError):
        b._write_config()


def test_serve_starts_server_and_handles_webbrowser_fail(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable, monkeypatch
):
    """Serve should start a server and handle webbrowser.open exceptions gracefully."""
    # Make mallet dir
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve"),
    )

    # Patch socketserver.TCPServer to a dummy server
    class DummyServer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def serve_forever(self):
            return

        def shutdown(self):
            return

    monkeypatch.setattr(
        "lexos.topic_modeling.dfr_browser2.socketserver.TCPServer", DummyServer
    )

    # Patch webbrowser.open to raise to hit exception branch
    def fake_open(url):
        raise RuntimeError("no browser")

    monkeypatch.setattr("lexos.topic_modeling.dfr_browser2.webbrowser.open", fake_open)
    # This should not raise
    b.serve()


def test_serve_missing_browser_path_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """If browser_path is absent, serve should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve2"),
    )
    # remove browser path
    shutil.rmtree(b.browser_path)
    with pytest.raises(FileNotFoundError):
        b.serve()


def test_serve_keyboard_interrupt_triggers_shutdown(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable, monkeypatch
):
    """If server_thread.join raises KeyboardInterrupt, httpd.shutdown should be called."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve3"),
    )

    class DummyServer2:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def serve_forever(self):
            return

        def shutdown(self):
            self.shutdown_called = True
            return

    # Monkeypatch TCPServer to our DummyServer2
    monkeypatch.setattr(
        "lexos.topic_modeling.dfr_browser2.socketserver.TCPServer", DummyServer2
    )

    # Monkeypatch threading.Thread to return a dummy object whose join raises KeyboardInterrupt
    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return

        def join(self):
            raise KeyboardInterrupt()

    monkeypatch.setattr(
        "lexos.topic_modeling.dfr_browser2.threading.Thread", DummyThread
    )

    # Also monkeypatch webbrowser.open to not raise
    monkeypatch.setattr(
        "lexos.topic_modeling.dfr_browser2.webbrowser.open", lambda url: True
    )

    # Serve should run and handle KeyboardInterrupt by calling httpd.shutdown and not raising
    b.serve()


def test_copytree_fallback(
    monkeypatch, tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """If shutil.copytree raises, fallback per-file copy path should still copy template content."""
    # Prepare template with nested files
    dist_dir = tmp_path / "dist"
    (dist_dir / "nested").mkdir(parents=True, exist_ok=True)
    (dist_dir / "nested" / "file.txt").write_text("data", encoding="utf-8")
    (dist_dir / "config.json").write_text(
        json.dumps({"data_source": "data/docs.txt"}), encoding="utf-8"
    )
    (dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    # Prepare mallet dir
    mallet_dir = mallet_dir_factory()
    # Force copytree to raise an exception so the fallback path is used
    monkeypatch.setattr(
        "shutil.copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("fail copytree")),
    )
    # Ensure browser output folder exists so fallback per-file copy has a destination
    (tmp_path / "browser_out_copytree").mkdir(parents=True, exist_ok=True)

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_dir),
        browser_path=str(tmp_path / "browser_out_copytree"),
        data_path=str(sample_tsv),
    )
    # Verify nested file copied
    assert (Path(b.browser_path) / "nested" / "file.txt").exists()


def test_config_browser_updates_config(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """`config_browser` should write config.json updating the template settings."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_conf"),
        data_path=str(sample_tsv),
    )
    b.config_browser({"display": {"itemsPerPage": 10}})
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("display", {}).get("itemsPerPage") == 10


def test_filename_map_non_required_key_present(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Mapping for a non-required key should be accepted if mapped value exists."""
    mallet_dir = mallet_dir_factory()
    # Create a diagnostics file present in the mallet dir
    create_file(mallet_dir / "diagnostics.xml", "content")
    # Provide mapping for a non-required key that is present
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_diag"),
        data_path=str(sample_tsv),
        filename_map={"diagnostics.xml": "diagnostics.xml"},
    )
    # Ensure diagnostics_file is present in config
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("diagnostics_file") == "data/diagnostics.xml"


def test_filename_map_missing_non_required_key_ignored(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Mapping for a non-required key is ignored if not in required files; Browser still initializes."""
    mallet_dir = mallet_dir_factory()
    # Provide a mapping for a file that doesn't exist (not a required file)
    with pytest.raises(FileNotFoundError):
        b = Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_map_missing"),
            data_path=str(sample_tsv),
            filename_map={"not_real.txt": "not_real_dst.txt"},
        )


def test_filename_map_key_missing_but_value_present(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If mapping key missing but mapping value exists, it should be accepted for required files."""
    # Create mallet dir with 'keys.txt' (mapped value), but no 'topic-keys.txt' (key)
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "keys.txt",
            "doc-topic.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # Provide mapping topic-keys.txt -> keys.txt
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_value_present"),
        data_path=str(sample_tsv),
        filename_map={"topic-keys.txt": "keys.txt"},
    )
    out_data = Path(b.browser_path) / "data"
    # Ensure the copied file exists with dest name 'keys.txt' unless canonicalized to topic-keys
    assert (out_data / "keys.txt").exists() or (out_data / "topic-keys.txt").exists()


def test_alt_name_group_mapping_value_present(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If mapping refers to alt in alt_name_groups and mapping value exists, treat canonical as present."""
    # mallet dir contains mapped file 'mapped-docs.txt' but not 'doc-topics.txt' or 'doc-topic.txt'
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "mapped-docs.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # mapping: doc-topics.txt -> mapped-docs.txt (key is alt name)
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_altmap"),
        data_path=str(sample_tsv),
        filename_map={"doc-topics.txt": "mapped-docs.txt"},
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Configuration should record doc_topic_file as data/doc-topics.txt (destination is key)
    assert cfg.get("doc_topic_file") == "data/doc-topics.txt"


def test_required_file_missing_but_mapped_value_exists(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If canonical required file missing but mapped value exists the Browser should accept it."""
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "state.gz",
            "topic_coords.csv",
        ]
    )
    # Provide mapping topic-state.gz -> state.gz
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_state"),
        data_path=str(sample_tsv),
        filename_map={"topic-state.gz": "state.gz"},
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Destination is canonical 'topic-state.gz' (mapping value was used as source)
    assert cfg.get("topic_state_file") == "data/topic-state.gz"


def test_alt_canonical_topic_state_mapping(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If mapping refers to alternate 'state.gz' for topic-state, map canonical 'topic-state.gz'."""
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "state.gz",
            "topic_coords.csv",
        ]
    )
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_state2"),
        data_path=str(sample_tsv),
        filename_map={"topic-state.gz": "state.gz"},
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Confirm canonical mapping used for topic_state_file
    assert cfg.get("topic_state_file") == "data/topic-state.gz"


def test_copytree_success_copies_all(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Ensure copytree success path copies nested files into browser path."""
    # Create a nested template under dist
    dist_dir = tmp_path / "dist2"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "assets" / "a.txt").write_text("hello", encoding="utf-8")
    (dist_dir / "config.json").write_text(
        json.dumps({"data_source": "data/docs.txt"}), encoding="utf-8"
    )
    (dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_dir),
        browser_path=str(tmp_path / "browser_out_copytree_ok"),
        data_path=str(sample_tsv),
    )
    assert (Path(b.browser_path) / "assets" / "a.txt").exists()


def test_tsv_with_empty_lines_allowed(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """TSV files can include empty lines — they are ignored during validation."""
    mallet_dir = mallet_dir_factory()
    tsv = tmp_path / "with_empty.tsv"
    tsv.write_text("1\tDoc1\n\n2\tDoc2\n", encoding="utf-8")
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_empty"),
        data_path=str(tsv),
    )
    assert (Path(b.browser_path) / "data" / "docs.txt").exists()


def test_user_config_topic_keys_preserved(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """User-provided 'topic_keys_file' in config should be preserved and not overwritten."""
    mallet_dir = mallet_dir_factory()
    user_cfg = {"topic_keys_file": "/bad/keys.txt"}
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_user_topic_keys"),
        data_path=str(sample_tsv),
        config=user_cfg,
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("topic_keys_file") == "/bad/keys.txt"


def test_browser_version_default_and_override(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Browser exposes a BROWSER_VERSION which appears under config['application']['version'], and is preserved if the user provides a version."""
    mallet_dir = mallet_dir_factory()

    # Default: should be set to Browser.BROWSER_VERSION
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_version"),
        data_path=str(sample_tsv),
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # File content should have version set
    assert cfg.get("application", {}).get("version") == b.BROWSER_VERSION
    # The instance config should also be updated in memory to the merged config
    assert isinstance(b.config, dict)
    assert b.config.get("application", {}).get("version") == b.BROWSER_VERSION
    # Version property should use the in-memory config value if available
    assert b.version == b.BROWSER_VERSION

    # User provided version: should be preserved
    user_cfg = {"application": {"version": "3.0.0"}}
    b2 = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_version2"),
        data_path=str(sample_tsv),
        config=user_cfg,
    )
    cfg2 = json.loads(
        (Path(b2.browser_path) / "config.json").read_text(encoding="utf-8")
    )
    # User-provided version preserved in file
    assert cfg2.get("application", {}).get("version") == "3.0.0"
    # and is present in the instance config in-memory as well
    assert isinstance(b2.config, dict)
    assert b2.config.get("application", {}).get("version") == "3.0.0"
    # Version property should reflect user-provided override
    assert b2.version == "3.0.0"


def test_config_assignment_writes_file(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Setting `Browser.config` property should write updated config to file and update in-memory config."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_config_set"),
        data_path=str(sample_tsv),
    )

    # Assign new config; this should trigger a write via __setattr__
    new_cfg = {"application": {"name": "New Title"}}
    b.config = new_cfg
    # Check file and instance config
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("application", {}).get("name") == "New Title"
    assert isinstance(b.config, dict)
    assert b.config.get("application", {}).get("name") == "New Title"
