"""__init__.py."""

import json
import os
import shutil
import socketserver
import tempfile
import threading
import uuid
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, validate_call


class Browser(BaseModel):
    """Browser class to create and serve DFR Browser 2.

    filename_map usage:
    - Provide a mapping of `original_filename` -> `destination_filename` where
      `original_filename` is the filename present in `mallet_files_path` and
      `destination_filename` is the name to use under the browser's `data/` folder.

    Example:
        filename_map = {"doc-topics.txt": "doc-topic.txt"}
        Browser(..., filename_map=filename_map)
    """

    mallet_files_path: str = Field(
        ..., description="Path to the folder containing Mallet output files."
    )
    browser_path: str = Field(
        None, description="The folder where the browser will be saved."
    )
    template_path: str = Field(
        "dist", description="Path to the DFR Browser 2 template folder."
    )
    data_path: str | None = Field(
        None,
        description=(
            "Path to a tab-separated (TSV) file containing the original data used "
            "to generate the topic model. Each row must contain 2 or 3 columns. "
            "This file will be copied into the browser's data folder."
        ),
    )
    config: dict | None = Field(
        None, description="Configuration dictionary for the DFR Browser 2."
    )
    port: int = Field(8000, description="Port number for serving the DFR Browser 2.")
    filename_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original filenames to new filenames.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_call
    def __init__(self, **data) -> None:
        """Initialize the DFR Browser 2 class."""
        # First call BaseModel initializer
        super().__init__(**data)

        # Convert paths into Path objects
        self.mallet_files_path = Path(self.mallet_files_path)
        self.template_path = Path(self.template_path)
        if self.browser_path:
            self.browser_path = Path(self.browser_path)
        else:
            # Create temp directory if none provided
            self.browser_path = Path(tempfile.mkdtemp(prefix="dfr_browser_"))

        # Required mallet files - canonical names. Some files may have alternate
        # acceptable names (e.g. doc-topic vs doc-topics)
        required_files = {
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "topic-state.gz",
            "topic_coords.csv",
        }

        # Alternate names mapping: canonical -> list of alternative names
        alt_name_groups = {
            "doc-topic.txt": ["doc-topics.txt", "doc-topic.txt"],
            # Accept 'state.gz' as an alternate name for 'topic-state.gz'
            "topic-state.gz": ["topic-state.gz", "state.gz"],
        }

        # Check the template and mallet path exist
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template path does not exist: {self.template_path}"
            )
        if not self.mallet_files_path.exists():
            raise FileNotFoundError(
                f"Mallet files path does not exist: {self.mallet_files_path}"
            )

        # Determine filenames to check. If user provided filename_map, treat
        # keys as filenames that exist in the mallet files path (orig names),
        # and values as the desired destination names.
        filenames_to_check = set(required_files)
        if self.filename_map:
            # Include keys from the filename_map (original filenames)
            filenames_to_check.update(self.filename_map.keys())

        # Missing check attempts to accept either the key or mapped value in case the provided map was reversed
        missing = []
        # First check canonical required files with alternates
        for f in required_files:
            # If this canonical file is part of an alternate group, check if any alt exists
            if f in alt_name_groups:
                any_present = False
                for alt in alt_name_groups[f]:
                    if (self.mallet_files_path / alt).exists():
                        any_present = True
                        break
                # Also check mapping values in case filename_map was supplied with alternative names
                if not any_present and self.filename_map:
                    for alt in alt_name_groups[f]:
                        # If mapping key exists and maps to a file that exists, treat present
                        # If filename_map maps the alt name to another filename, check whether
                        # the mapped source exists in the mallet dir. This handles reversed
                        # filename_map cases where the mapping indicates a source that must be used.
                        if alt in self.filename_map:
                            if (
                                self.mallet_files_path / self.filename_map[alt]
                            ).exists():
                                any_present = True
                                break
                        # The following duplicate check (looking up mapping via .get)
                        # is unnecessary — `alt in self.filename_map` already handles
                        # the same case. It is intentionally omitted to remove unreachable
                        # duplicate logic.
                if not any_present:
                    missing.append(f)
            else:
                if (self.mallet_files_path / f).exists():
                    continue
                if self.filename_map and f in self.filename_map:
                    if (self.mallet_files_path / self.filename_map[f]).exists():
                        continue
                missing.append(f)

        # Now check any filename_map keys that aren't in required_files (validate mapping keys)
        if self.filename_map:
            for key in self.filename_map.keys():
                # If key is one of the canonical required files, we've already checked it
                if key in required_files:
                    continue
                if (self.mallet_files_path / key).exists():
                    continue
                # check if mapped value exists in mallet files
                if (self.mallet_files_path / self.filename_map[key]).exists():
                    continue
                missing.append(key)
        if missing:
            raise FileNotFoundError(
                f"Missing required mallet files in {self.mallet_files_path}: {missing}"
            )
        try:
            shutil.copytree(self.template_path, self.browser_path, dirs_exist_ok=True)
        except Exception:
            # On some systems, copying into existing directory fails for file metadata; fallback to per-file copy
            for src in self.template_path.rglob("*"):
                rel = src.relative_to(self.template_path)
                dest = self.browser_path / rel
                if src.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    shutil.copy2(src, dest)

        # If data_path provided, it must be a TSV file (2 or 3 columns per row)
        # and it will be copied into a 'data' subfolder in browser_path.
        if self.data_path:
            data_src = Path(self.data_path)
            if not data_src.exists():
                raise FileNotFoundError(f"data_path does not exist: {data_src}")
            if data_src.is_dir():
                raise ValueError(
                    "data_path must be a path to a TSV file, not a directory"
                )

            # Validate TSV structure: each non-empty row must have 2 or 3 columns
            with open(data_src, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    cols = line.split("\t")
                    if len(cols) not in (2, 3):
                        raise ValueError(
                            f"Invalid TSV format in {data_src} at line {i}: expected 2 or 3 columns, got {len(cols)}"
                        )

            data_target = self.browser_path / "data"
            data_target.mkdir(parents=True, exist_ok=True)
            # Copy file to the data directory using the expected filename 'docs.txt'
            docs_filename = "docs.txt"
            shutil.copy2(data_src, data_target / docs_filename)
            # Track which files we copied (so we can update config.json paths)
            self._copied_files = getattr(self, "_copied_files", {})
            # Save relative path as used by the template
            self._copied_files["data_source"] = f"data/{docs_filename}"

        # Copy mallet files into the 'data' folder (not a mallet subfolder)
        data_target = self.browser_path / "data"
        data_target.mkdir(parents=True, exist_ok=True)
        # Ensure we have a holder for copied files metadata
        self._copied_files = getattr(self, "_copied_files", {})
        copied_destnames = set()
        # Expand filenames to include alternate names so we can copy the actual file
        copy_candidates = set(filenames_to_check)
        for canonical, alts in alt_name_groups.items():
            for alt in alts:
                copy_candidates.add(alt)
        for src_name in copy_candidates:
            # If the src_name is a filename that only appears as a mapping value
            # (i.e., the mapping is reversed where key is the destination), then
            # skip copying this source directly — the mapping key will handle copying/rename.
            if (
                self.filename_map
                and src_name in set(self.filename_map.values())
                and (src_name not in self.filename_map)
            ):
                continue
            # Interpret src_name as the original filename (key)
            src_path = self.mallet_files_path / src_name
            dest_filename = src_name
            # If the filename is an alternate name for a canonical file, default to the canonical
            # filename as the destination unless the user explicitly specified a mapping for it.
            if not (self.filename_map and src_name in self.filename_map):
                for canonical, alts in alt_name_groups.items():
                    if src_name in alts:
                        dest_filename = canonical
                        break
            # If a mapping exists for this src_name, the mapped value is the destination filename
            if self.filename_map and src_name in self.filename_map:
                dest_filename = self.filename_map[src_name]
            # If the file doesn't exist under the original name, try the destination name
            if not src_path.exists():
                fallback_path = self.mallet_files_path / dest_filename
                if fallback_path.exists():
                    src_path = fallback_path
                    # If the user supplied a mapping where the key was actually the
                    # desired destination (i.e., mapping was reversed), then swap
                    # the dest filename so we rename original->dest correctly.
                    if (
                        self.filename_map
                        and src_name in self.filename_map
                        and (self.filename_map[src_name] == fallback_path.name)
                    ):
                        # key was destination, value was source, so swap
                        dest_filename = src_name
                else:
                    # If missing, skip
                    continue
            # Prevent duplicate destination filenames (e.g. doc-topic vs doc-topics)
            if dest_filename in copied_destnames:
                continue
            dest_path = data_target / dest_filename
            shutil.copy2(src_path, dest_path)
            copied_destnames.add(dest_filename)
            # Record the copied file and map to expected config.json key
            # Determine canonical group for src_name (if any) — this ensures that even when
            # the destination filename doesn't contain the canonical substring (because of filename_map),
            # we still map config entries against the canonical group.
            canonical_for_src = None
            for canonical, alts in alt_name_groups.items():
                for alt in alts:
                    if src_name == alt:
                        canonical_for_src = canonical
                        break
                if canonical_for_src:
                    break

            lower = dest_filename.lower()
            if canonical_for_src == "doc-topic.txt":
                # If source belonged to doc-topic alt group, map regardless of dest filename
                self._copied_files["doc_topic_file"] = f"data/{dest_filename}"
            if "topic-keys" in lower or "topic_keys" in lower:
                self._copied_files["topic_keys_file"] = f"data/{dest_filename}"
            elif "doc-topic" in lower or "doc-topics" in lower or "doc_topic" in lower:
                # Template uses 'doc_topic_file'
                self._copied_files["doc_topic_file"] = f"data/{dest_filename}"
            elif canonical_for_src == "topic-state.gz":
                # ‘topic-state.gz’ canonical group — ensure mapping written even if dest filename doesn't contain keyword
                self._copied_files["topic_state_file"] = f"data/{dest_filename}"
            elif "metadata" in lower:
                self._copied_files["metadata_file"] = f"data/{dest_filename}"
            elif "topic-state" in lower or "topic_state" in lower:
                self._copied_files["topic_state_file"] = f"data/{dest_filename}"
            elif "topic_coords" in lower or "topic-coords" in lower:
                self._copied_files["topic_coords_file"] = f"data/{dest_filename}"
            elif "diagnostics" in lower:
                self._copied_files["diagnostics_file"] = f"data/{dest_filename}"

        # Write or update config.json if present in the template
        # First, read and merge the template config with any user-provided `self.config`.
        # Make sure file paths inside the config point to the files we copied into `data/`.
        self._write_config()

    def config_browser(self, config: dict) -> None:
        """Set the browser configuration after initialization."""
        # Update the config attribute
        self.config = config
        # Write the new config to config.json in browser_path
        self._write_config()

    def serve(self) -> None:
        """Serve the DFR Browser 2 in a web browser."""
        # Launch a local web server to serve the browser_path and open the user's browser
        # Ensure browser_path exists
        if not self.browser_path.exists():
            raise FileNotFoundError(f"Browser path does not exist: {self.browser_path}")

        # Save current working directory
        cwd = Path.cwd()
        try:
            os.chdir(self.browser_path)

            handler = SimpleHTTPRequestHandler
            # Attempt to bind to the specified port
            with socketserver.TCPServer(("", int(self.port)), handler) as httpd:
                url = f"http://localhost:{self.port}/"
                print(f"Serving DFR Browser at {url}")
                # Open in web browser
                try:
                    webbrowser.open(url)
                except Exception:
                    print(
                        "Unable to open web browser automatically. Please open in a browser manually."
                    )

                # Serve in a background thread so the call doesn't block
                server_thread = threading.Thread(
                    target=httpd.serve_forever, daemon=True
                )
                server_thread.start()
                print("Server started (press CTRL+C to stop)")
                try:
                    # Keep main thread alive while server is running
                    server_thread.join()
                except KeyboardInterrupt:
                    print("Shutting down server")
                    httpd.shutdown()
        finally:
            os.chdir(cwd)

    def _write_config(self) -> None:
        """Internal helper to write config.json in the browser_path."""
        if not self.browser_path:
            raise ValueError("browser_path is not set")
        self.browser_path.mkdir(parents=True, exist_ok=True)
        cfg_path = Path(self.browser_path) / "config.json"
        # Load existing template config.json (if it exists) as the base
        base_cfg = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    base_cfg = json.load(fh)
            except Exception:
                base_cfg = {}

        # Merge user-provided config into base (user overrides base)
        merged_cfg = dict(base_cfg)
        if self.config:
            merged_cfg.update(self.config)

        # Ensure file paths for known data files point to the data/ folder
        copied = getattr(self, "_copied_files", {}) or {}
        for key, rel_path in copied.items():
            # File path precedence:
            # 1. User-specified config (self.config) — should win and be preserved
            # 2. Copied file paths (this process) — override template values
            # 3. Template defaults — only used if no user or copied path applies.
            # If the user explicitly provided this key in the config, preserve it.
            if self.config and key in self.config:
                continue
            # Otherwise, set the key to the path of the copied file (overriding template)
            merged_cfg[key] = rel_path

        # Save merged config
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(merged_cfg, f, indent=2)
