"""__init__.py.

Last updated: April 28, 2025
Last tested: Last updated: April 28, 2025

Instantiating the object automatically creates the browser directory, scales the model state file,
and copies the metadata and template files into the browser directory.

The metadata must be a pandas dataframe with the columns described in the dfr-browser documentation:
https://github.com/agoldst/dfr-browser?tab=readme-ov-file#browser-data-file-specifications.

The browser is launched with `DfrBrowser.serve()`.

If you set `embargo=True`, the original documents will be inaccesible from the dfr-browser. If you include the path to the model's `data.txt` file, the documents will be copied to the browser's `docs` directory and accessed from that location. On the other hand, if the metadata contains a `doc_uri` field, the documents will accessed via the uri provided in that field. If you do not provide the path to the model's `data.txt` file or a `doc_uri` field in the metadata, the documents will be treated as embargoed.
"""

import re
import shutil
import socketserver
import sys
import threading
import time
import webbrowser
from csv import QUOTE_ALL
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call

from lexos.exceptions import LexosException
from lexos.topic_modeling.dfr_browser import prepare_data, scale_model

DFR_BROWSER_TEMPLATE_DIR = Path(__file__).parent / "template"


class DfrBrowser(BaseModel):
    """DfrBrowser class."""

    path_to_browser_dir: str = Field(
        json_schema_extra={"description": "The path to the browser directory."}
    )
    metadata: Optional[list[dict[str, Any]]] = Field(
        None,
        json_schema_extra={
            "description": "A list of dicts containing dfr-browser compatible metadata. It should be possible to convert this to a pandas dataframe using the records orientation."
        },
    )
    num_topics: int = Field(
        json_schema_extra={"description": "The number of topics in the model."}
    )
    path_to_state_file: str = Field(
        json_schema_extra={"description": "The path to the model state file."}
    )
    path_to_data_file: Optional[str] = Field(
        None,
        json_schema_extra={
            "description": "The path to the model's data.txt file. Supply this if you wish to copy documents to the browser directory."
        },
    )
    properties: Optional[dict] = Field(
        None,
        json_schema_extra={
            "description": "A dictionary containing properties for the dfr-browser's `info.json` file."
        },
    )
    embargo: Optional[bool] = Field(
        False,
        json_schema_extra={
            "description": "Whether to embargo the data. If True, the data will not be displayed."
        },
    )
    path_to_template_dir: Optional[str] = Field(
        DFR_BROWSER_TEMPLATE_DIR,
        json_schema_extra={"description": "The template directory."},
    )
    port: Optional[int] = Field(
        8888, json_schema_extra={"description": "The port to run the browser on."}
    )
    handler: Optional[Callable] = Field(
        None,
        json_schema_extra={
            "description": "The handler for the HTTP server. This attribute does not need to be set manually; it is set automatically when the server is started."
        },
    )
    path_to_scaled_file: Optional[str] = Field(
        None,
        json_schema_extra={
            "description": "The path to the scaled file produced by the _scale_model method. This attribute does not need to be set manually; it is set automatically when the model is built."
        },
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Initialize the DfrBrowser class.

        Args:
            **data: The data to initialize the class with.
        """
        super().__init__(**data)
        self.properties = self.properties or {}
        # If the metadata is provided as a list of dicts, convert it to a pandas dataframe
        if self.metadata:
            self.metadata = pd.DataFrame(
                self.metadata,
            )
            self.metadata = self.metadata.fillna("")
            # self.metadata = self.metadata.astype(str)
        else:
            self.metadata = pd.DataFrame()

    def _convert_state(self, data_dir: str) -> None:
        """Convert the state file to JSON for dfr-browser.

        Args:
            data_dir (str): The path to the data directory.
        """
        # Convert the state file to JSON for dfr-browser
        prepare_data.convert_state(
            self.path_to_state_file,
            f"{data_dir}/tw.json",
            f"{data_dir}/dt.json.zip",
            self.num_topics,
        )

    def _copy_docs(self) -> None:
        """Copy the document texts to the browser `docs` directory."""
        try:
            df = pd.read_csv(self.path_to_data_file, sep="\t", header=None)
        except FileNotFoundError as e:
            raise LexosException(f"Could not find data file: {e}")
        except pd.errors.EmptyDataError as e:
            raise LexosException(f"Data file is empty: {e}")
        except pd.errors.ParserError as e:
            raise LexosException(f"Data file is not in the correct format: {e}")
        try:
            docs = df.iloc[:, 2].tolist()
        except IndexError as e:
            raise LexosException(
                f"Data file does not have the correct number of columns: {e}"
            )
        docs_dir = Path(self.path_to_browser_dir) / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        self.properties["doc_uris"] = []
        for i, doc in enumerate(docs):
            filepath = docs_dir / f"doc{i}.txt"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc)
            self.properties["doc_uris"].append(filepath.as_posix())

    def _copy_metadata(self, data_dir: str) -> None:
        """Copy the metadata to the browser directory.

        Args:
            data_dir (str): The path to the data directory.
        """
        if not isinstance(self.metadata, pd.DataFrame):
            try:
                self.metadata = pd.DataFrame(self.metadata)
                self.metadata = self.metadata.fillna("")
                # self.metadata = self.metadata.astype(str)
            except (AttributeError, ValueError) as e:
                raise LexosException(
                    f"Cannot parse metadata as a pandas dataframe: {e}"
                )

        # TODO: Currently, dfr-browser requires both a zipped and an unzipped version of the metada.
        # Add metadata to data_dir
        self.metadata.to_csv(
            Path(data_dir) / "meta.csv",
            index=False,
            header=False,
            quoting=QUOTE_ALL,
        )

        # Add metadata zip to data_dir
        self.metadata.to_csv(
            Path(data_dir) / "meta.csv.zip",
            index=False,
            header=False,
            quoting=QUOTE_ALL,
            compression={"method": "zip"},
        )

    def _copy_template(self):
        """Copy the template directory to the browser directory."""
        try:
            shutil.copytree(self.path_to_template_dir, self.path_to_browser_dir)
        except FileNotFoundError as e:
            raise LexosException(f"Could not find dfr-browser template: {e}")

    def _scale_model(self) -> None:
        """Scale the model using the MALLET state file."""
        try:
            # Convert the mallet output_state file to a pyLDAvis data object
            converted_data = scale_model.convert_mallet_data(self.path_to_state_file)

            # Get the topic coordinates in a dataframe
            topic_coordinates = scale_model.get_topic_coordinates(**converted_data)

            # Save the topic coordinates to a CSV file
            topic_coordinates.to_csv(
                self.path_to_scaled_file, index=False, header=False
            )
        except BaseException as e:
            raise LexosException(f"Failed!: {e}")

    def _serve_forever(self) -> None:
        """Serve the browser."""
        if self.handler is None:
            raise LexosException("Handler is not set. Please call serve() first.")
        with socketserver.TCPServer(("", self.port), self.handler) as httpd:
            httpd.serve_forever()

    def _update_assets(self):
        """Update browser assets."""
        # Tweak default index.html to link to JSON, not JSTOR
        with open(f"{self.path_to_browser_dir}/index.html", "r") as f:
            filedata = f.read().replace("on JSTOR", "JSON")
        with open(f"{self.path_to_browser_dir}/index.html", "w") as f:
            f.write(filedata)
        # Tweak js file to link to the domain
        with open(
            f"{self.path_to_browser_dir}/js/dfb.min.js.custom", "r", encoding="utf-8"
        ) as f:
            filedata = f.read()
        pat = r"t\.select\(\"#doc_remark a\.url\"\).attr\(\"href\", .+?\);"
        new_pat = r'var doc_url = document.URL.split("modules")[0] + "project_data"; t.select("#doc_remark a.url")'
        new_pat += r'.attr("href", doc_url + "/" + e.url);'
        filedata = re.sub(pat, new_pat, filedata)
        with open(
            f"{self.path_to_browser_dir}/js/dfb.min.js", "w", encoding="utf-8"
        ) as f:
            f.write(filedata)

    def _validate_settings(self) -> None:
        """Validate the settings."""
        for path in [
            self.path_to_state_file,
            self.path_to_template_dir,
        ]:
            if not Path(path).exists():
                raise LexosException(f"Path does not exist: {path}")
            if not self.num_topics:
                raise LexosException("Number of topics must be provided.")
            if self.port is None:
                raise LexosException("Port number must be provided.")

    @validate_call(config=model_config)
    def build(self) -> None:
        """Build the dfr-browser."""
        # Validate that the paths and port exist
        self._validate_settings()

        # Make a browser directory and copy the template into it
        if not Path(self.path_to_browser_dir).exists():
            self._copy_template()

        # Make a data directory
        data_dir = Path(self.path_to_browser_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Scale the model
        path_to_scaled_file = Path(self.path_to_browser_dir) / "topic_scaled.csv"
        self.path_to_scaled_file = path_to_scaled_file.as_posix()
        self._scale_model()

        # Convert state file to JSON for dfr-browser
        self._convert_state(data_dir)

        # Create info.json for dfr-browser
        if self.embargo:
            self.properties["embargo"] = True

        # If the data file is provided, copy the documents to the browser directory
        if self.path_to_data_file:
            self._copy_docs()

        # If uris are provided, in the metadata, use those
        elif "doc_uri" in self.metadata.columns:
            self.properties["doc_uris"] = self.metadata.doc_uri.tolist()

        # Otherwise, treat the docs as embargoed
        else:
            self.properties["embargo"] = True
        prepare_data.info_stub(f"{data_dir.as_posix()}/info.json", self.properties)

        # Copy scaled file into data dir
        shutil.copy(self.path_to_scaled_file, data_dir)

        # Copy the metadata to the browser directory
        self._copy_metadata(data_dir)

        # Update assets
        self._update_assets()
        print("Done!")

    @validate_call(config=model_config)
    def serve(self, port: Optional[int] = None) -> None:
        """Launch the dfr-browser.

        Args:
            port (int): The port to run the browser on.
        """
        if port:
            self.port = port

        directory = Path(self.path_to_browser_dir)
        self.handler = partial(SimpleHTTPRequestHandler, directory=directory)
        try:
            thread = threading.Thread(target=self._serve_forever)
            thread.daemon = True  # Let the parent kill the child thread at exit
            thread.start()
            print(f"Serving Dfr-Browser from {directory}.")
            print(
                "Type Ctrl-C to stop the server. If you are serving from a notebook, interrupt the kernel."
            )
            time.sleep(2)
            webbrowser.open(f"http://127.0.0.1:{self.port}/")
            while thread.is_alive():
                thread.join(1)  # time out not to block KeyboardInterrupt
        except KeyboardInterrupt:
            print("Server interrupted.")
            sys.exit(1)
