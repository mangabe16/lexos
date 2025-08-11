"""d3.bubbleviz.py.

Last Updated: 10 August, 2025
Last Tested: TBD

Usage:

```python
D3BubbleChart(data=data, title="My Bubble Chart", auto_open=True)
```

By default, it will auto-open in a web browser. However, you can set `auto_open=False` to prevent this behavior. You can then call the `save` method to save the HTML to a file.

# TODO: Add doc selection.
"""

import tempfile
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.tokens import Doc, Span, Token

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization import processors

# Valid input types
single_doc_types = dict[str, int] | Doc | Span | str | list[str] | list[Token]
multi_doc_types = (
    str
    | list[str]
    | list[list[str]]
    | list[Doc]
    | list[Span]
    | list[list[Token]]
    | dict[str, int]
    | pd.DataFrame
    | DTM
)


class D3BubbleChart(BaseModel):
    """Class to render a D3 bubble chart visualization in HTML format."""

    data: single_doc_types | multi_doc_types | pd.DataFrame = Field(
        ...,
        description="The data to generate the bubble chart from. Accepts data from a string, list of lists or tuples, a dict with terms as keys and counts/frequencies as values, or a dataframe.",
    )
    title: Optional[str] = Field(
        "Bubble Chart Visualization", description="The title of the chart."
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of bubbles to display."
    )
    height: int = Field(600, description="The height of the chart.")
    width: int = Field(960, description="The width of the chart.")
    margin: dict[str, int] = Field(
        {"top": 20, "right": 20, "bottom": 20, "left": 20},
        description="The margin around the chart.",
    )
    color: str = Field(
        "schemeCategory10",
        description="The color scheme for the chart, either the name D3 color scheme or a list of custom colors.",
    )
    template: Path | str = Field(
        "d3_bubbles_template-1.0.html",
        description="The template file for the bubble chart visualization.",
    )
    auto_open: bool = Field(
        True, description="Whether to open the chart in a web browser automatically."
    )
    html: str = Field(None, description="The rendered HTML for the bubble chart.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the D3BubbleChart with the provided data."""
        super().__init__(**data)
        self.template = self._get_asset_path(self.template)
        # Process the data into a consistent format
        self.counts = self._process_data()
        self._render()

    def _get_asset_path(self, filename: str) -> Path:
        """Centralized asset path resolution.

        Args:
            filename (str): The name of the asset file.

        Returns:
            Path: The full path to the asset file.
        """
        return Path(__file__).parent / "d3_cloud_assets" / filename

    def _load_template(self) -> str:
        """Load the HTML template for the bubble chart."""
        with open(self.template) as f:
            return f.read()

    def _process_data(self) -> dict[str, int]:
        """Process the input data into a consistent format of term counts.

        Returns:
            dict[str, int]: Dictionary with terms as keys and counts as values.
        """
        # Handle simple string input
        if isinstance(self.data, str):
            counts = Counter(self.data.split())  # TODO: Use better tokenizer

        # Handle spaCy objects
        if isinstance(self.data, (Doc, Span)):
            counts = Counter([token.text for token in self.data])

        # Handle dictionary input (already in correct format)
        if isinstance(self.data, dict):
            counts = Counter(self.data)

        # Handle list inputs
        if isinstance(self.data, list):
            counts = self._process_list_data()

        # Handle structured data types
        counts = self._process_structured_data()

        # Limit the number of terms if specified
        if self.limit is not None:
            counts = dict(counts.most_common(self.limit))

        return dict(counts)

    def _process_list_data(self) -> dict[str, int]:
        """Process list-type data inputs.

        Returns:
            dict[str, int]: Dictionary with terms as keys and counts as values.
        """
        if not self.data:
            return {}

        first_item = self.data[0]

        # List of lists
        if isinstance(first_item, list):
            return processors.process_list(self.data, self.docs)

        # List of spaCy objects
        if isinstance(first_item, (Doc, Span)):
            return processors.process_docs(self.data, self.docs)

        # Simple list of strings/tokens
        return processors.process_item(self.data)

    def _process_structured_data(self) -> dict[str, int]:
        """Process structured data types (DTM, DataFrame).

        Returns:
            dict[str, int]: Dictionary with terms as keys and counts as values.
        """
        if isinstance(self.data, DTM):
            return processors.process_dtm(self.data, self.docs)

        if isinstance(self.data, pd.DataFrame):
            return processors.process_dataframe(self.data, self.docs)

        raise LexosException(
            f"Unsupported data type: {type(self.data)}. "
            "Supported types: str, dict, list, DTM, DataFrame, spaCy Doc/Span objects."
        )

    def _open(self) -> None:
        """Open the HTML file in a web browser."""
        # Create a temporary file to store the HTML
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".html", encoding="utf-8"
        ) as temp_file:
            temp_file.write(self.html)
            temp_file_path = temp_file.name

            # Open the temporary HTML file in the default web browser
            webbrowser.open(f"file:///{temp_file_path}")

    def _render(self) -> None:
        """Render the bubble chart as an HTML string."""
        # Load the template
        template = Template(self._load_template())

        # Render the template with the instance variables
        self.html = template.render(
            title=self.title,
            term_counts=self.data,
            height=self.height,
            width=self.width,
            margin=self.margin,
            color=self.color,
        )

        # If auto_open is True, open the chart in a web browser
        if self.auto_open:
            self._open()

    @validate_call
    def save(self, path: Path | str) -> None:
        """Save the HTML file.

        Args:
            path (Path | str): The path where the HTML file will be saved.
        """
        with open(path, "w") as f:
            f.write(self.html)
