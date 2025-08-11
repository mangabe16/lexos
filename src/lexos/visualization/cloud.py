"""cloud.py.

Last Update: August 11, 2025
Last Tested: TBD
"""

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from wordcloud import WordCloud as PythonWordCloud

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


class WordCloud(BaseModel):
    """A Pydantic model for WordCloud options."""

    data: single_doc_types | multi_doc_types | pd.DataFrame = Field(
        ...,
        description="The data to generate the word cloud from. Accepts data from a string, list of lists or tuples, a dict with terms as keys and counts/frequencies as values, or a dataframe.",
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM."
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of terms to plot."
    )
    title: Optional[str] = Field(None, description="The title of the plot.")
    height: int = Field(
        200, gt=50, description="The height of the word cloud in pixels."
    )
    width: int = Field(200, gt=50, description="The width of the word cloud in pixels.")
    opts: Optional[dict[str, Any]] = Field(
        {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options.",
    )
    figure_opts: Optional[dict[str, Any]] = Field(
        {}, description="A dict of matplotlib figure options."
    )
    round: Optional[int] = Field(
        0,
        description="An integer to apply a mask that rounds the word cloud. It is best to use 100 or higher for a circular mask, but it will depend on the height and width of the word cloud.",
    )
    counts: dict[str, int] = Field(None, description="A dictionary of term counts.")
    cloud: PythonWordCloud | None = Field(
        None, description="The generated WordCloud object."
    )
    fig: Optional[plt.Figure] = Field(
        None, description="The matplotlib figure object for the word cloud."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the WordCloud model."""
        super().__init__(**data)

        # Set the figure dimensions
        self.opts["height"] = self.height
        self.opts["width"] = self.width

        # Set the mask, if using
        if self.round > 0:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > self.round**2
            mask = 255 * mask.astype(int)
            self.opts["mask"] = mask

        # Process the data into a consistent format
        self.counts = processors.process_data(self.data, self.docs, self.limit)
        self.cloud = PythonWordCloud(**self.opts).generate_from_frequencies(self.counts)
        plt.close()

    @validate_call
    def save(self, path: Path | str) -> None:
        """Save the WordCloud to a file."""
        if self.cloud is None:
            raise LexosException("No WordCloud object to save.")
        self.cloud.to_file(path)

    def show(self) -> None:
        """Show the figure if it is hidden.

        This is a helper method. It will generally display in a
        Jupyter notebook.
        """
        if self.title:
            plt.title(self.title)
        plt.axis("off")
        plt.imshow(self.cloud)


class MultiCloud(BaseModel):
    """A Pydantic model for creating multiple WordClouds arranged in a grid."""

    data: list[str] | list[Doc] | list[Span] | DTM | pd.DataFrame = Field(
        ...,
        description="The data to generate word clouds from. Accepts list of documents, DTM, or DataFrame.",
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM/DataFrame."
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of terms to plot."
    )
    ncols: int = Field(3, gt=0, description="Number of columns in the grid layout.")
    height: int = Field(
        200, gt=50, description="The height of each word cloud in pixels."
    )
    width: int = Field(
        200, gt=50, description="The width of each word cloud in pixels."
    )
    opts: Optional[dict[str, Any]] = Field(
        {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options applied to each word cloud.",
    )
    figure_opts: Optional[dict[str, Any]] = Field(
        {}, description="A dict of matplotlib figure options."
    )
    round: Optional[int] = Field(
        0,
        description="An integer to apply a mask that rounds each word cloud. It is best to use 100 or higher for a circular mask.",
    )
    title: Optional[str] = Field(None, description="Overall title for the figure.")
    labels: Optional[list[str]] = Field(
        None, description="Labels for each subplot/word cloud."
    )
    padding: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Amount of padding between subplots (0.0 to 1.0).",
    )
    clouds: list[WordCloud] = Field(
        default_factory=list, description="List of generated WordCloud objects."
    )
    fig: Optional[plt.Figure] = Field(
        None, description="The matplotlib figure object for the multi-cloud plot."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the MultiCloud model."""
        super().__init__(**data)

        # Process different data types to get individual document data
        doc_data = self._process_data()

        # Create individual WordCloud objects
        self.clouds = []
        for doc in doc_data:
            try:
                # Create a WordCloud instance for each document
                wc = WordCloud(
                    data=doc,
                    limit=self.limit,
                    opts=self.opts,
                    round=self.round,
                    width=self.width,
                    height=self.height,
                )
                self.clouds.append(wc)
            except Exception as e:
                raise LexosException(f"Failed to create word cloud: {e}")

    def _process_data(self) -> list:
        """Process the input data into individual documents."""
        if isinstance(self.data, DTM):
            # Extract documents from DTM
            doc_data = []
            selected_docs = (
                self.docs
                if self.docs is not None
                else range(len(self.data.doc_term_matrix))
            )
            if isinstance(selected_docs, (int, str)):
                selected_docs = [selected_docs]

            for doc_idx in selected_docs:
                # Get term frequencies for this document
                if isinstance(doc_idx, str):
                    doc_idx = self.data.doc_labels.index(doc_idx)
                doc_counts = {}
                for term_idx, count in enumerate(self.data.doc_term_matrix[doc_idx]):
                    if count > 0:
                        doc_counts[self.data.terms[term_idx]] = count
                doc_data.append(doc_counts)

        elif isinstance(self.data, pd.DataFrame):
            # Process DataFrame - assume it's a document-term matrix
            doc_data = []
            selected_docs = (
                self.docs if self.docs is not None else range(len(self.data))
            )
            if isinstance(selected_docs, (int, str)):
                selected_docs = [selected_docs]

            for doc_idx in selected_docs:
                if isinstance(doc_idx, str):
                    doc_idx = self.data.index.get_loc(doc_idx)
                doc_counts = self.data.iloc[doc_idx].to_dict()
                # Filter out zero counts
                doc_counts = {k: v for k, v in doc_counts.items() if v > 0}
                doc_data.append(doc_counts)

        elif isinstance(self.data, list):
            # Handle list of documents
            doc_data = self.data
        else:
            raise LexosException("Unsupported data type for MultiCloud")

        return doc_data

    @validate_call
    def save(self, path: Path | str) -> None:
        """Save the MultiCloud figure to a file."""
        if self.fig is None:
            self.show()  # Generate the figure first
        if self.fig is None:
            raise LexosException("No figure to save.")
        self.fig.savefig(path)

    def show(self) -> plt.Figure:
        """Generate and display the multi-cloud figure."""
        # Calculate layout
        num_clouds = len(self.clouds)
        nrows = int(np.ceil(num_clouds / self.ncols))

        # Set up figure with padding
        figure_opts = self.figure_opts.copy()
        figure_opts.setdefault("figsize", (self.ncols * 4, nrows * 3))

        # Remove constrained_layout if it exists since we're setting manual spacing
        figure_opts.pop("constrained_layout", None)

        self.fig, axes = plt.subplots(nrows, self.ncols, **figure_opts)

        # Add padding between subplots and adjust top margin for title
        if self.title:
            # More space below title when there's a suptitle
            self.fig.subplots_adjust(
                wspace=self.padding,
                hspace=self.padding,
                top=0.82,  # Leaves more space at the top for the title
            )
        else:
            # Normal spacing when no title
            self.fig.subplots_adjust(wspace=self.padding, hspace=self.padding)

        # Add padding between subplots
        self.fig.subplots_adjust(wspace=self.padding, hspace=self.padding)

        # Handle single row case
        if nrows == 1:
            axes = axes.reshape(1, -1) if self.ncols > 1 else np.array([[axes]])
        elif self.ncols == 1:
            axes = axes.reshape(-1, 1)

        # Add overall title
        if self.title:
            self.fig.suptitle(self.title, fontsize=16, y=0.90)  # Positioned lower

        # Plot each word cloud
        for i, cloud in enumerate(self.clouds):
            row = i // self.ncols
            col = i % self.ncols

            ax = axes[row, col]

            # Display the word cloud
            ax.imshow(cloud.cloud, interpolation="bilinear")
            ax.axis("off")

            # Add label if provided
            if self.labels and i < len(self.labels):
                ax.set_title(self.labels[i])
            elif hasattr(cloud.data, "__len__"):
                ax.set_title(f"Doc {i + 1}", fontdict={"fontsize": 10})

        # Hide unused subplots
        for i in range(num_clouds, nrows * self.ncols):
            row = i // self.ncols
            col = i % self.ncols
            axes[row, col].axis("off")
            axes[row, col].set_visible(False)

        plt.close()

        return self.fig

    def get_clouds(self) -> list[WordCloud]:
        """Return the list of individual WordCloud objects."""
        return self.clouds
