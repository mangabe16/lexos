"""refactored_wordcloud.py.

Last Update: July 7, 2025
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
    opts: Optional[dict[str, Any]] = Field(
        {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options.",
    )
    showfig: Optional[bool] = Field(
        True,
        description="Whether to show the plotted word cloud or return it as a WordCloud object.",
    )
    figure_opts: Optional[dict[str, Any]] = Field(
        {}, description="A dict of matplotlib figure options."
    )
    round: Optional[int] = Field(
        0, description="An integer to apply a mask that rounds the word cloud."
    )
    path: Optional[Path | str] = Field(
        None, description="The filepath to save the word cloud to."
    )
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

        # Set the mask, if using
        if self.round > 0:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > self.round**2
            mask = 255 * mask.astype(int)
            self.opts["mask"] = mask

        # Process the data into a consistent format
        data = self.data
        if isinstance(self.data, str):
            self.cloud = PythonWordCloud(**self.opts).generate_from_text(self.data)
        elif isinstance(self.data, (Doc | Span)):
            self.cloud = PythonWordCloud(**self.opts).generate_from_frequencies(
                Counter([t.text for t in self.data])
            )
        else:
            if isinstance(self.data, DTM):
                counts = processors.process_dtm(self.data, self.docs)
            elif isinstance(self.data, pd.DataFrame):
                counts = processors.process_self.dataframe(self.data, self.docs)
            elif isinstance(self.data, list) and isinstance(self.data[0], list):
                counts = processors.process_list(self.data, self.docs)
            elif isinstance(self.data, list) and isinstance(self.data[0], (Doc | Span)):
                counts = processors.process_docs(self.data, self.docs)
            elif isinstance(self.data, list) and not isinstance(self.data[0], list):
                counts = processors.process_item(self.data)
            elif isinstance(self.data, dict):
                counts = self.data
            else:
                raise LexosException(
                    "Cannot process data. Make sure that all items in the input belong to the same data type."
                )
            print("Counts:", counts)  # Debugging line to check counts
            self.cloud = PythonWordCloud(**self.opts).generate_from_frequencies(counts)

        # If a path is provided, save the WordCloud to that path
        if self.path:
            self.cloud.to_file(self.path)

        # Create matplotlib figure
        plt.ioff()  # Turn off interactive mode
        plt.axis("off")
        plt.imshow(self.cloud, interpolation="bilinear")
        self.fig = plt.figure(**self.figure_opts)
        if not self.showfig:
            plt.close("all")
        # if self.showfig:
        #     print("Showing figure...")  # Debugging line to indicate showing the figure
        #     plt.ion()  # Turn on interactive mode if showfig is True
        #     plt.show()
        #     # plt.close()
        #     return None
        # else:
        #     # plt.ioff()
        #     # plt.close()
        #     return None
        # print(
        #     "Returning figure..."
        # )  # Debugging line to indicate returning the figure
        # plt.close("all")
        # return self.fig

        # if self.showfig is True:
        #     plt.axis("off")
        #     plt.figure(self.fig)  # Set the current figure to self.fig
        #     plt.imshow(self.cloud, interpolation="bilinear")
        #     plt.show()
        #     # plt.close("all")  # Close the display but keep the figure object
        #     return None  # or None / return fig if you want the figure even when showing
        # else:
        #     plt.close("all")  # Close the display but keep the figure object
        #     return None  # Return the matplotlib figure

    def generate(
        self,
        data: Optional[single_doc_types | multi_doc_types | pd.DataFrame] = None,
        docs: Optional[int | str | list[int] | list[str]] = None,
        opts: Optional[dict[str, Any]] = None,
        showfig: Optional[bool] = True,
        figure_opts: Optional[dict[str, Any]] = None,
        round: Optional[int] = None,
        path: Optional[Path | str] = None,
    ) -> PythonWordCloud | None:
        """Make a word cloud.

        Returns:
            WordCloud: A Python WordCloud object if show is set to False or None.

        Notes:
            - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
            - If `show=False` the function returns a Python WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the Python  WordCloud documentation for a list of methods.
        """
        # Set the default options
        if opts is None:
            opts = {
                "background_color": "white",
                "max_words": 2000,
                "contour_width": 0,
                "contour_color": "steelblue",
            }
        self.opts = opts

        if figure_opts is None:
            figure_opts = {}
        self.figure_opts = figure_opts

        # Set the mask, if using
        if self.round > 0:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > self.round**2
            mask = 255 * mask.astype(int)
            opts["mask"] = mask

        # Process the data into a consistent format
        if not data:
            data = self.data
        if isinstance(data, str):
            self.cloud = PythonWordCloud(**opts).generate_from_text(data)
        elif isinstance(data, (Doc | Span)):
            self.cloud = PythonWordCloud(**opts).generate_from_frequencies(
                Counter([t.text for t in data])
            )
        else:
            if isinstance(data, DTM):
                counts = processors.process_dtm(data, docs)
            elif isinstance(data, pd.DataFrame):
                counts = processors.process_dataframe(data, docs)
            elif isinstance(data, list) and isinstance(data[0], list):
                counts = processors.process_list(data, docs)
            elif isinstance(data, list) and isinstance(data[0], (Doc | Span)):
                counts = processors.process_docs(data, docs)
            elif isinstance(data, list) and not isinstance(data[0], list):
                counts = processors.process_item(data)
            elif isinstance(data, dict):
                counts = data
            else:
                raise LexosException(
                    "Cannot process data. Make sure that all items in the input belong to the same data type."
                )
            self.cloud = PythonWordCloud(**opts).generate_from_frequencies(counts)

        # If a path is provided, save the WordCloud to that path
        if path:
            self.cloud.to_file(path)

        # Create matplotlib figure
        plt.ioff()
        plt.imshow(self.cloud)
        fig, _ = plt.subplots(**self.figure_opts)
        self.fig = fig

        # Show the figure if showfig is True
        if self.showfig:
            plt.ion()
            self.fig.show()

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
        plt.axis("off")
        plt.imshow(self.cloud)
        return self.fig


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def multicloud(
    data: str
    | list[str]
    | list[list[str]]
    | list[Doc]
    | list[Span]
    | list[list[Token]]
    | dict[str, int]
    | pd.DataFrame
    | DTM,
    docs: Optional[int | str | list[int] | list[str]] = None,
    opts: Optional[dict[str, Any]] = None,
    ncols: Optional[int] = 3,
    title: Optional[str] = None,
    labels: Optional[list[str]] = None,
    show: Optional[bool] = True,
    figure_opts: Optional[dict[str, Any]] = None,
    round: Optional[int] = None,
    filename: Optional[str] = None,
) -> object:
    """Make multiclouds.

    Returns:
        list[WordCloud]: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    # TODO: As with WordCloud
    if opts is None:
        opts = {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        }
    if figure_opts is None:
        figure_opts = {}

    # Create a rounded mask
    if round:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > round**2
        mask = 255 * mask.astype(int)
        opts["mask"] = mask

    # TODO: This is where we do the loop. We need to create WordCloud objects for each item in the data.
    if isinstance(data, list) and isinstance(data[0], str):
        clouds = [WordCloud(**opts).generate_from_text(item) for item in data]
    else:
        data = processors.multicloud_processor(data, docs)
        clouds = [WordCloud(**opts).generate_from_frequencies(item) for item in data]

    # List for multiple word clouds if they are to be returned.
    multiclouds = []

    # Constrain the layout
    figure_opts["constrained_layout"] = True

    # Create the figure
    fig = plt.figure(**figure_opts)

    # Add the title
    if title:
        fig.suptitle(title)

    # Calculate the number of rows and columns

    nrows = int(np.ceil(len(clouds) / ncols))
    spec = fig.add_gridspec(nrows, ncols)

    # Divide the data into rows
    rows = list(processors.get_rows(clouds, ncols))

    # Set an index for labels
    i = 0

    # Loop through the rows
    for row, doc in enumerate(rows):
        # Loop through the documents in the row
        for col, wordcloud in enumerate(doc):
            # Create a subplot
            ax = fig.add_subplot(spec[row, col])

            # If `show=True`, show the word cloud
            if show:
                ax.imshow(wordcloud)
                ax.axis("off")

                # Set the image title from the label
                if labels:
                    ax.set_title(labels[i])
                    i += 1

            # Otherwise, add the word cloud to the multiclouds list.
            else:
                multiclouds.append(wordcloud)

    # If a filename is provided, save the figure
    if filename:
        fig.savefig(filename)

    # If `show=False`, return the multiclouds list.
    if not show:
        return multiclouds
