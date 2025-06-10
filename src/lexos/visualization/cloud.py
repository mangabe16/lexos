"""wordcloud.py.

Last Update: March 1, 2025
Last Tested: March 1, 2025
"""

from collections import Counter
from pathlib import Path
from typing import Any, Optional


import matplotlib
matplotlib.use("Agg")  # additional line
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from wordcloud import WordCloud

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization import processors

# Valid input types
single_doc_types = dict[str, int] | Doc | Span | str | list[str] | list[Token]
multi_doc_types = str | list[str] | list[list[str]] | list[Doc] | list[Span] | list[list[Token]] | dict[str, int] | pd.DataFrame | DTM

@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def wordcloud(
    data: single_doc_types | multi_doc_types | pd.DataFrame,
    docs: Optional[int | str | list[int] | list[str]] = None,
    opts: Optional[dict[str, Any]] = None,
    show: Optional[bool] = True,
    figure_opts: Optional[dict[str, Any]] = None,
    round: Optional[int] = None,
    path: Optional[Path | str] = None,
) -> WordCloud:
    """Make a word cloud.

    Accepts data from a string, list of lists or tuples, a dict with
    terms as keys and counts/frequencies as values, or a dataframe.

    Args:
        data (single_doc_types | multi_doc_types): The data.
        A single_doc_type can be a dict with terms as keys and counts as values, a Doc, a Span, a string, a list of strings, or a list of Tokens. A multi_doc_type can be a DTM, a DataFrame, a list of Docs, Spans or term-count dicts, or a list of lists of strings or Tokens. Dataframes must have terms as the index and documents as the columns.
        docs (Optional[int | str | list[int] | list[str]]): A list of documents to be selected from the DTM. Values can be either indices or labels.
        opts (Optional[dict[str, Any]): The WordCloud() options.
            For testing, try {"background_color": "white", "max_words": 2000, "contour_width": 3, "contour_color": "steelblue"}
        show (Optional[bool]): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (Optional[dict[str, Any]]): A dict of matplotlib figure options.
        round (Optional[int]): An integer (generally between 100-300) to apply a mask that rounds the word cloud.
        path (Optional[Path | str]): The filepath to save the word cloud to.

    Returns:
        WordCloud: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    # Set the default options
    if opts is None:
        opts = {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        }
    if figure_opts is None:
        figure_opts = {}

    # Set the mask, if using
    if round:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > round**2
        mask = 255 * mask.astype(int)
        opts["mask"] = mask

    # Process the data into a consistent format
    if isinstance(data, str):
        wc = WordCloud(**opts).generate_from_text(data)
    elif isinstance(data, (Doc | Span)):
        wc = WordCloud(**opts).generate_from_frequencies(
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
        wc = WordCloud(**opts).generate_from_frequencies(counts)

    # Plot or return the WordCloud
    # Why is it only if show=true? what if the user wants show=false? We can still save the figure, just dont show it by plotting it
    if show:
        if figure_opts:
            plt.figure(**figure_opts)
        plt.axis("off")
        # If a filepath is provided, save the figure
        if path:
            wc.to_file(path)
        plt.imshow(wc)
        plt.show()
    else:
        if path:
            wc.to_file(path)
        return wc


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def multicloud(
    data: str | list[str] | list[list[str]] | list[Doc] | list[Span] | list[list[Token]] | dict[str, int] | pd.DataFrame | DTM,
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

    Accepts data from a string, list of lists, a dict with
    terms as keys and counts/frequencies as values, or a dataframe.

    The best input is a dtm produced by `get_dtm_table()`.

    Args:
        data (multi_doc_types): The data.
            A multi_doc_type can be a DTM, a DataFrame, a list of Docs, Spans or term-count dicts, or a list of lists of strings or Tokens. Dataframes must have terms as the index and documents as the columns.
            as values, or a dataframe with "term" and "count" columns.
        docs: (Optional[int | str | list[int] | list[str]]): A list of documents to be selected from the DTM.
        opts (Optional[dict[str, Any]]): The WordCloud() options.
            For testing, try {"background_color": "white", "max_words": 2000, "contour_width": 3, "contour_color": "steelblue"}
        ncols (Optional[int]): The number of columns in the grid.
        title (Optional[str]): The title of the grid.
        labels (Optional[list[str]]): The document labels for each subplot.
        show (Optional[bool]): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (Optional[dict[str, Any]]): A dict of matplotlib figure options.
        round (Optional[int]): An integer (generally between 100-300) to apply a mask that rounds the word cloud.
        filename (Optional[str]): The filename to save the figure to.

    Returns:
        list[WordCloud]: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
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
