"""plotly_wordcloud.py.

Last Update: March 3, 2025
Last Tested: March 4, 2025

This module could possibly be better implemented as a class, rather than as
a function. That would allow calling the various Plotly graph_objects write
methods, rather than making write_html() a default format to save. However,
since the function returns aseparate call. However, since the function returns
a Plotly graph_objects Figure, you can then call any write method on that.
"""

from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
from pydantic import ConfigDict, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from wordcloud import WordCloud

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


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def plotly_wordcloud(
    data: single_doc_types | multi_doc_types | pd.DataFrame,
    docs: Optional[int | str | list[int] | list[str]] = None,
    opts: Optional[dict[str, Any]] = None,
    layout: (Optional[dict[str, Any]]) = None,
    show: Optional[bool] = True,
    round: Optional[int] = None,
    path: Optional[Path | str] = None,
) -> go.Figure:
    """Make a Plotly word cloud.

    Args:
        data (single_doc_types | multi_doc_types): The data.
        A single_doc_type can be a dict with terms as keys and counts as values, a Doc, a Span, a string, a list of strings, or a list of Tokens. A multi_doc_type can be a DTM, a DataFrame, a list of Docs, Spans or term-count dicts, or a list of lists of strings or Tokens. Dataframes must have terms as the index and documents as the columns.
        docs (Optional[int | str | list[int] | list[str]]): A list of documents to be selected from the DTM. Values can be either indices or labels.
        opts (Optional[dict[str, Any]): The WordCloud() options.
        layout: (Optional[dict[str, Any]]): A dict of options to pass to Plotly.
        show (Optional[bool]): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (Optional[dict[str, Any]]): A dict of matplotlib figure options.
        round (Optional[int]): An integer (generally between 100-300) to apply a mask that rounds the word cloud.
        path (Optional[Path | str]): The filepath to save the word cloud to.

    Returns:
        go.Figure: A Plotly graph_objects Figure.

    Notes:
        - Accepts data from a string, list of lists or tuples, a dict with terms as keys and counts/frequencies as values, or a dataframe.
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`. This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the WordCloud documentation for a list of methods.
        - This is some prototype code for generating word clouds in Plotly based on https://github.com/PrashantSaikia/Wordcloud-in-Plotly. This is really a case study because Plotly does not do good word clouds. One of the limitations is that `WordCloud.layout_` always returns `None` for orientation and frequencies for counts. That limits the options for replicating its output.
    """
    # Set the default WordCloud options
    if opts is None:
        opts = {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 3,
            "contour_color": "steelblue",
        }
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

    # Convert the WordCloud object to a Plotly graph_objects Figure
    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []
    layout_opts = {
        "xaxis": {"showgrid": False, "showticklabels": False, "zeroline": False},
        "yaxis": {"showgrid": False, "showticklabels": False, "zeroline": False},
        "autosize": False,
        "width": 750,
        "height": 750,
        "margin": {"l": 50, "r": 50, "b": 100, "t": 100, "pad": 4},
    }
    if layout:
        for k, v in layout.items():
            layout_opts[k] = v

    # Plot the word cloud
    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # Get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # Get the relative occurence frequencies
    new_freq_list = [f"{Decimal(str(n*100)):.2f}%" for n in freq_list]

    trace = go.Scatter(
        x=x,
        y=y,
        textfont=dict(size=fontsize_list, color=color_list),
        hoverinfo="text",
        hovertext=[f"{w}: {f}" for w, f in zip(word_list, new_freq_list)],
        mode="text",
        text=word_list,
    )

    # Set the layout and create the figure
    layout = go.Layout(layout_opts)
    fig = go.Figure(data=[trace], layout=layout)

    # Save the figure to disk
    if path:
        fig.write_html(path)

    # Show the plot and/or return the figure
    if show:
        fig.show()
        return fig
    else:
        return fig
