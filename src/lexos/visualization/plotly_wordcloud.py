"""plotly_wordcloud.py.

Last Update: July 29, 2025
Last Tested: March 4, 2025

The plotly_wordcloud function has been re-implemented as a class. However,
it may need to have some extra save methods and an implementation for
multiclouds.
"""

from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
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
        wc = PythonWordCloud(**opts).generate_from_text(data)
    elif isinstance(data, (Doc | Span)):
        wc = PythonWordCloud(**opts).generate_from_frequencies(
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
        wc = PythonWordCloud(**opts).generate_from_frequencies(counts)

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
    new_freq_list = [f"{Decimal(str(n * 100)):.2f}%" for n in freq_list]

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


class PlotlyWordcloud(BaseModel):
    """A Pydantic model for WordCloud options."""

    data: single_doc_types | multi_doc_types | pd.DataFrame = Field(
        ...,
        description="The data to generate the word cloud from. Accepts data from a string, list of lists or tuples, a dict with terms as keys and counts/frequencies as values, or a dataframe.",
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM."
    )
    layout: Optional[dict[str, Any]] = {}
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
    round: Optional[int] = Field(
        0,
        description="An integer to apply a mask that rounds the word cloud. It is best to use 100 or higher for a circular mask, but it will depend on the height and width of the word cloud.",
    )
    cloud: PythonWordCloud | None = Field(
        None, description="The generated WordCloud object."
    )
    fig: Optional[go.Figure] = Field(
        None, description="The Plotly figure object for the word cloud."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the WordCloud model."""
        super().__init__(**data)

        # Set the default WordCloud options
        self.opts = {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 3,
            "contour_color": "steelblue",
            "width": self.width,
            "height": self.height,
        }

        # Process the data into a consistent format
        if isinstance(self.data, str):
            wc = PythonWordCloud(**self.opts).generate_from_text(self.data)
        elif isinstance(self.data, (Doc | Span)):
            wc = PythonWordCloud(**self.opts).generate_from_frequencies(
                Counter([t.text for t in self.data])
            )
        else:
            if isinstance(self.data, DTM):
                counts = processors.process_dtm(self.data, self.docs)
            elif isinstance(self.data, pd.DataFrame):
                counts = processors.process_dataframe(self.data, self.docs)
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
            wc = PythonWordCloud(**self.opts).generate_from_frequencies(counts)

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
        for k, v in self.layout.items():
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
        new_freq_list = [f"{Decimal(str(n * 100)):.2f}%" for n in freq_list]

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
        self.fig = go.Figure(data=[trace], layout=layout)

    @validate_call
    def save(self, path: Path | str) -> None:
        """Save the word cloud figure."""
        self.fig.write_image(path)

    def show(self) -> go.Figure:
        """Show the word cloud figure."""
        return self.fig
