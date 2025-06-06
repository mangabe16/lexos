"""corpus_stats.py.

Last updated: June 5, 2025
Last tested: TBD.
"""

from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from plotly import express as px
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.tokens import Doc

from lexos.dtm import DTM


class CorpusStats(BaseModel):
    """A class to hold statistics about a Corpus.

    The input should be a list of tuples, where each tuple contains:
        - id: A unique identifier for the document.
        - label: A label for the document.
        - token list: A list of tokens in the document. Tokens can be words, n-grams, or any other token unit.
        - Settings to pass to the DTM vectorizer, such as min_df, max_df, and max_n_terms.

    To reproduce the webapp:

      - stats = CorpusStats(docs=docs)
      - stats.doc_stats_df # The DataFrame containing document statistics.
      - stats.mean # The mean count for the entire corpus.
      - stats.standard_deviation # The standard deviation for the entire corpus.
      - stats.get_iqr_outliers() # Get outliers based on interquartile range (IQR).
      - stats.get_std_outliers() # Get outliers based on standard deviation.
      - stats.plot(column="total_tokens", type="plotly_boxplot" title="Corpus Boxplot") # Plot the boxplot of total tokens with Plotly.
    """

    docs: list[tuple[str, str, list[str]]]
    min_df: int | None = None
    max_df: int | None = None
    max_n_terms: int | None = None
    dtm: DTM = Field(
        default_factory=DTM(), description="Document-Term Matrix (DTM) for the Corpus."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the CorpusStats and create the DTM."""
        super().__init__(**data)
        # Separate the ids and labels from the docs
        self.ids = [doc[0] for doc in self.docs]
        self.labels = [doc[0] for doc in self.docs]

        # Configure the DTM vectorizer with the provided settings
        self.dtm.vectorizer.min_df = (self.min_df,)
        self.dtm.vectorizer.max_df = (self.max_df,)
        self.dtm.vectorizer.max_n_terms = (self.max_n_terms,)

        # Create the Document-Term Matrix (DTM) using the provided token lists
        self.dtm(docs=[doc[2] for doc in self.docs], labels=self.labels)

    @property
    def df(self) -> pd.DataFrame:
        """Get the Document-Term Matrix (DTM) in sparse format."""
        return self.dtm.to_df()

    @cached_property
    def doc_stats_df(self) -> pd.DataFrame:
        """Get a Pandas dataframe containing the statistics of each document.

        Returns:
            A Pandas dataframe containing statistics of each document.
        """
        return self._get_doc_stats_df()

    @cached_property
    def mean_and_spread(self) -> tuple[float, float]:
        """Get the mean and standard deviation of the total tokens in the Corpus."""
        df = self.df.sparse.to_dense().T
        df["Total"] = df.sum(axis=1)
        return df["Total"].mean(), df["Total"].std()

    @property
    def mean(self) -> float:
        """Get the mean of the total tokens in the Corpus."""
        return self.mean_and_spread[0]

    @property
    def standard_deviation(self) -> float:
        """Get the standard deviation of the total tokens in the Corpus."""
        return self.mean_and_spread[1]

    def _get_doc_stats_df(self) -> pd.DataFrame:
        """Get a Pandas dataframe containing the statistics of each document.

        Args:
            df: A DTM.to_df() dataframe.
            names: A list of document names corresponding to the rows in the dataframe.

        Returns:
            A Pandas dataframe containing statistics of each document.
        """
        # Check if empty corpus is given.
        if self.df.empty:
            raise ValueError(
                "The DataFrame is empty. Please provide a valid DataFrame."
            )

        # Convert the DataFrame to dense format and transpose to docs are rows.
        df = self.dtm.to_df().sparse.to_dense().T

        # Create file_stats DataFrame
        file_stats = pd.DataFrame(self.labels, columns=["Documents"])
        file_stats.set_index("Documents", inplace=True)

        # Count terms appearing exactly once in each document
        file_stats[f"hapax_legomena"] = df.eq(1).sum(axis=1)

        # Calculate total tokens in each document
        file_stats["total_tokens"] = df.sum(axis=1)

        # Number of distinct terms in each document
        file_stats["total_terms"] = df.ne(0).sum(axis=1)

        # Calculate vocabulary density
        file_stats["vocabulary_density"] = (
            file_stats["total_terms"] / file_stats["total_tokens"] * 100
        ).round(2)

        return file_stats

    def get_iqr_outliers(self) -> list[tuple[str, str, int]]:
        """Get the interquartile range (IQR) outliers in the Corpus.

        Returns:
            list[tuple[str, str, int]]: A list of tuples containing the document ID,
            document name, and document length for each outlier.
        """
        # Get doc lengths from the doc_stats_df
        doc_lengths = np.array(self.doc_stats_df["total_tokens"].values.tolist())

        # Convert to DataFrame for easier calculations
        df = pd.DataFrame([self.ids, self.labels, doc_lengths]).fillna(0.0)
        q1 = df[2].quantile(0.25)
        q3 = df[2].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        self.iqr = iqr  # Store the IQR for later access
        return [
            (self.ids[i], self.labels[i])
            for i, length in enumerate(doc_lengths)
            if length < lower_bound or length > upper_bound
        ]

    def get_std_outliers(self) -> list[tuple[str, str, int]]:
        """Get the standard deviation outliers in the Corpus.

        Returns:
            list[tuple[str, str, int]]: A list of tuples containing the document ID,
            document name, and document length for each outlier.
        """
        # Get doc lengths from the doc_stats_df
        doc_lengths = np.array(self.doc_stats_df["total_tokens"].values.tolist())

        # Convert to DataFrame for easier calculations
        df = pd.DataFrame([self.ids, self.labels, doc_lengths]).fillna(0.0)
        mean = df[2].mean()
        std_dev = df[2].std()
        return [
            (self.ids[i], self.labels[i])
            for i, length in enumerate(doc_lengths)
            if length < mean - 2 * std_dev or length > mean + 2 * std_dev
        ]

    @validate_call(config=model_config)
    def plot(
        self,
        column: str = "token_lengths",
        type: str = "seaborn_boxplot",
        title: str = None,
    ) -> None:
        """Generate a plot of the Corpus.

        Args:
            column: The column to plot from the doc_stats_df.
            type: The type of plot to generate. Currently only "seaborn_boxplot" and "plotly_boxplot" are supported.
            title: The title of the plot. If None, the plotting function's default is used.
        """
        supported_types = ["seaborn_boxplot", "plotly_boxplot"]
        if type not in supported_types:
            raise ValueError(
                f"Unsupported plot type: {type}. The following types are supported: {supported_types.split(', ')}."
            )
        if type == "seaborn":
            get_seaborn_boxplot(self.doc_stats_df, column=column, title=title)
        elif type == "plotly":
            get_plotly_boxplot(self.doc_stats_df, column=column, title=title)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_seaborn_boxplot(
    df: pd.DataFrame, column: str, title: str = "Corpus Boxplot"
) -> None:
    """Get a boxplot of the specified column in the DataFrame.

    Args:
        df: A Pandas DataFrame.
        column: The column to plot.
        title: The title of the plot.
    """
    sns.set_theme(style="darkgrid")
    ax = sns.boxplot(y=df[column], width=0.25)
    sns.swarmplot(y=column, data=df, color="black", ax=ax)
    ax.set_title(title)
    plt.show()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_plotly_boxplot(
    df: pd.DataFrame, column: str, title: str = "Corpus Boxplot"
) -> None:
    """Get a boxplot of the specified column in the DataFrame using Plotly.

    Args:
        df: A Pandas DataFrame.
        column: The column to plot.
        title: The title of the plot.
    """
    # Get file names.
    labels = df.index.tolist()

    # Set up the points.
    scatter_plot = go.Scatter(
        x=labels,
        y=df[column].values,
        hoverinfo="text",
        mode="markers",
        marker=dict(color="green"),
        text=labels,
    )

    # Set up the box plot.
    box_plot = go.Box(
        x0=0,  # Initial position of the box plot
        y=df[column].values,
        hoverinfo="y",
        marker=dict(color="green"),
        jitter=0.15,
    )

    # Create a figure with two subplots and fill the figure.
    figure = make_subplots(rows=1, cols=2, shared_yaxes=False)
    figure.append_trace(trace=scatter_plot, row=1, col=1)
    figure.append_trace(trace=box_plot, row=1, col=2)

    # Hide useless information on x-axis and set up title.
    figure.layout.update(
        title={
            "text": title,
            "x": 0.5,  # x position (0-1)
            "xanchor": "center",  # Horizontal alignment
            "y": 0.99,  # y position (0-1)
            "yanchor": "top",  # Vertical alignment
        },
        height=300,
        width=500,
        dragmode="pan",
        showlegend=False,
        margin=dict(r=0, b=30, t=15, pad=4),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showline=False,
            zeroline=False,
            gridcolor="black",
            title="Total Tokens",
        ),
        xaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=dict(
            showline=False,
            zeroline=False,
            gridcolor="black",
        ),
        hovermode="closest",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="black", size=14),
    )

    # Show the Plotly figure.
    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["toImage", "toggleSpikelines"],
        "scrollZoom": True,
    }
    figure.show(showlink=False, config=config)
