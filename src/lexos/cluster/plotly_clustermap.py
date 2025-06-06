"""plotly_clustermap.py.

Last Updated: February 25, 2025
Last Tested: February 25, 2025

Typical usage:
    clustermap = PlotlyClustermap(dtm)
    clustermap.show()
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.colors as colors
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from numpy.typing import ArrayLike
from plotly.figure_factory import create_dendrogram
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.spatial.distance import pdist, squareform

from lexos.dtm import DTM
from lexos.exceptions import LexosException


class PlotlyClustermap(BaseModel):
    """PlotlyClustermap."""

    dtm: Optional[ArrayLike | DTM | pd.DataFrame] = Field(
        None, json_schema_extra={"The document-term matrix."}
    )
    labels: Optional[list[str]] = Field(
        None, json_schema_extra={"description": "The labels for the clustermap."}
    )
    metric: Optional[str] = Field(
        "euclidean",
        json_schema_extra={"description": "The metric to use for the dendrograms."},
    )
    method: Optional[str] = Field(
        "average",
        json_schema_extra={"description": "The method to use for the dendrograms."},
    )
    hide_upper: Optional[bool] = Field(
        False, json_schema_extra={"Hide the upper dendrogram."}
    )
    hide_side: Optional[bool] = Field(
        False, json_schema_extra={"Hide the side dendrogram."}
    )
    width: Optional[int] = Field(
        600, json_schema_extra={"The width for the clustermap."}
    )
    height: Optional[int] = Field(
        600, json_schema_extra={"The height for the clustermap."}
    )
    title: Optional[str] = Field(
        None, json_schema_extra={"description": "The title for the clustermap."}
    )
    showfig: Optional[bool] = Field(
        False,
        json_schema_extra={
            "description": "Whether to show the figure when the instance is called."
        },
    )
    colorscale: Optional[str] = Field(
        "Viridis",
        json_schema_extra={
            "description": "The colorscale for the heatmap portion of the clustermap. Can be a one of `Blackbody, Bluered, Blues, Cividis, Earth, Electric, Greens, Greys, Hot, Jet, Picnic, Portl and, Rainbow, RdBu, Reds, Viridis, YlGnBu, YlOrRd`."
        },
    )
    config: dict = dict(
        displaylogo=False,
        modeBarButtonsToRemove=["toImage", "toggleSpikelines"],
        scrollZoom=True,
    )
    fig: Optional[plt.Figure] = Field(
        None, json_schema_extra={"description": "The figure for the clustermap."}
    )
    layout: Optional[dict] = Field(
        {},
        json_schema_extra={
            "description": "The layout for the dendrogram. Keywords and values to be passed to plotly.graph_objects.Figure.update_layout()."
        },
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_call(config=model_config)
    def __call__(
        self,
        dtm: Optional[ArrayLike | DTM | pd.DataFrame] = None,
        labels: Optional[list[str]] = None,
        metric: Optional[str] = None,
        method: Optional[str] = None,
        title: Optional[str] = None,
        hide_upper: Optional[bool] = None,
        hide_side: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        showfig: Optional[bool] = None,
        colorscale: Optional[str] = None,
        config: Optional[dict] = None,
        layout: Optional[dict] = None,
    ):
        """Call the instance."""

        def distfun(x: ArrayLike) -> ArrayLike:
            """Get the pairwise distance matrix.

            Args:
                x (ArrayLike): The distance matrix.

            Returns:
                ArrayLike: The pairwise distance matrix.
            """
            return pdist(x, metric=self.metric)

        def linkagefun(x: ArrayLike) -> ArrayLike:
            """Get the hierarchical clustering encoded as a linkage matrix.

            Args:
                x (ArrayLike): The pairwise distance matrix.

            Returns:
                ArrayLike: The linkage matrix.
            """
            return sch.linkage(x, self.method)

        # Set the attributes of the class
        self._set_attrs(
            dtm=dtm,
            labels=labels,
            metric=metric,
            method=method,
            title=title,
            hide_upper=hide_upper,
            hide_side=hide_side,
            width=width,
            height=height,
            showfig=showfig,
            colorscale=colorscale,
            config=config,
            layout=layout,
        )

        # Ensure there is a document-term matrix
        if self.dtm is None:
            raise LexosException("You must provide a document-term matrix.")

        # Ensure there are labels
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.index.values.tolist()
            else:
                self.labels = [f"Doc{i + 1}" for i, _ in enumerate(self.dtm)]

        # Get the matrix based on the data type
        matrix = self._get_valid_matrix()

        # Create the figure
        fig = create_dendrogram(
            matrix,
            labels=self.labels,
            distfun=distfun,
            linkagefun=linkagefun,
            orientation="bottom",
            colorscale=colors.get_colorscale(self.colorscale),
            color_threshold=None,
        )

        for i in range(len(fig["data"])):
            fig["data"][i]["yaxis"] = "y2"

        # Renders the upper dendrogram invisible
        # Also removes the labels, so you have to rely on hovertext
        if self.hide_upper:
            fig.for_each_trace(lambda trace: trace.update(visible=False))

        # Create Side Dendrogram
        dendro_side = create_dendrogram(
            matrix,
            distfun=distfun,
            linkagefun=linkagefun,
            orientation="right",
            colorscale=colors.get_colorscale(self.colorscale),
            color_threshold=None,
        )
        for i in range(len(dendro_side["data"])):
            dendro_side["data"][i]["xaxis"] = "x2"

        # Add Side Dendrogram Data to Figure
        if not self.hide_side:
            for data in dendro_side["data"]:
                fig.add_trace(data)

        # Create Heatmap
        dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
        dendro_leaves = list(map(int, dendro_leaves))
        data_dist = pdist(matrix)
        heat_data = squareform(data_dist)
        heat_data = heat_data[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]

        num = len(self.labels)
        heatmap = [
            go.Heatmap(
                x=dendro_leaves,
                y=dendro_leaves,
                z=heat_data,
                colorscale=self.colorscale,
                hovertemplate="X: %{x}<br>Y: %{customdata}<br>Z: %{z}<extra></extra>",
                customdata=[[label for x in range(num)] for label in self.labels],
            )
        ]

        heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
        heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)

        # Edit Layout
        fig.update_layout(
            {
                "width": self.width,
                "height": self.height,
                "showlegend": False,
                "hovermode": "closest",
            }
        )

        # Edit xaxis (dendrogram)
        if not self.hide_side:
            x = 0.15
        else:
            x = 0
        fig.update_layout(
            xaxis={
                "domain": [x, 1],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "ticks": "",
            }
        )
        # Edit xaxis2 (heatmap)
        fig.update_layout(
            xaxis2={
                "domain": [0, 0.15],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            }
        )

        # Edit yaxis (heatmap)
        fig.update_layout(
            yaxis={
                "domain": [0, 0.85],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            }
        )
        # Edit yaxis2 (dendrogram)
        fig.update_layout(
            yaxis2={
                "domain": [0.840, 0.975],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            }
        )

        fig.update_layout(
            margin=dict(l=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickfont=dict(color="rgba(0,0,0,0)"),
        )

        # Set the title
        if self.title:
            title = dict(
                text=self.title, x=0.5, y=0.95, xanchor="center", yanchor="top"
            )
            fig.update_layout(title=title, margin=dict(t=40))

        # Save the figure variable
        self.fig = fig

        # Show the plot
        if self.showfig:
            self.fig.show(config=self.config)

    def _get_valid_matrix(self):
        """Get a valid matrix based on the data type of the dtm."""
        if isinstance(self.dtm, DTM):
            matrix = self.dtm.to_df()
            matrix.index.name = "terms"
            matrix = matrix.T
        else:
            matrix = self.dtm
        if isinstance(matrix, list):
            first_row = len(matrix[0])
            first_row = len(matrix)
        else:
            first_row = matrix.shape[0]
        if first_row < 2:
            raise LexosException(
                "The document-term matrix must have more than one document."
            )
        return matrix

    def _set_attrs(self, **kwargs):
        """Set the attributes of the class."""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def show(self):
        """Show the figure."""
        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        self.fig.show(config=self.config)

    def to_html(self, **kwargs):
        """Create an HTML representation of the figure.

        Wrapper from the Plotly Figure to_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        if self.fig is None:
            raise LexosException("You must call the instance before generating HTML.")
        return self.fig.to_html(**kwargs)

    def to_image(self, **kwargs):
        """Create a static image of the figure.

        Wrapper from the Plotly Figure to_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        if self.fig is None:
            raise LexosException(
                "You must call the instance before generating an image."
            )
        return self.fig.to_image(**kwargs)

    @validate_call(config=model_config)
    def write_html(self, path: Path | str, **kwargs):
        """Save an HTML representation of the figure to disk.

        Wrapper from the Plotly Figure write_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        if self.fig is None:
            raise LexosException("You must call the instance before saving the figure.")
        if "file" in kwargs:
            kwargs["file"] = path
        return self.fig.write_html(**kwargs)

    @validate_call(config=model_config)
    def write_image(self, path: Path | str, **kwargs):
        """Save a static image of the figure to disk.

        Wrapper from the Plotly Figure write_image method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        if self.fig is None:
            raise LexosException("You must call the instance before saving the figure.")
        if "file" in kwargs:
            kwargs["file"] = path
        return self.fig.write_image(**kwargs)
