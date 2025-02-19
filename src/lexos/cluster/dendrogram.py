"""dendrogram.py.

Last Updated: February 18, 2025
Last Tested: February 18, 2025
"""

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.spatial.distance import pdist

from lexos.dtm import DTM
from lexos.exceptions import LexosException


class Dendrogram(BaseModel):
    """Dendrogram.

    Typical usage:

    ```python
    from lexos.cluster.dendrogram import Dendrogram

    dendrogram = Dendrogram(dtm, show=True)

    or

    dendrogram = Dendrogram(dtm, show=False)
    dendrogram.fig
    ```

    The dtm parameter can be a a DTM instance or a pandas DataFrame with terms
    as columns and docs as rows (the output of `DTM.to_df(transpose=True)`).
    It can also be an equivalent numpy array or list of lists. But in most cases,
    it will be most convenient to use a DTM instance.
    """

    dtm: Optional[ArrayLike | DTM | pd.DataFrame] = Field(
        None, json_schema_extra={"The document-term matrix."}
    )
    labels: Optional[list[str]] = Field(
        None, json_schema_extra={"description": "The labels for the dendrogram."}
    )
    metric: Optional[str] = Field(
        "euclidean",
        json_schema_extra={"description": "The metric to use for the dendrogram."},
    )
    method: Optional[str] = Field(
        "average",
        json_schema_extra={"description": "The method to use for the dendrogram."},
    )
    truncate_mode: Optional[str] = Field(
        None, json_schema_extra={"description": "The truncate mode for the dendrogram."}
    )
    color_threshold: Optional[str] = Field(
        None,
        json_schema_extra={"description": "The color threshold for the dendrogram."},
    )
    get_leaves: Optional[bool] = Field(
        True, json_schema_extra={"description": "The get leaves for the dendrogram."}
    )
    orientation: Optional[str] = Field(
        "top", json_schema_extra={"description": "The orientation for the dendrogram."}
    )
    count_sort: Optional[bool | str] = Field(
        None, json_schema_extra={"description": "The count sort for the dendrogram."}
    )
    distance_sort: Optional[bool | str] = Field(
        None, json_schema_extra={"description": "The distance sort for the dendrogram."}
    )
    show_leaf_counts: Optional[bool] = Field(
        False,
        json_schema_extra={"description": "The show leaf counts for the dendrogram."},
    )
    no_plot: Optional[bool] = Field(
        False, json_schema_extra={"description": "The no plot for the dendrogram."}
    )
    no_labels: Optional[bool] = Field(
        False, json_schema_extra={"description": "The no labels for the dendrogram."}
    )
    leaf_rotation: Optional[int] = Field(
        90, json_schema_extra={"description": "The leaf rotation for the dendrogram."}
    )
    leaf_font_size: Optional[int] = Field(
        None,
        json_schema_extra={"description": "The leaf font size for the dendrogram."},
    )
    leaf_label_func: Optional[Callable] = Field(
        None,
        json_schema_extra={
            "description": "The leaf label function for the dendrogram."
        },
    )
    show_contracted: Optional[bool] = Field(
        False,
        json_schema_extra={"description": "The show contracted for the dendrogram."},
    )
    link_color_func: Optional[Callable] = Field(
        None,
        json_schema_extra={
            "description": "The link color function for the dendrogram."
        },
    )
    ax: Optional[Axes] = Field(
        None, json_schema_extra={"description": "The ax for the dendrogram."}
    )
    above_threshold_color: Optional[str] = Field(
        "C0",
        json_schema_extra={
            "description": "The above threshold color for the dendrogram."
        },
    )
    title: Optional[str] = Field(
        None, json_schema_extra={"description": "The title for the dendrogram."}
    )
    figsize: Optional[tuple] = Field(
        (10, 10), json_schema_extra={"description": "The figsize for the dendrogram."}
    )
    show: Optional[bool] = Field(
        False, json_schema_extra={"description": "The show for the dendrogram."}
    )
    fig: Optional[plt.Figure] = Field(
        None, json_schema_extra={"description": "The figure for the dendrogram."}
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, **kwargs):
        """Call the instance."""
        # Set the attributes of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensure there is a document-term matrix with more than one document
        if self.dtm is None:
            raise LexosException("You must provide a document-term matrix.")

        # Ensure there are labels
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.columns.values.tolist()
            else:
                self.labels = [f"Doc{i + 1}" for i, _ in enumerate(self.dtm)]

        # Get the matrix based on the data type
        matrix = self._get_valid_matrix()

        # Generate the pairwise distance and linkage matrices
        X = pdist(matrix, metric=self.metric)
        Z = sch.linkage(X, self.method)

        # Generate the dendrogram
        fig, ax = plt.subplots(figsize=self.figsize)
        if self.title:
            plt.title(self.title)
        sch.dendrogram(
            Z,
            labels=self.labels,
            truncate_mode=self.truncate_mode,
            color_threshold=self.color_threshold,
            get_leaves=self.get_leaves,
            orientation=self.orientation,
            count_sort=self.count_sort,
            distance_sort=self.distance_sort,
            show_leaf_counts=self.show_leaf_counts,
            no_plot=self.no_plot,
            no_labels=self.no_labels,
            leaf_rotation=self.leaf_rotation,
            leaf_font_size=self.leaf_font_size,
            leaf_label_func=self.leaf_label_func,
            show_contracted=self.show_contracted,
            link_color_func=self.link_color_func,
            ax=self.ax,
            above_threshold_color=self.above_threshold_color,
        )
        self.fig = fig

        if not self.show:
            plt.close()

    def _get_valid_matrix(self):
        """Get a valid matrix based on the data type of the dtm."""
        error_msg = "The document-term matrix must have more than one document."
        if isinstance(self.dtm, pd.DataFrame) and self.dtm.shape[0] < 3:
            raise LexosException(error_msg)
        elif self.dtm.shape[0] < 2:
            raise LexosException(error_msg)
        if isinstance(self.dtm, DTM):
            df = self.dtm.to_df()
            df.index.name = "terms"
            return df.T
        return self.dtm

    @validate_call
    def save(self, path: Path | str):
        """Save the figure as a file.

        Args:
            path (Path | str): The path to the file to save.
        """
        if not path or path == "":
            raise LexosException("You must provide a valid path.")
        self.fig.savefig(path)

    def showfig(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `Dendrogram.fig`. This will generally display in a
        Jupyter notebook.
        """
        return self.fig
