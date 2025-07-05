"""clustermap.py.

Last Updated: July 5, 2025
Last Tested: February 27, 2025
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.cluster.hierarchy import is_valid_linkage

from lexos.dtm import DTM
from lexos.exceptions import LexosException

sns.set_theme()


class ClusterMap(BaseModel):
    """ClusterMap."""

    dtm: Optional[ArrayLike | DTM | pd.DataFrame] = Field(
        None, description="The document-term matrix."
    )
    labels: Optional[list[str]] = Field(
        None, description="The labels for the clustermap."
    )
    metric: Optional[str] = Field(
        "euclidean",
        description="The metric to use for the dendrograms.",
    )
    method: Optional[str] = Field(
        "average",
        description="The method to use for the dendrograms.",
    )
    hide_upper: Optional[bool] = Field(False, description="Hide the upper dendrogram.")
    hide_side: Optional[bool] = Field(False, description="Hide the side dendrogram.")
    title: Optional[str] = Field(None, description="The title for the dendrogram.")
    showfig: Optional[bool] = Field(
        False, description="Whether to show the figure when the instance is called."
    )
    fig: Optional[matplotlib.figure.Figure] = Field(
        None, description="The figure for the dendrogram."
    )
    z_score: Optional[int] = Field(1, description="The z-score for the clustermap.")
    pivot_kws: Optional[dict[str, str]] = Field(
        None, description="The pivot kwargs for the clustermap."
    )
    standard_scale: Optional[int] = Field(
        None,
        description="The standard scale for the clustermap.",
    )
    figsize: Optional[tuple[int, int]] = Field(
        (8, 8), description="The figure size for the clustermap."
    )
    cbar_kws: Optional[dict] = Field(
        None, description="The cbar kwargs for the clustermap."
    )
    row_cluster: Optional[bool] = Field(
        True, description="Whether to cluster the rows."
    )
    col_cluster: Optional[bool] = Field(
        True, description="Whether to cluster the columns."
    )
    row_linkage: Optional[np.ndarray] = Field(
        None,
        description="Precomputed linkage matrix for the rows. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage for specific formats.",
    )
    col_linkage: Optional[np.ndarray] = Field(
        None,
        description="Precomputed linkage matrix for the columns. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage for specific formats.",
    )
    row_colors: Optional[list | pd.DataFrame | pd.Series | str | ListedColormap] = (
        Field(None, description="The row colors.")
    )
    col_colors: Optional[list | pd.DataFrame | pd.Series | str | ListedColormap] = (
        Field(None, description="The column colors.")
    )
    mask: Optional[np.ndarray | pd.DataFrame] = Field(
        None, description="The mask for the clustermap."
    )
    dendrogram_ratio: Optional[float | tuple[float, float]] = Field(
        (0.1, 0.2),
        description="The dendrogram ratio for the clustermap.",
    )
    colors_ratio: Optional[float] = Field(
        0.03, description="The colors ratio for the clustermap."
    )
    cbar_pos: Optional[tuple[str]] = Field(
        (0.02, 0.32, 0.03, 0.2),
        description="The cbar position for the clustermap.",
    )
    tree_kws: Optional[dict] = Field(
        None, description="The tree kwargs for the dendrograms."
    )
    center: Optional[int] = Field(0, description="The center for the clustermap.")
    cmap: Optional[str] = Field("vlag", description="The cmap for the clustermap.")
    linewidths: Optional[float] = Field(
        0.75, description="The linewidths for the dendrograms."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_call(config=model_config)
    def __call__(
        self,
        dtm: Optional[ArrayLike | DTM | pd.DataFrame] = None,
        labels: Optional[list[str]] = None,
        metric: Optional[str] = None,
        method: Optional[str] = None,
        hide_upper: Optional[bool] = None,
        hide_side: Optional[bool] = None,
        title: Optional[str] = None,
        showfig: Optional[bool] = None,
        z_score: Optional[int] = None,
        pivot_kws: Optional[dict[str, str]] = None,
        standard_scale: Optional[int] = None,
        figsize: Optional[tuple[int, int]] = None,
        cbar_kws: Optional[dict] = None,
        row_cluster: Optional[bool] = None,
        col_cluster: Optional[bool] = None,
        row_linkage: Optional[np.ndarray] = None,
        col_linkage: Optional[np.ndarray] = None,
        row_colors: Optional[
            list | pd.DataFrame | pd.Series | str | ListedColormap
        ] = None,
        col_colors: Optional[
            list | pd.DataFrame | pd.Series | str | ListedColormap
        ] = None,
        mask: Optional[np.ndarray | pd.DataFrame] = None,
        dendrogram_ratio: Optional[float | tuple[float, float]] = None,
        colors_ratio: Optional[float] = None,
        cbar_pos: Optional[tuple[str]] = None,
        tree_kws: Optional[dict] = None,
        center: Optional[int] = None,
        cmap: Optional[str] = None,
        linewidths: Optional[float] = None,
    ):
        """Call the ClusterMap instance."""
        # Set the attributes of the class
        self._set_attrs(
            dtm=dtm,
            labels=labels,
            metric=metric,
            method=method,
            title=title,
            hide_upper=hide_upper,
            hide_side=hide_side,
            showfig=showfig,
            z_score=z_score,
            pivot_kws=pivot_kws,
            standard_scale=standard_scale,
            figsize=figsize,
            cbar_kws=cbar_kws,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            row_colors=row_colors,
            col_colors=col_colors,
            mask=mask,
            dendrogram_ratio=dendrogram_ratio,
            colors_ratio=colors_ratio,
            cbar_pos=cbar_pos,
            tree_kws=tree_kws,
            center=center,
            cmap=cmap,
            linewidths=linewidths,
        )

        """Build the clustermap."""
        # Ensure there is a document-term matrix
        if self.dtm is None:
            raise LexosException("You must provide a document-term matrix.")

        # Ensure there are labels
        self._set_labels()

        # Get the matrix based on the data type
        matrix = self._get_valid_matrix()

        # Get colour palettes for the dendrograms
        col_colors, row_colors = self._get_colors()

        # Ensure that the matrix is not a sparse array
        if self.row_linkage is None and self.col_linkage is None:
            matrix = matrix.sparse.to_dense()
        else:
            matrix = pd.DataFrame(matrix, index=self.labels)

        # Validate the linkage matrices
        self._validate_linkage_matrices()

        # Perform the clustering
        g = sns.clustermap(
            matrix,
            cmap=self.cmap,
            method=self.method,
            metric=self.metric,
            figsize=self.figsize,
            col_colors=col_colors,
            row_colors=row_colors,
            center=self.center,
            linewidths=self.linewidths,
            z_score=self.z_score,
            pivot_kws=self.pivot_kws,
            standard_scale=self.standard_scale,
            cbar_kws=self.cbar_kws,
            row_linkage=self.row_linkage,
            col_linkage=self.col_linkage,
            mask=self.mask,
            dendrogram_ratio=self.dendrogram_ratio,
            colors_ratio=self.colors_ratio,
            cbar_pos=self.cbar_pos,
            tree_kws=self.tree_kws,
        )

        # Remove the dendrogram on the top
        if self.hide_upper:
            g.ax_col_dendrogram.remove()

        # Remove the dendrogram on the left
        if self.hide_side:
            g.ax_row_dendrogram.remove()

        # Add the title
        if self.title:
            g.figure.suptitle(self.title, y=1.05)

        # Save the fig variable
        self.fig = g.figure

        # Suppress the output
        if not self.showfig:
            plt.close()
            return self.fig

    def _get_colors(self) -> ListedColormap | None:
        """Get the row and column colors for the clustermap.

        Notes:
        - For valid palettes, see https://seaborn.pydata.org/generated/seaborn.color_palette.html.
        - The value "default" will use the husl palette with 8 colours.

        Returns:
            A matplotlib ListedColormap or None.
        """
        # Ensure that lists of colours are longer than the number of labels
        # Not sure if this is necessary for column colours
        # if isinstance(self.col_colors, list) and len(self.dtm.labels) >= len(self.col_colors):
        #     raise LexosException("The length of `col_colors` must have be greater than the number of labels.")
        if isinstance(self.row_colors, list) and len(self.dtm.labels) >= len(
            self.row_colors
        ):
            raise LexosException(
                "The length of `row_colors` must have be greater than the number of labels."
            )

        # Convert palette to vectors drawn on the side of the matrix
        # None means no colours, "default" means use the husl palette
        if self.col_colors is None:
            col_colors = None
        elif self.col_colors == "default":
            col_colors = sns.husl_palette(8, s=0.45)
        else:
            try:
                col_colors = sns.color_palette(self.col_colors, len(self.col_colors))
            except ValueError:
                raise LexosException("Invalid column palette.")

        if self.row_colors is None:
            row_colors = None
        elif self.row_colors == "default":
            row_colors = sns.husl_palette(8, s=0.45)
        else:
            try:
                row_colors = sns.color_palette(self.row_colors, len(self.row_colors))
            except ValueError:
                raise LexosException("Invalid row palette.")

        return col_colors, row_colors

    def _get_valid_matrix(self):
        """Get a valid matrix based on the data type of the dtm."""
        if isinstance(self.dtm, DTM):
            matrix = self.dtm.to_df()
            matrix.index.name = "terms"
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

    def _set_labels(self):
        """Set the labels for the clustermap."""
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.columns.values.tolist()[1:]
            else:
                self.labels = [f"Doc{i + 1}" for i, _ in enumerate(self.dtm)]

    def _validate_linkage_matrices(self):
        """Validate the linkage matrices."""
        if self.row_linkage is not None:
            try:
                is_valid_linkage(self.row_linkage, throw=True)
            except TypeError as e:
                raise LexosException(f"Invalid `row_linkage` value: {e}")
        if self.col_linkage is not None:
            try:
                is_valid_linkage(self.col_linkage, throw=True)
            except TypeError as e:
                raise LexosException(f"Invalid `col_linkage` value: {e}")

    def save(self, path: Path | str, **kwargs):
        """Save the figure to a file.

        Args:
            path (Path | str): The path of the file to save.
            **kwargs: Additional keyword arguments for pyplot.savefig. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html.
        """
        self.fig.savefig(path, **kwargs)

    def show(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `ClusterMap.fig`. This will generally display in a
        Jupyter notebook.
        """
        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        return self.fig
