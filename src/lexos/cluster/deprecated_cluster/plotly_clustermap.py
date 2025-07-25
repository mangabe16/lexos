"""Plotly implementation of Seaborn's clustermap functionality with integrated dendrograms.

Sample usage:

dtm = DTM()
dtm(doc_tokens, labels)
df = dtm.to_df()

fig = plotly_clustermap(
    df,
    show_heatmap_labels=True,
    figsize=(800, 1000)
)
fig.show()

Last Updated: July 16, 2025
Last Tested: July 16, 2025

# TODO:
- Automatically detect when sparse matrices need to be converted to dense for scaling or normalisation.
- Make `fastcluster` a hard dependency for better performance on large datasets.
"""

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy


def _create_dendrogram_traces(
    linkage_matrix: np.ndarray,
    labels: Optional[list[str]] = None,
    orientation: str = "bottom",
    color: str = "rgb(50,50,50)",
    line_width: float = 1.0,
) -> list[go.Scatter]:
    """Create dendrogram traces from linkage matrix.

    Args:
        linkage_matrix (array-like): Linkage matrix from scipy.cluster.hierarchy.linkage
        labels (list, optional): Labels for the leaves
        orientation (str): Orientation of dendrogram ('top', 'bottom', 'left', 'right')
        color (str): Color for dendrogram lines
        line_width (float): Width of dendrogram lines

    Returns:
        traces (list): List of plotly scatter traces for dendrogram
    """
    dendro_data = hierarchy.dendrogram(
        linkage_matrix, labels=labels, no_plot=True, color_threshold=-np.inf
    )

    traces = []

    # Extract coordinates
    icoord = np.array(dendro_data["icoord"])
    dcoord = np.array(dendro_data["dcoord"])

    # Create line traces for each dendrogram segment
    for i in range(len(icoord)):
        x_coords = icoord[i]
        y_coords = dcoord[i]

        if orientation in ["top", "bottom"]:
            # Standard orientation
            if orientation == "bottom":
                y_coords = -y_coords + max(dcoord.flatten())
        else:
            # Swap coordinates for left/right orientation
            x_coords, y_coords = y_coords, x_coords
            if orientation == "left":
                x_coords = -x_coords + max(dcoord.flatten())
                # Shift dendrogram to touch the right edge
                x_coords = x_coords + (max(x_coords) - min(x_coords)) * 0.03

        # Create scatter trace for this segment
        trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line=dict(color=color, width=line_width),
            showlegend=False,
            hoverinfo="skip",
        )
        traces.append(trace)

    return traces, dendro_data


class PlotlyClusterGrid:
    """Plotly implementation of clustered heatmap with dendrograms."""

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        z_score: Optional[int] = None,
        standard_scale: Optional[int] = None,
        mask: Optional[np.ndarray | pd.DataFrame] = None,
        figsize: tuple[int, int] = (800, 600),
        dendrogram_ratio: float | tuple[float, float] = 0.2,
    ) -> None:
        """Initialize the cluster grid.

        Args:
            data (DataFrame or array-like): Rectangular data for clustering
            z_score (int, optional): Whether to z-score rows (0) or columns (1)
            standard_scale (int, optional): Whether to standard scale rows (0) or columns (1)
            mask (bool array or DataFrame, optional): Mask for data visualization
            figsize (tuple[int, int]): Figure size (width, height)
            dendrogram_ratio (float | tuple[float, float]): Ratio of dendrogram size to heatmap size
        """
        # Convert data to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = pd.DataFrame(data)

        # Process data
        self.data2d = self._format_data(z_score, standard_scale)
        self.mask = self._process_mask(mask)

        # Store configuration
        self.figsize = figsize
        self.dendrogram_ratio = dendrogram_ratio

    def _format_data(
        self, z_score: Optional[int] = None, standard_scale: Optional[int] = None
    ) -> pd.DataFrame:
        """Format and normalize data.

        Args:
            z_score (int, optional): Whether to z-score rows (0) or columns (1)
            standard_scale (int, optional): Whether to standard scale rows (0) or columns (1)

        Returns:
            pd.DataFrame: Formatted data
        """
        data2d = self.data.copy()

        if z_score is not None and standard_scale is not None:
            raise ValueError(
                "Cannot perform both z-scoring and standard-scaling on data"
            )

        if z_score is not None:
            data2d = self._z_score(data2d, z_score)
        if standard_scale is not None:
            data2d = self._standard_scale(data2d, standard_scale)

        return data2d

    @staticmethod
    def _z_score(data2d: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """Standardize the mean and variance of the data axis.

        Args:
            data2d (pd.DataFrame): Data to z-score
        Returns:
            pd.DataFrame: Z-scored data
        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        z_scored = (z_scored - z_scored.mean()) / z_scored.std()

        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def _standard_scale(data2d: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """Divide the data by the difference between the max and min.

        Args:
            data2d (pd.DataFrame): Data to standard scale
            axis (int, optional): Axis along which to scale (0 for rows, 1 for columns)

        Returns:
            pd.DataFrame: Standard scaled data
        """
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        subtract = standardized.min()
        standardized = (standardized - subtract) / (
            standardized.max() - standardized.min()
        )

        if axis == 1:
            return standardized
        else:
            return standardized.T

    def _process_mask(
        self, mask: Optional[np.ndarray | pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Process mask for data visualization.

        Args:
            mask (np.ndarray | pd.DataFrame, optional): Mask to apply to the data

        Returns:
            pd.DataFrame: Processed mask
        """
        if mask is None:
            return None

        if isinstance(mask, pd.DataFrame):
            if not (
                mask.index.equals(self.data2d.index)
                and mask.columns.equals(self.data2d.columns)
            ):
                raise ValueError("Mask must have the same index and columns as data.")
        else:
            mask = np.asarray(mask)
            if mask.shape != self.data2d.shape:
                raise ValueError("Mask must have the same shape as data.")
            mask = pd.DataFrame(
                mask, index=self.data2d.index, columns=self.data2d.columns, dtype=bool
            )

        # Add missing data to mask
        mask = mask | pd.isnull(self.data2d)
        return mask

    def _calculate_linkage(
        self, data: np.ndarray, method: str = "average", metric: str = "euclidean"
    ) -> np.ndarray:
        """Calculate linkage matrix.

        Args:
            data (np.ndarray): Data to cluster
            method (str): Linkage method
            metric (str): Distance metric

        Returns:
            np.ndarray: Linkage matrix
        """
        try:
            import fastcluster

            euclidean_methods = ("centroid", "median", "ward")
            euclidean = metric == "euclidean" and method in euclidean_methods
            if euclidean or method == "single":
                return fastcluster.linkage_vector(data, method=method, metric=metric)
            else:
                return fastcluster.linkage(data, method=method, metric=metric)
        except ImportError:
            if np.prod(data.shape) >= 10000:
                warnings.warn(
                    "Clustering large matrix with scipy. Installing "
                    "`fastcluster` may give better performance."
                )
            return hierarchy.linkage(data, method=method, metric=metric)


def plotly_clustermap(
    data: pd.DataFrame | np.ndarray,
    method: str = "average",
    metric: str = "euclidean",
    z_score: Optional[int] = None,
    standard_scale: Optional[int] = None,
    figsize: tuple[int, int] = (800, 600),
    row_cluster: bool = True,
    col_cluster: bool = True,
    row_linkage: Optional[np.ndarray] = None,
    col_linkage: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray | pd.DataFrame] = None,
    dendrogram_ratio: float | tuple[float, float] = (0.8, 0.2),
    cmap: str = "RdBu_r",
    center: Optional[float] = None,
    annot: bool = False,
    fmt: str = ".2g",
    linewidths: float = 0,
    tree_kws: Optional[dict[str, Any]] = None,
    show_dendrogram_labels: bool = False,
    show_heatmap_labels: Optional[bool] = None,
    colorbar: Optional[dict[str, Any]] = dict(x=0.11, xref="container"),
    **kwargs,
) -> go.Figure:
    """Create a clustered heatmap using Plotly with integrated dendrograms.

    This is the main function that reproduces seaborn.clustermap functionality
    with proper dendrogram integration.

    Args:
        data (DataFrame or array-like): Rectangular data for clustering
        method (str): Linkage method for clustering ('average', 'complete', 'ward', etc.). Default is 'average'.
        metric (str): Distance metric for clustering ('euclidean', 'correlation', 'cosine', etc.). Default is 'euclidean'.
        z_score (int or None): Whether to z-score rows (0) or columns (1). Default is None (no z-scoring).
        standard_scale (int or None): Whether to standard scale rows (0) or columns (1). Default is None (no scaling).
        figsize (tuple): Figure size (width, height) in pixels. Default is (800, 600).
        row_cluster (bool): Whether to cluster rows. Default is True.
        col_cluster (bool): Whether to cluster columns. Default is True.
        row_linkage (array-like, optional): Precomputed row linkage matrix
        col_linkage (array-like, optional): Precomputed column linkage matrix
        mask (bool array or DataFrame, optional): Mask for hiding certain cells.
        dendrogram_ratio (float or tuple): Proportion of figure devoted to dendrograms
            Proportion of figure devoted to dendrograms (default (0.8, 0.02) for wide, thin dendrograms)
        cmap (str): Plotly colorscale name. Default is 'RdBu_r'.
        center (float, optional): Value to center colormap around
        annot (bool): Whether to annotate cells with values. Default is False.
        fmt (str): String formatting for annotations. Default is '.2g'.
        linewidths (float): Width of lines separating cells. Default is 0.
        tree_kws (dict, optional): Keyword arguments for dendrogram styling.
        show_dendrogram_labels (bool): Whether to show labels on dendrograms (default False for cleaner look).
        show_heatmap_labels (bool, optional): Whether to show axis labels on heatmap. If None (auto-mode), bottom labels
            are always shown for readability, but left labels are hidden when row
            dendrogram is present. Manual True/False controls both axes uniformly.
        colorbar (dict, optional): Dictionary of colorbar properties. Default is {'x': 0.11, 'xref': 'container'}.
        **kwargs: Additional arguments passed to heatmap

    Returns:
        fig (plotly.graph_objects.Figure): Clustered heatmap figure with integrated dendrograms
    """
    # Extract our custom parameters from kwargs to prevent them being passed to plotly components
    filtered_kwargs = kwargs.copy()
    filtered_kwargs.pop(
        "show_dendrogram_labels", None
    )  # This is already a function parameter
    filtered_kwargs.pop(
        "show_heatmap_labels", None
    )  # This is already a function parameter
    filtered_kwargs.pop("title", None)  # Title should go to layout, not heatmap trace

    # Determine whether to show heatmap labels
    if show_heatmap_labels is None:
        # Auto-mode: hide y-axis (left) labels when row dendrogram is present,
        # but always show x-axis (bottom) labels for readability
        show_heatmap_x_labels = (
            True  # Always show bottom labels unless explicitly disabled
        )
        show_heatmap_y_labels = (
            not row_cluster
        )  # Hide left labels only if row dendrogram present
    else:
        # Manual mode: use the same setting for both axes
        show_heatmap_x_labels = show_heatmap_labels
        show_heatmap_y_labels = show_heatmap_labels

    # Initialize cluster grid
    grid = PlotlyClusterGrid(
        data=data,
        z_score=z_score,
        standard_scale=standard_scale,
        mask=mask,
        figsize=figsize,
        dendrogram_ratio=dendrogram_ratio,
    )

    # Handle dendrogram ratios
    if isinstance(dendrogram_ratio, (list, tuple)):
        row_dendrogram_ratio, col_dendrogram_ratio = dendrogram_ratio
    else:
        row_dendrogram_ratio = col_dendrogram_ratio = dendrogram_ratio

    # Handle tree styling
    if tree_kws is None:
        tree_kws = {}
    tree_color = tree_kws.get("color", "rgb(50,50,50)")
    tree_width = tree_kws.get("linewidth", 1.0)

    # Calculate clustering
    data_array = grid.data2d.values

    # Row clustering
    if row_cluster:
        if row_linkage is None:
            row_linkage = grid._calculate_linkage(data_array, method, metric)
        row_dendro_traces, row_dendro_data = _create_dendrogram_traces(
            row_linkage,
            labels=[str(x) for x in grid.data2d.index]
            if show_dendrogram_labels
            else None,
            orientation="left",
            color=tree_color,
            line_width=tree_width,
        )
        row_order = row_dendro_data["leaves"]
    else:
        row_order = list(range(len(grid.data2d.index)))
        row_dendro_traces = []
        row_dendro_data = None

    # Column clustering
    if col_cluster:
        if col_linkage is None:
            col_linkage = grid._calculate_linkage(data_array.T, method, metric)
        col_dendro_traces, col_dendro_data = _create_dendrogram_traces(
            col_linkage,
            labels=[str(x) for x in grid.data2d.columns]
            if show_dendrogram_labels
            else None,
            orientation="top",
            color=tree_color,
            line_width=tree_width,
        )
        col_order = col_dendro_data["leaves"]
    else:
        col_order = list(range(len(grid.data2d.columns)))
        col_dendro_traces = []
        col_dendro_data = None

    # Reorder data
    ordered_data = grid.data2d.iloc[row_order, col_order]

    # Create subplot layout
    n_rows = 2 if col_cluster else 1
    n_cols = 2 if row_cluster else 1

    # Calculate subplot dimensions
    if row_cluster and col_cluster:
        row_heights = [col_dendrogram_ratio, 1 - col_dendrogram_ratio]
        col_widths = [1 - row_dendrogram_ratio, row_dendrogram_ratio]
        subplot_titles = ["", "Column Dendrogram", "Row Dendrogram", "Heatmap"]
    elif col_cluster:
        row_heights = [col_dendrogram_ratio, 1 - col_dendrogram_ratio]
        col_widths = [1.0]
        subplot_titles = ["Column Dendrogram", "Heatmap"]
    elif row_cluster:
        row_heights = [1.0]
        col_widths = [1 - row_dendrogram_ratio, row_dendrogram_ratio]
        subplot_titles = ["Heatmap", "Row Dendrogram"]
    else:
        row_heights = [1.0]
        col_widths = [1.0]
        subplot_titles = ["Heatmap"]

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        row_heights=row_heights,
        column_widths=col_widths,
        horizontal_spacing=0,  # Remove padding between dendrograms and heatmap
        vertical_spacing=0,  # Remove padding between dendrograms and heatmap
        subplot_titles=None,  # We'll add custom titles if needed
    )

    # Determine subplot positions
    heatmap_row = n_rows
    heatmap_col = 1 if not row_cluster else n_cols

    # Prepare heatmap data
    z_data = ordered_data.values
    x_labels = [str(x) for x in ordered_data.columns]
    y_labels = [str(y) for y in ordered_data.index]

    # Apply mask if provided
    if grid.mask is not None:
        mask_ordered = grid.mask.iloc[row_order, col_order]
        z_data = np.where(mask_ordered.values, np.nan, z_data)

    # Add heatmap
    heatmap_trace = go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=cmap,
        zmid=center,
        showscale=True,
        colorbar=colorbar,
        **filtered_kwargs,
    )

    fig.add_trace(heatmap_trace, row=heatmap_row, col=heatmap_col)

    # Add column dendrogram
    if col_cluster and col_dendro_traces:
        for trace in col_dendro_traces:
            fig.add_trace(trace, row=1, col=heatmap_col)

    # Add row dendrogram
    if row_cluster and row_dendro_traces:
        for trace in row_dendro_traces:
            fig.add_trace(trace, row=heatmap_row, col=1)

    # Add annotations if requested
    if annot:
        annotations = []
        for i, row in enumerate(y_labels):
            for j, col in enumerate(x_labels):
                if not (grid.mask is not None and mask_ordered.iloc[i, j]):
                    cell_value = z_data[i, j]
                    if not np.isnan(cell_value):
                        max_abs_val = np.nanmax(np.abs(z_data))
                        text_color = (
                            "white" if abs(cell_value) > max_abs_val / 2 else "black"
                        )

                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=format(cell_value, fmt),
                                showarrow=False,
                                font=dict(color=text_color, size=10),
                                xref=f"x{heatmap_col}" if heatmap_col > 1 else "x",
                                yref=f"y{heatmap_row}" if heatmap_row > 1 else "y",
                            )
                        )

        fig.update_layout(annotations=annotations)

    # Update layout
    fig.update_layout(
        title="Clustered Heatmap with Dendrograms",
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
    )

    # Configure axes for each subplot
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            # Generate xaxis and yaxis references
            xaxis_ref = (
                f"xaxis{col + (row - 1) * n_cols}"
                if col + (row - 1) * n_cols > 1
                else "xaxis"
            )
            yaxis_ref = (
                f"yaxis{col + (row - 1) * n_cols}"
                if col + (row - 1) * n_cols > 1
                else "yaxis"
            )

            # Default settings for all subplots
            fig.update_layout(
                {
                    xaxis_ref: dict(
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks="",
                    ),
                    yaxis_ref: dict(
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks="",
                    ),
                }
            )

    # Special configuration for heatmap
    heatmap_xaxis = (
        f"xaxis{heatmap_col + (heatmap_row - 1) * n_cols}"
        if heatmap_col + (heatmap_row - 1) * n_cols > 1
        else "xaxis"
    )
    heatmap_yaxis = (
        f"yaxis{heatmap_col + (heatmap_row - 1) * n_cols}"
        if heatmap_col + (heatmap_row - 1) * n_cols > 1
        else "yaxis"
    )

    fig.update_layout(
        {
            heatmap_xaxis: dict(
                showticklabels=show_heatmap_x_labels,
                tickmode="array" if show_heatmap_x_labels else "linear",
                tickvals=list(range(len(x_labels))) if show_heatmap_x_labels else [],
                ticktext=x_labels if show_heatmap_x_labels else [],
                tickangle=45 if show_heatmap_x_labels else 0,
                side="bottom",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="" if not show_heatmap_x_labels else "outside",
            ),
            heatmap_yaxis: dict(
                showticklabels=show_heatmap_y_labels,
                tickmode="array" if show_heatmap_y_labels else "linear",
                tickvals=list(range(len(y_labels))) if show_heatmap_y_labels else [],
                ticktext=y_labels if show_heatmap_y_labels else [],
                autorange="reversed",  # Reverse to match typical heatmap orientation
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="" if not show_heatmap_y_labels else "outside",
                side="right",
            ),
        }
    )

    # Configure dendrogram axes ranges
    if col_cluster and col_dendro_data:
        col_dend_xaxis = f"xaxis{heatmap_col}" if heatmap_col > 1 else "xaxis"
        col_dend_yaxis = f"yaxis{heatmap_col}" if heatmap_col > 1 else "yaxis"

        # Set ranges for column dendrogram
        fig.update_layout(
            {
                col_dend_xaxis: dict(
                    range=[0, len(ordered_data.columns) * 10 + 5],
                    showticklabels=show_dendrogram_labels,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="" if not show_dendrogram_labels else "outside",
                ),
                col_dend_yaxis: dict(
                    range=[
                        0,
                        max(np.array(col_dendro_data["dcoord"]).flatten()) * 1.05,
                    ],
                    showticklabels=show_dendrogram_labels,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="" if not show_dendrogram_labels else "outside",
                ),
            }
        )

    if row_cluster and row_dendro_data:
        row_dend_xaxis = (
            f"xaxis{1 + (heatmap_row - 1) * n_cols}"
            if 1 + (heatmap_row - 1) * n_cols > 1
            else "xaxis"
        )

        row_dend_yaxis = (
            f"yaxis{1 + (heatmap_row - 1) * n_cols}"
            if 1 + (heatmap_row - 1) * n_cols > 1
            else "yaxis"
        )

        # Set ranges for row dendrogram
        fig.update_layout(
            {
                row_dend_xaxis: dict(
                    range=[
                        0,
                        max(np.array(row_dendro_data["dcoord"]).flatten()) * 1.02,
                    ],
                    showticklabels=show_dendrogram_labels,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="" if not show_dendrogram_labels else "outside",
                ),
                row_dend_yaxis: dict(
                    range=[0, len(ordered_data.index) * 10],
                    showticklabels=show_dendrogram_labels,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="" if not show_dendrogram_labels else "outside",
                ),
            }
        )

    fig.update_layout(title_x=0.5)  # Automatically adjust y-axis margins

    return fig
