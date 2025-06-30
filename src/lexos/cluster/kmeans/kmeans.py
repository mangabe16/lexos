"""kmeans.py.

Lexos KMeans clustering module for document-term matrices.

Last Updated: June 27, 2025
Last Tested: 

This module defines the KMeansCluster class, which supports:
- Running KMeans clustering on a DTM, DataFrame, or NumPy array
- 2D and 3D PCA visualizations of clusters using Plotly
- Exporting clustering results to CSV or image formats
- Interactive Voronoi-style cluster visualization
- Elbow method for detecting the optimal number of clusters

Notes:
- PCA is used for dimensionality reduction prior to plotting.
- Input must be 2D; labels are optional but improve visualization.
"""

from typing import Optional, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pydantic import BaseModel, ConfigDict, Field, validate_call

from lexos.dtm import DTM
from lexos.exceptions import LexosException

class KMeansCluster(BaseModel):
    """Perform and visualize KMeans clustering with optional dimensionality reduction."""

    # Configurable parameters for clustering
    dtm: Optional[DTM | pd.DataFrame | np.ndarray] = Field(
        default=None, description="Input document-term matrix.")
    k: Optional[int] = Field(
        default=None, description="Number of clusters to use.")
    init: Literal['k-means++', 'random'] = Field(
        default="k-means++", description="Initialization method for centroids.")
    max_iter: int = Field(
        default=300, description="Maximum number of iterations for the algorithm.")
    n_init: int = Field(
        default=10, description="Number of initializations to perform.")
    tol: float = Field(
        default=1e-4, description="Relative tolerance for convergence.")

    # Attributes populated after clustering
    labels: Optional[list[str]] = None
    cluster_assignments: Optional[np.ndarray] = None
    plotly_fig: Optional[go.Figure] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_call(config=model_config)
    def __call__(
        self,
        dtm: Optional[DTM | pd.DataFrame | np.ndarray] = None,
        k: Optional[int] = None,
        init: Optional[Literal['k-means++', 'random']] = None,
        max_iter: Optional[int] = None,
        n_init: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> np.ndarray:
        """Run KMeans clustering on the input matrix.

        Args:
            dtm (DTM | pd.DataFrame | np.ndarray, optional): Input matrix.
            k (int, optional): Number of clusters.
            init (str, optional): Initialization strategy.
            max_iter (int, optional): Maximum iterations.
            n_init (int, optional): Number of initializations.
            tol (float, optional): Tolerance for convergence.

        Returns:
            np.ndarray: Array of cluster labels for each document.
        """
        self._set_attrs(dtm=dtm, k=k, init=init, max_iter=max_iter, n_init=n_init, tol=tol)
        matrix = self._get_valid_matrix()

        if self.k is None:
            raise LexosException("Number of clusters 'k' must be specified for KMeans clustering.")
        try:
            kmeans = KMeans(
                n_clusters=self.k,
                init=self.init,
                max_iter=self.max_iter,
                n_init=self.n_init,
                tol=self.tol,
                random_state=42,
            )
            self.cluster_assignments = kmeans.fit_predict(matrix)
        except Exception as e:
            raise LexosException(f"KMeans clustering failed: {e}")
        
        return self.cluster_assignments

    def _set_attrs(self, **kwargs) -> None:
        """Update instance attributes only if new values are provided."""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
    
    def _get_valid_matrix(self) -> np.ndarray:
        """Convert the input into a valid NumPy matrix format.

        Supports DTM (Lexos), pandas DataFrame, or NumPy array.
        Raises an error for unsupported formats or too few documents.
        """
        if isinstance(self.dtm, DTM):
            df = self.dtm.to_df(transpose=True)
            self.labels = self.dtm.labels # Save labels for plotting
        elif isinstance(self.dtm, pd.DataFrame):
            df = self.dtm
        elif isinstance(self.dtm, np.ndarray):
            df = pd.DataFrame(self.dtm)
        else:
            raise LexosException("Unsupported input: must be DTM, DataFrame, or ndarray.")

        # Must have more than 1 document to cluster
        if df.shape[0] < 2:
            raise LexosException("Need at least 2 documents for clustering.")

        return df.values

    def plot_2d(self, show: bool = False, save_path: Optional[str] = None) -> go.Figure:
        """Generate a 2D PCA scatter plot of the KMeans clusters.

        Args:
            show (bool): Whether to display the plot.
            save_path (Optional[str]): Optional file path to save the plot.

        Returns:
            go.Figure: The Plotly 2D scatter plot.
        """
        if self.cluster_assignments is None:
            raise LexosException("You must run clustering before plotting.")
        
        # Perform PCA for 2D projection
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(matrix)

        # Build DataFrame for plotting
        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "Cluster": self.cluster_assignments.astype(str),
            "Document": self.labels or [f"Doc{i+1}" for i in range(len(matrix))]
        })

        # Create scatter plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="Cluster",
            hover_name="Document",
            title="KMeans Clustering 2D Plot",
        )
        fig.update_layout(margin=dict(l=12, r=10, t=40, b=10))

        self.plotly_fig = fig
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()

        return fig

    def plot_3d(self, show: bool = False, save_path: Optional[str] = None) -> go.Figure:
        """Generate a 3D PCA scatter plot of the KMeans clusters.

        Args:
            show (bool): Whether to display the plot.
            save_path (Optional[str]): Optional file path to save the plot.

        Returns:
            go.Figure: The Plotly 3D scatter plot.
        """
        if self.cluster_assignments is None:
            raise LexosException("You must run clustering before plotting.")

        # Reduce dimensions to 3 for 3D plot
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(matrix)

        # Build DataFrame for plotting
        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "z": reduced[:, 2],
            "Cluster": self.cluster_assignments.astype(str),
            "Document": self.labels or [f"Doc{i+1}" for i in range(len(matrix))]
        })

        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="Cluster",
            hover_name="Document",
            title="KMeans Clustering 3D Plot",
        )
        fig.update_layout(margin=dict(l=12, r=10, t=40, b=10))

        self.plotly_fig = fig
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()

        return fig

    def plot_voronoi(self, show: bool = True, save_path: Optional[str | Path] = None, grid_step: Optional[float] = None, max_points: int = 200_000) -> go.Figure:
        """Plot Voronoi-like decision regions for KMeans clustering using 2D PCA.

        Args:
            show (bool): Whether to display the plot interactively.
            save_path (Optional[str | Path]): File path to save the plot.
            grid_step (Optional[float]): Grid step size; estimated if None.
            max_points (int): Maximum grid points for memory efficiency.
        """
        # Reduce dimensions for 2D Voronoi visualization
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(matrix)

        if self.k is None:
            raise LexosException("Number of clusters 'k' must be specified for KMeans clustering.")

        # Fit KMeans on reduced data for plotting
        kmeans = KMeans(
            n_clusters=self.k,
            init=self.init,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=42,
        ).fit(reduced)

        centroids = kmeans.cluster_centers_

        # Define grid boundaries with buffer
        x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
        y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1

        # Estimate grid resolution to avoid memory overload
        if grid_step is None:
            range_area = (x_max - x_min) * (y_max - y_min)
            grid_step = (range_area / max_points) ** 0.5
            print(f"Grid step auto-adjusted to {grid_step:.2f} to avoid memory overload.")

        # Create mesh grid and predict cluster for each point
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))
        grid = np.c_[xx.ravel(), yy.ravel()]
        z = kmeans.predict(grid).reshape(xx.shape)

        fig = go.Figure()

        # Add background colored Voronoi regions
        fig.add_trace(go.Heatmap(
            x=xx[0], y=yy[:, 0], z=z,
            colorscale='YlGnBu', showscale=False, opacity=0.4
        ))

        # Overlay documents per cluster
        doc_labels = np.array(self.labels or [f"Doc{i+1}" for i in range(len(reduced))])
        for i in range(self.k):
            cluster_mask = self.cluster_assignments == i
            fig.add_trace(go.Scatter(
                x=reduced[cluster_mask, 0],
                y=reduced[cluster_mask, 1],
                mode='markers',
                name=f"Cluster {i+1}",
                text=doc_labels[cluster_mask],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(size=8)
            ))

        # Add centroid markers
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers+text', name='Centroids',
            text=[f"C{i+1}" for i in range(self.k)],
            hoverinfo='text', textposition="top center",
            marker=dict(symbol='x', size=14, color='black')
        ))

        fig.update_layout(
            title="Interactive KMeans Voronoi Plot (PCA Reduced)",
            xaxis_title="PC1", yaxis_title="PC2"
        )

        self.plotly_fig = fig
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    
        return fig

    def elbow_plot(
        self,
        k_range: range = range(1, 10),
        show: bool = True,
        save_path: Optional[str] = None,
        return_knee: bool = False,
    ) -> Optional[int]:
        """Generate an elbow plot to help determine the optimal number of clusters (k).

        Args:
            k_range (range): Range of k values to evaluate.
            show (bool): Whether to display the plot interactively.
            save_path (Optional[str]): Optional file path to save the elbow plot.
            return_knee (bool): If True, return the detected elbow point (optimal k).

        Returns:
            Optional[int]: Optimal number of clusters, only if return_knee is True.
        """
        # Ensure valid matrix and k range based on document count
        matrix = self._get_valid_matrix()

        min_k = min(k_range)
        max_k = min(len(matrix), max(k_range))

        if min_k > max_k:
            raise LexosException(
                f"Invalid k range ({min_k}–{max(k_range)}) exceeds document count ({len(matrix)})."
            )

        adjusted_range = range(min_k, max_k + 1)
        print(f"Running elbow plot for k = {min_k} to {max_k} (limited to document count)")

        # Run KMeans for each k in the specified range and record inertia
        inertias = []
        for k in adjusted_range:
            try:
                model = KMeans(
                    n_clusters=k,
                    init=self.init,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    tol=self.tol,
                    random_state=42,
                )
                model.fit(matrix)
                inertias.append(model.inertia_)
            except Exception as e:
                raise LexosException(f"Error fitting KMeans for k={k}: {e}")

        # Use the "maximum distance to line" method to detect elbow
        point1 = np.array([adjusted_range[0], inertias[0]])
        point2 = np.array([adjusted_range[-1], inertias[-1]])

        def distance_to_line(p):
            return np.linalg.norm(np.cross(point2 - point1, point1 - p)) / np.linalg.norm(point2 - point1)

        distances = [
            distance_to_line(np.array([k, inertia]))
            for k, inertia in zip(adjusted_range, inertias)
        ]
        optimal_k = adjusted_range[np.argmax(distances)]

        # Plot inertia vs. number of clusters and show elbow with vertical line
        plt.figure(figsize=(8, 5))
        plt.plot(list(adjusted_range), inertias, marker="o", label="Inertia")
        plt.axvline(optimal_k, color="red", linestyle="--", label=f"Elbow at k={optimal_k}")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Within-cluster Sum of Squares)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

        if return_knee:
            return optimal_k

    def save_png(self, path: str) -> None:
        """Save the most recent Plotly figure to a PNG file.

        Args:
            path (str): Path to the output PNG file.
        """
        if self.plotly_fig is None:
            raise LexosException("No figure available: run a plot method first.")
        # Save as PNG
        self.plotly_fig.write_image(path, format="png")

    def save_svg(self, path: str) -> None:
        """Save the most recent Plotly figure to an SVG file.

        Args:
            path (str): Path to the output SVG file.
        """
        if self.plotly_fig is None:
            raise LexosException("No figure available: run a plot method first.")
        # Save as SVG
        self.plotly_fig.write_image(path, format="svg")

    def export_csv(self, path: str) -> None:
        """Export a CSV of PCA coordinates and cluster labels.

        Args:
            path (str): File path to save the CSV.
        """
        if self.cluster_assignments is None:
            raise LexosException("No clustering results: run clustering first.")

        # Perform PCA to 2 components
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)

        # Create output DataFrame
        df = pd.DataFrame({
            "Document": self.labels or [f"Doc{i+1}" for i in range(len(coords))],
            "Cluster": self.cluster_assignments.astype(str),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        })

        # Export to CSV
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            raise LexosException(f"Failed to export CSV: {e}")