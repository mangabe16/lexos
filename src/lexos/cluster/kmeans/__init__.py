"""KMeans clustering module for Lexos.

Last Updated: July 20, 2025
Last Tested: 

This module provides a KMeansCluster class for performing KMeans clustering on document-term matrices (DTM), pandas DataFrames, or numpy arrays.
"""

from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, validate_call

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from lexos.dtm import DTM
from lexos.exceptions import LexosException

class KMeansCluster(BaseModel):
    """Lexos KMeans clustering module."""
    dtm: Optional[DTM | pd.DataFrame | np.ndarray] = Field(default=None)
    k: Optional[int] = None
    init: Literal['k-means++', 'random'] = "k-means++"
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-4
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
        """Perform KMeans clustering and return cluster assignments."""
        # Set or update attributes
        self._set_attrs(
            dtm=dtm, k=k, init=init, max_iter=max_iter, n_init=n_init, tol=tol
        )

        # Validate and convert input matrix
        matrix = self._get_valid_matrix()

        # Run KMeans
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
    
    def _get_valid_matrix(self) -> np.ndarray:
        """Convert supported input types into a 2D array for clustering."""
        if isinstance(self.dtm, DTM):
            df = self.dtm.to_df(transpose=True)
            self.labels = self.dtm.labels
        elif isinstance(self.dtm, pd.DataFrame):
            df = self.dtm
        elif isinstance(self.dtm, np.ndarray):
            df = pd.DataFrame(self.dtm)
        else: 
            raise LexosException("Unsupported input: must be DTM, DataFrame, or ndarray.")
        
        if df.shape[0] < 2:
            raise LexosException("Need at least 2 documents for clustering.")
        
        return df.values
    
    def _set_attrs(self, **kwargs):
        """Set attributes from kwargs if provided."""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def plot_2d(self, show: bool = False, save_path: Optional[str] = None) -> go.Figure:
        """Generate a 2D PCA scatter plot of the clusters."""
        if self.cluster_assignments is None:
            raise LexosException("You must run clustering before plotting.")
        
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(matrix)

        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "Cluster": self.cluster_assignments.astype(str),
            "Document": self.labels or [f"Doc{i+1}" for i in range(len(matrix))]
        })

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
        """Generate a 3D PCA scatter plot of the clusters."""
        if self.cluster_assignments is None:
            raise LexosException("You must run clustering before plotting.")

        matrix = self._get_valid_matrix()
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(matrix)

        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "z": reduced[:, 2],
            "Cluster": self.cluster_assignments.astype(str),
            "Document": self.labels or [f"Doc{i+1}" for i in range(len(matrix))]
        })

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

    def elbow_plot(
        self,
        k_range: range = range(1, 10),
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot the Elbow curve to help choose the optimal number of clusters (k)."""
        # Ensure matrix is valid
        matrix = self._get_valid_matrix()

        min_k = min(k_range)
        max_k = min(len(matrix), max(k_range))

        # Sanity check: avoid invalid ranges
        if min_k > max_k:
            raise LexosException(
                f"Cannot fit KMeans: requested k range ({min_k}–{max(k_range)}) exceeds document count ({len(matrix)})."
            )

        adjusted_range = range(min_k, max_k + 1)
        print(f"Running elbow plot for k = {min_k} to {max_k} (limited to document count)")

        # Run KMeans for each k
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

        # Plot the elbow
        plt.figure(figsize=(8, 5))
        plt.plot(list(adjusted_range), inertias, marker="o")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Within-cluster Sum of Squares)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
            

    def save_png(self, path: str) -> None:
        """
        Save the last Plotly figure as a PNG.
        Usage: kmeans.plot_2d(...); kmeans.save_png('out.png')
        """
        if self.plotly_fig is None:
            raise LexosException("No figure available: run plot_2d or plot_voronoi first.")
        self.plotly_fig.write_image(path, format="png")

    def save_svg(self, path: str) -> None:
        """
        Save the last Plotly figure as an SVG.
        Usage: kmeans.plot_3d(...); kmeans.save_svg('out.svg')
        """
        if self.plotly_fig is None:
            raise LexosException("No figure available: run plot_2d or plot_voronoi first.")
        self.plotly_fig.write_image(path, format="svg")

    def export_csv(self, path: str) -> None:
        """
        Export a CSV of each document’s PCA coordinates and cluster assignment.
        Usage: kmeans(dtm); kmeans.export_csv('out.csv')
        """
        if self.cluster_assignments is None:
            raise LexosException("No clustering results: run clustering first.")
        
        # Perform PCA to get 2D coordinates
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)
        
        # Create a DataFrame with document labels, cluster assignments, and PCA coordinates
        df = pd.DataFrame({
            "Document": self.labels or [f"Doc{i+1}" for i in range(len(coords))],
            "Cluster": self.cluster_assignments.astype(str),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        })
        
        # Export the DataFrame to a CSV file
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            raise LexosException(f"Failed to export CSV: {e}")
