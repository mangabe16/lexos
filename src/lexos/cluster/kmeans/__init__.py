"""KMeans clustering module for Lexos.

This module provides a KMeansCluster class for performing KMeans clustering on document-term matrices (DTM), pandas DataFrames, or numpy arrays.
"""

from typing import Optional, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, validate_call

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from lexos.dtm import DTM
from lexos.exceptions import LexosException

class KMeansCluster(BaseModel):
    """Lexos KMeans clustering module."""
    dtm: Optional[Union[DTM, pd.DataFrame, np.ndarray]] = Field(default=None)
    k: int = Field(..., description="Number of clusters")
    init: Literal['k-means++', 'random'] = Field("k-means++", description="KMeans init method")
    max_iter: int = Field(300, description="Maximum number of iterations")
    n_init: int = Field(10, description ="Number of centroid seeds")
    tol: float = Field(1e-4, description="Convergence tolerance")
    labels: Optional[list[str]] = None
    cluster_assignments: Optional[np.ndarray] = None
    plotly_fig: Optional[go.Figure] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_call(config=model_config)
    def __call__(
        self,
        dtm: Optional[Union[DTM, pd.DataFrame, np.ndarray]] = None,
        k: Optional[int] = None,
        init: Optional[Literal['k-means++', 'random']] = None,
        max_iter: Optional[int] = None,
        n_init: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> np.ndarray:
        """Perform KMeans clustering and return cluster assignments."""
        # Set or update attributes
        self._set_attrs(
            dtm=dtm, k=k, init=init, max_iter=max_iter, n_init=n_init, told=tol
        )

        # Validate and convert input matrix
        matrix = self._get_valid_matrix()

        # Run KMeans
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
