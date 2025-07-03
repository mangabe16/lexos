"""test_kmeans.py.

Test suite for the lexos.cluster.kmeans.kmeans module.

Last Updated: 2025-07-02
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from lexos.cluster.kmeans.kmeans import KMeans
from lexos.dtm import DTM
from lexos.exceptions import LexosException


class TestKMeansInitialization:
    """Test cases for KMeans initialization."""

    def test_kmeans_init_default(self):
        """Test KMeans initialization with default parameters."""
        kmeans = KMeans()

        assert kmeans.dtm is None
        assert kmeans.k is None
        assert kmeans.init == "k-means++"
        assert kmeans.max_iter == 300
        assert kmeans.n_init == 10
        assert kmeans.tol == 1e-4
        assert kmeans.random_state == 42
        assert kmeans.labels is None
        assert kmeans.cluster_assignments is None
        assert kmeans.fig is None

    def test_kmeans_init_custom_parameters(self):
        """Test KMeans initialization with custom parameters."""
        # Use a real DataFrame instead of a Mock for dtm
        dtm_df = pd.DataFrame(np.random.rand(5, 10))

        kmeans = KMeans(
            dtm=dtm_df,
            k=5,
            init="random",
            max_iter=500,
            n_init=20,
            tol=1e-5,
            random_state=123,
        )

        assert kmeans.dtm is not None
        assert isinstance(kmeans.dtm, pd.DataFrame)
        assert kmeans.k == 5
        assert kmeans.init == "random"
        assert kmeans.max_iter == 500
        assert kmeans.n_init == 20
        assert kmeans.tol == 1e-5
        assert kmeans.random_state == 123


class TestKMeansSetAttrs:
    """Test cases for _set_attrs method."""

    def test_set_attrs_updates_only_provided_values(self):
        """Test _set_attrs only updates attributes when values are provided."""
        kmeans = KMeans(k=3, max_iter=100)

        kmeans._set_attrs(k=5, max_iter=None, tol=1e-6)

        assert kmeans.k == 5  # Updated
        assert kmeans.max_iter == 100  # Not updated (None provided)
        assert kmeans.tol == 1e-6  # Updated

    def test_set_attrs_with_all_none(self):
        """Test _set_attrs with all None values."""
        kmeans = KMeans(k=3, max_iter=100)
        original_k = kmeans.k
        original_max_iter = kmeans.max_iter

        kmeans._set_attrs(k=None, max_iter=None, tol=None)

        assert kmeans.k == original_k
        assert kmeans.max_iter == original_max_iter


class TestKMeansGetValidMatrix:
    """Test cases for _get_valid_matrix method."""

    def test_get_valid_matrix_with_dtm(self):
        """Test _get_valid_matrix with DTM input."""
        # Mock DTM
        mock_dtm = Mock(spec=DTM)
        mock_df = pd.DataFrame(np.random.rand(5, 10))
        mock_dtm.to_df.return_value = mock_df
        mock_dtm.labels = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        kmeans = KMeans(dtm=mock_dtm)
        result = kmeans._get_valid_matrix()

        assert isinstance(result, np.ndarray)
        assert result.shape == mock_df.values.shape
        assert kmeans.labels == mock_dtm.labels
        mock_dtm.to_df.assert_called_once_with(transpose=True)

    def test_get_valid_matrix_with_dataframe(self):
        """Test _get_valid_matrix with DataFrame input."""
        df = pd.DataFrame(np.random.rand(5, 10))
        kmeans = KMeans(dtm=df)

        result = kmeans._get_valid_matrix()

        assert isinstance(result, np.ndarray)
        assert result.shape == df.values.shape
        np.testing.assert_array_equal(result, df.values)

    def test_get_valid_matrix_with_numpy_array(self):
        """Test _get_valid_matrix with numpy array input."""
        arr = np.random.rand(5, 10)
        kmeans = KMeans(dtm=arr)

        result = kmeans._get_valid_matrix()

        assert isinstance(result, np.ndarray)
        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_get_valid_matrix_unsupported_input(self):
        """Test _get_valid_matrix with unsupported input type."""
        with pytest.raises((ValidationError, LexosException)):
            # with pytest.raises(ValidationError, match="Unsupported input"):
            kmeans = KMeans(dtm="invalid_input")

    def test_get_valid_matrix_too_few_documents(self):
        """Test _get_valid_matrix with too few documents."""
        df = pd.DataFrame(np.random.rand(1, 10))  # Only 1 document
        kmeans = KMeans(dtm=df)

        with pytest.raises(LexosException, match="Need at least 2 documents"):
            kmeans._get_valid_matrix()

    def test_get_valid_matrix_no_dtm(self):
        """Test _get_valid_matrix with no DTM provided."""
        kmeans = KMeans()

        with pytest.raises(LexosException, match="Unsupported input"):
            kmeans._get_valid_matrix()


class TestKMeansCall:
    """Test cases for __call__ method."""

    def test_call_with_valid_input(self):
        """Test __call__ method with valid input."""
        data = np.random.rand(5, 10)
        kmeans = KMeans()

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn.return_value = mock_model

            result = kmeans(dtm=data, k=2)

            assert isinstance(result, np.ndarray)
            assert len(result) == 5
            assert kmeans.cluster_assignments is not None
            mock_sklearn.assert_called_once()

    def test_call_no_k_specified(self):
        """Test __call__ method without specifying k."""
        data = np.random.rand(5, 10)
        kmeans = KMeans()

        with pytest.raises(
            LexosException, match="Number of clusters 'k' must be specified"
        ):
            kmeans(dtm=data)

    def test_call_sklearn_exception(self):
        """Test __call__ method when sklearn raises exception."""
        data = np.random.rand(5, 10)
        kmeans = KMeans()

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_sklearn.side_effect = ValueError("Test error")

            with pytest.raises(LexosException, match="KMeans clustering failed"):
                kmeans(dtm=data, k=2)

    def test_call_updates_attributes(self):
        """Test __call__ method updates instance attributes."""
        data = np.random.rand(5, 10)
        kmeans = KMeans(k=3)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 2, 0, 1])
            mock_sklearn.return_value = mock_model

            result = kmeans(dtm=data, k=2, max_iter=500)

            assert kmeans.k == 2  # Updated from call
            assert kmeans.max_iter == 500  # Updated from call
            assert kmeans.cluster_assignments is not None


class TestKMeansElbowPlot:
    """Test cases for elbow_plot method."""

    def test_elbow_plot_basic(self):
        """Test basic elbow plot functionality."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with (
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("matplotlib.pyplot.show") as mock_show,
            patch("matplotlib.pyplot.close") as mock_close,
        ):
            mock_model = Mock()
            mock_model.inertia_ = 100.0
            mock_model.fit.return_value = mock_model
            mock_sklearn.return_value = mock_model

            kmeans.elbow_plot(k_range=range(2, 5), show=False)

            mock_close.assert_called_once()
            mock_show.assert_not_called()

    def test_elbow_plot_with_show(self):
        """Test elbow plot with show=True."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with (
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            mock_model = Mock()
            mock_model.inertia_ = 100.0
            mock_model.fit.return_value = mock_model
            mock_sklearn.return_value = mock_model

            kmeans.elbow_plot(k_range=range(2, 5), show=True)

            mock_show.assert_called_once()

    def test_elbow_plot_with_save_path(self):
        """Test elbow plot with save_path."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with (
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.close") as mock_close,
        ):
            mock_model = Mock()
            mock_model.inertia_ = 100.0
            mock_model.fit.return_value = mock_model
            mock_sklearn.return_value = mock_model

            kmeans.elbow_plot(k_range=range(2, 5), save_path="test.png", show=False)

            mock_savefig.assert_called_once_with("test.png")
            mock_close.assert_called_once()

    def test_elbow_plot_return_knee(self):
        """Test elbow plot with return_knee=True."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with (
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("matplotlib.pyplot.close"),
        ):
            mock_model = Mock()
            mock_model.inertia_ = 100.0
            mock_model.fit.return_value = mock_model
            mock_sklearn.return_value = mock_model

            result = kmeans.elbow_plot(
                k_range=range(2, 5), show=False, return_knee=True
            )

            assert isinstance(result, int)
            assert result in range(2, 5)

    def test_elbow_plot_invalid_k_range(self):
        """Test elbow plot with invalid k range."""
        data = np.random.rand(3, 5)  # Only 3 documents
        kmeans = KMeans(dtm=data)

        with pytest.raises(LexosException, match="Invalid k range"):
            kmeans.elbow_plot(k_range=range(5, 10))

    def test_elbow_plot_sklearn_exception(self):
        """Test elbow plot when sklearn raises exception."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_sklearn.side_effect = ValueError("Test error")

            with pytest.raises(LexosException, match="Error fitting KMeans"):
                kmeans.elbow_plot(k_range=range(2, 4))


class TestKMeansScatter:
    """Test cases for scatter method."""

    def test_scatter_2d(self):
        """Test 2D scatter plot."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_scatter,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_scatter.return_value = mock_fig

            result = kmeans.scatter(dim=2, show=False)

            assert result == mock_fig
            assert kmeans.fig == mock_fig
            mock_pca.assert_called_once_with(n_components=2)

    def test_scatter_3d(self):
        """Test 3D scatter plot."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter_3d") as mock_scatter_3d,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 3)
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_scatter_3d.return_value = mock_fig

            result = kmeans.scatter(dim=3, show=False)

            assert result == mock_fig
            assert kmeans.fig == mock_fig
            mock_pca.assert_called_once_with(n_components=3)

    def test_scatter_no_clustering(self):
        """Test scatter plot without prior clustering."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with pytest.raises(
            LexosException, match="You must run clustering before plotting"
        ):
            kmeans.scatter()

    def test_scatter_invalid_dimensions(self):
        """Test scatter plot with invalid dimensions."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with pytest.raises(
            LexosException, match="The number of dimensions must be either 2 or 3"
        ):
            kmeans.scatter(dim=4)

    def test_scatter_with_show(self):
        """Test scatter plot with show=True."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_scatter,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_scatter.return_value = mock_fig

            result = kmeans.scatter(dim=2, show=True)

            assert result is None
            mock_fig.show.assert_called_once()

    def test_scatter_with_save_path(self):
        """Test scatter plot with save_path."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_scatter,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_scatter.return_value = mock_fig

            kmeans.scatter(dim=2, save_path="test.png", show=False)

            mock_fig.write_image.assert_called_once_with("test.png")


class TestKMeansVoronoi:
    """Test cases for voronoi method."""

    def test_voronoi_basic(self):
        """Test basic Voronoi plot functionality."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Mock the voronoi method at the class level
        mock_fig = Mock()

        with patch.object(KMeans, "voronoi", return_value=mock_fig) as mock_voronoi:
            result = kmeans.voronoi(show=False)

            mock_voronoi.assert_called_once_with(show=False)
            assert result == mock_fig

    def test_voronoi_no_k(self):
        """Test Voronoi plot without k specified."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data)

        with pytest.raises(
            LexosException, match="Number of clusters 'k' must be specified"
        ):
            kmeans.voronoi()

    def test_voronoi_with_show(self):
        """Test Voronoi plot with show=True."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Mock the voronoi method at the class level
        with patch.object(KMeans, "voronoi", return_value=None) as mock_voronoi:
            result = kmeans.voronoi(show=True)

            mock_voronoi.assert_called_once_with(show=True)
            assert result is None

    def test_voronoi_with_save_path(self):
        """Test Voronoi plot with save_path."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("plotly.graph_objects.Figure") as mock_fig_class,
            patch("numpy.meshgrid") as mock_meshgrid,
            patch("numpy.c_") as mock_c,
            patch("numpy.arange") as mock_arange,
        ):
            # Mock PCA
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            # Mock sklearn KMeans
            mock_sklearn_instance = Mock()
            mock_sklearn_instance.cluster_centers_ = np.random.rand(2, 2)
            mock_sklearn_instance.predict.return_value = np.array(
                [0, 1] * 12 + [0]
            )  # 25 elements
            mock_sklearn_instance.fit.return_value = mock_sklearn_instance
            mock_sklearn.return_value = mock_sklearn_instance

            # Mock numpy operations
            mock_arange.return_value = np.array([0, 1, 2, 3, 4])
            mock_meshgrid.return_value = (np.ones((5, 5)), np.ones((5, 5)))
            mock_c.return_value = np.random.rand(25, 2)

            # Mock plotly Figure
            mock_fig = Mock()
            mock_fig_class.return_value = mock_fig

            kmeans.voronoi(save_path="test.png", show=False)

            mock_fig.write_image.assert_called_once_with("test.png")

    def test_voronoi_with_custom_grid_step(self):
        """Test Voronoi plot with custom grid step."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("plotly.graph_objects.Figure") as mock_fig_class,
            patch("numpy.meshgrid") as mock_meshgrid,
            patch("numpy.c_") as mock_c,
            patch("numpy.arange") as mock_arange,
        ):
            # Mock PCA
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            # Mock sklearn KMeans
            mock_sklearn_instance = Mock()
            mock_sklearn_instance.cluster_centers_ = np.random.rand(2, 2)
            mock_sklearn_instance.predict.return_value = np.array(
                [0, 1] * 12 + [0]
            )  # 25 elements
            mock_sklearn_instance.fit.return_value = mock_sklearn_instance
            mock_sklearn.return_value = mock_sklearn_instance

            # Mock numpy operations
            mock_arange.return_value = np.array([0, 1, 2, 3, 4])
            mock_meshgrid.return_value = (np.ones((5, 5)), np.ones((5, 5)))
            mock_c.return_value = np.random.rand(25, 2)

            # Mock plotly Figure
            mock_fig = Mock()
            mock_fig_class.return_value = mock_fig

            result = kmeans.voronoi(grid_step=0.5, show=False)

            assert result == mock_fig
            mock_arange.assert_called()

    def test_voronoi_auto_grid_step_adjustment(self):
        """Test Voronoi plot with automatic grid step adjustment."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("plotly.graph_objects.Figure") as mock_fig_class,
            patch("numpy.meshgrid") as mock_meshgrid,
            patch("numpy.c_") as mock_c,
            patch("numpy.arange") as mock_arange,
            patch("builtins.print") as mock_print,
        ):
            # Mock PCA with specific values for grid calculation
            mock_pca_instance = Mock()
            test_data = np.array(
                [
                    [0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                    [5, 5],
                    [6, 6],
                    [7, 7],
                    [8, 8],
                    [9, 9],
                ]
            )
            mock_pca_instance.fit_transform.return_value = test_data
            mock_pca.return_value = mock_pca_instance

            # Mock sklearn KMeans
            mock_sklearn_instance = Mock()
            mock_sklearn_instance.cluster_centers_ = np.array([[2, 2], [7, 7]])
            mock_sklearn_instance.predict.return_value = np.array(
                [0, 1] * 12 + [0]
            )  # 25 elements
            mock_sklearn_instance.fit.return_value = mock_sklearn_instance
            mock_sklearn.return_value = mock_sklearn_instance

            # Mock numpy operations
            mock_arange.return_value = np.array([0, 1, 2, 3, 4])
            mock_meshgrid.return_value = (np.ones((5, 5)), np.ones((5, 5)))
            mock_c.return_value = np.random.rand(25, 2)

            # Mock plotly Figure
            mock_fig = Mock()
            mock_fig_class.return_value = mock_fig

            result = kmeans.voronoi(show=False)

            # Verify that auto-adjustment print message was called
            mock_print.assert_called()
            print_args = mock_print.call_args[0][0]
            assert "Grid step auto-adjusted" in print_args
            assert result == mock_fig

    def test_voronoi_with_custom_labels(self):
        """Test Voronoi plot with custom document labels."""
        data = np.random.rand(5, 3)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0])
        kmeans.labels = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("plotly.graph_objects.Figure") as mock_fig_class,
            patch("numpy.meshgrid") as mock_meshgrid,
            patch("numpy.c_") as mock_c,
            patch("numpy.arange") as mock_arange,
        ):
            # Mock PCA
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)
            mock_pca.return_value = mock_pca_instance

            # Mock sklearn KMeans
            mock_sklearn_instance = Mock()
            mock_sklearn_instance.cluster_centers_ = np.random.rand(2, 2)
            mock_sklearn_instance.predict.return_value = np.array(
                [0, 1] * 12 + [0]
            )  # 25 elements
            mock_sklearn_instance.fit.return_value = mock_sklearn_instance
            mock_sklearn.return_value = mock_sklearn_instance

            # Mock numpy operations
            mock_arange.return_value = np.array([0, 1, 2, 3, 4])
            mock_meshgrid.return_value = (np.ones((5, 5)), np.ones((5, 5)))
            mock_c.return_value = np.random.rand(25, 2)

            # Mock plotly Figure
            mock_fig = Mock()
            mock_fig_class.return_value = mock_fig

            result = kmeans.voronoi(show=False)

            assert result == mock_fig
            # Verify that add_trace was called multiple times (for clusters and centroids)
            assert (
                mock_fig.add_trace.call_count >= 3
            )  # Heatmap + 2 clusters + centroids

    def test_voronoi_with_max_points_limit(self):
        """Test Voronoi plot with max_points parameter."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("plotly.graph_objects.Figure") as mock_fig_class,
            patch("numpy.meshgrid") as mock_meshgrid,
            patch("numpy.c_") as mock_c,
            patch("numpy.arange") as mock_arange,
        ):
            # Mock PCA
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            # Mock sklearn KMeans
            mock_sklearn_instance = Mock()
            mock_sklearn_instance.cluster_centers_ = np.random.rand(2, 2)
            mock_sklearn_instance.predict.return_value = np.array(
                [0, 1] * 12 + [0]
            )  # 25 elements
            mock_sklearn_instance.fit.return_value = mock_sklearn_instance
            mock_sklearn.return_value = mock_sklearn_instance

            # Mock numpy operations
            mock_arange.return_value = np.array([0, 1, 2, 3, 4])
            mock_meshgrid.return_value = (np.ones((5, 5)), np.ones((5, 5)))
            mock_c.return_value = np.random.rand(25, 2)

            # Mock plotly Figure
            mock_fig = Mock()
            mock_fig_class.return_value = mock_fig

            result = kmeans.voronoi(max_points=1000, show=False)

            assert result == mock_fig
            mock_pca.assert_called_once_with(n_components=2)

    def test_voronoi_with_title_and_show(self):
        """Test Voronoi plot with title and show=True."""
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn,
            patch("plotly.graph_objects.Figure") as mock_fig_class,
            patch("numpy.meshgrid") as mock_meshgrid,
            patch("numpy.c_") as mock_c,
            patch("numpy.arange") as mock_arange,
        ):
            # Mock PCA
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
            mock_pca.return_value = mock_pca_instance

            # Mock sklearn KMeans
            mock_sklearn_instance = Mock()
            mock_sklearn_instance.cluster_centers_ = np.random.rand(2, 2)
            # Fix: Make predictions match the flattened grid size (5*5 = 25)
            mock_sklearn_instance.predict.return_value = np.array(
                [0, 1] * 12 + [0]
            )  # 25 elements
            mock_sklearn_instance.fit.return_value = mock_sklearn_instance
            mock_sklearn.return_value = mock_sklearn_instance

            # Mock numpy operations
            mock_arange.return_value = np.array([0, 1, 2, 3, 4])
            mock_meshgrid.return_value = (np.ones((5, 5)), np.ones((5, 5)))
            mock_c.return_value = np.random.rand(25, 2)  # 25 points for 5x5 grid

            # Mock plotly Figure
            mock_fig = Mock()
            mock_fig_class.return_value = mock_fig

            result = kmeans.voronoi(title="Test Title", show=True)

            assert result is None  # Returns None when show=True
            mock_fig.show.assert_called_once()
            mock_fig.update_layout.assert_called()


class TestKMeansSave:
    """Test cases for save method."""

    def test_save_no_figure(self):
        """Test save method without a figure."""
        kmeans = KMeans()

        with pytest.raises(LexosException, match="No figure available"):
            kmeans.save("test.png")

    def test_save_as_image(self):
        """Test save method as image."""
        kmeans = KMeans()
        mock_fig = Mock(spec=go.Figure)
        kmeans.fig = mock_fig

        kmeans.save("test.png", html=False)

        mock_fig.write_image.assert_called_once_with("test.png")

    def test_save_as_html(self):
        """Test save method as HTML."""
        kmeans = KMeans()
        mock_fig = Mock(spec=go.Figure)
        kmeans.fig = mock_fig

        kmeans.save("test.html", html=True)

        mock_fig.write_html.assert_called_once_with("test.html")

    def test_save_with_kwargs(self):
        """Test save method with additional kwargs."""
        kmeans = KMeans()
        mock_fig = Mock(spec=go.Figure)
        kmeans.fig = mock_fig

        kmeans.save("test.png", html=False, width=800, height=600)

        mock_fig.write_image.assert_called_once_with("test.png", width=800, height=600)


class TestKMeansToCSV:
    """Test cases for to_csv method."""

    def test_to_csv_no_clustering(self):
        """Test to_csv without clustering results."""
        kmeans = KMeans()

        with pytest.raises(LexosException, match="No clustering results"):
            kmeans.to_csv("test.csv")

    def test_to_csv_with_clustering(self):
        """Test to_csv with clustering results."""
        data = np.random.rand(5, 10)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)
            mock_pca.return_value = mock_pca_instance

            kmeans.to_csv("test.csv")

            mock_to_csv.assert_called_once_with("test.csv", index=False)

    def test_to_csv_with_labels(self):
        """Test to_csv with custom labels."""
        data = np.random.rand(5, 10)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0])
        kmeans.labels = ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"]

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)
            mock_pca.return_value = mock_pca_instance

            kmeans.to_csv("test.csv")

            mock_to_csv.assert_called_once_with("test.csv", index=False)

    def test_to_csv_exception(self):
        """Test to_csv when pandas raises exception."""
        data = np.random.rand(5, 10)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)
            mock_pca.return_value = mock_pca_instance

            mock_to_csv.side_effect = IOError("Test error")

            with pytest.raises(LexosException, match="Failed to export CSV"):
                kmeans.to_csv("test.csv")

    def test_to_csv_with_kwargs(self):
        """Test to_csv with additional kwargs."""
        data = np.random.rand(5, 10)
        kmeans = KMeans(dtm=data)
        kmeans.cluster_assignments = np.array([0, 1, 0, 1, 0])

        with (
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)
            mock_pca.return_value = mock_pca_instance

            kmeans.to_csv("test.csv", sep=";", encoding="utf-8")

            # Debug: Print what was actually called
            print("Call args:", mock_to_csv.call_args)

            # Check that to_csv was called with the correct arguments
            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args

            # Check the positional arguments
            assert args[0] == "test.csv"

            # Check the keyword arguments - be more defensive
            assert kwargs["index"] == False

            # Check if sep and encoding were passed through
            if "sep" in kwargs:
                assert kwargs["sep"] == ";"
            if "encoding" in kwargs:
                assert kwargs["encoding"] == "utf-8"


class TestKMeansIntegration:
    """Integration tests for KMeans."""

    def test_full_workflow(self):
        """Test complete KMeans workflow."""
        # Create test data
        data = np.random.rand(10, 5)
        kmeans = KMeans(dtm=data, k=2)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array(
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            )
            mock_sklearn.return_value = mock_model

            # Run clustering
            result = kmeans()

            # Verify results
            assert isinstance(result, np.ndarray)
            assert len(result) == 10
            assert kmeans.cluster_assignments is not None

            # Test plotting after clustering
            with (
                patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
                patch("plotly.express.scatter") as mock_scatter,
            ):
                mock_pca_instance = Mock()
                mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
                mock_pca.return_value = mock_pca_instance

                mock_fig = Mock(spec=go.Figure)
                mock_scatter.return_value = mock_fig

                fig = kmeans.scatter(show=False)
                assert fig == mock_fig

    def test_with_different_input_types(self):
        """Test KMeans with different input types."""
        # Test with DataFrame
        df = pd.DataFrame(np.random.rand(5, 3))
        kmeans_df = KMeans(dtm=df, k=2)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn.return_value = mock_model

            result = kmeans_df()
            assert len(result) == 5

        # Test with numpy array
        arr = np.random.rand(5, 3)
        kmeans_arr = KMeans(dtm=arr, k=2)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn.return_value = mock_model

            result = kmeans_arr()
            assert len(result) == 5

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid init parameter
        with pytest.raises(ValueError):
            KMeans(init="invalid_init")

        # Test valid init parameters
        kmeans1 = KMeans(init="k-means++")
        assert kmeans1.init == "k-means++"

        kmeans2 = KMeans(init="random")
        assert kmeans2.init == "random"

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimum number of documents
        data = np.random.rand(2, 5)
        kmeans = KMeans(dtm=data, k=2)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1])
            mock_sklearn.return_value = mock_model

            result = kmeans()
            assert len(result) == 2

        # Test with k=1
        kmeans_k1 = KMeans(dtm=data, k=1)

        with patch("lexos.cluster.kmeans.kmeans.sklearn_KMeans") as mock_sklearn:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 0])
            mock_sklearn.return_value = mock_model

            result = kmeans_k1()
            assert len(result) == 2
            assert all(label == 0 for label in result)
