"""Test suite for new_clustermap.py.

Tests for clustermap functionality including Seaborn and Plotly implementations.

Last Updated: July 18, 2025
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.cluster import hierarchy

from lexos.cluster.new_clustermap import (
    Clustermap,
    PlotlyClusterGrid,
    PlotlyClustermap,
    _create_dendrogram_traces,
    get_matrix,
)
from lexos.dtm import DTM
from lexos.exceptions import LexosException


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    data = np.random.rand(5, 10)
    return pd.DataFrame(
        data,
        index=[f"Doc{i + 1}" for i in range(5)],
        columns=[f"Term{i + 1}" for i in range(10)],
    )


@pytest.fixture
def sample_dtm():
    """Create a sample DTM for testing."""
    doc_tokens = [
        ["hello", "world", "test"],
        ["world", "test", "sample"],
        ["hello", "sample", "data"],
    ]
    labels = ["Doc1", "Doc2", "Doc3"]
    dtm = DTM()
    dtm(doc_tokens, labels)
    return dtm


@pytest.fixture
def sample_array():
    """Create a sample numpy array for testing."""
    np.random.seed(42)
    return np.random.rand(4, 6)


@pytest.fixture
def sample_linkage_matrix():
    """Create a sample linkage matrix for testing."""
    np.random.seed(42)
    data = np.random.rand(4, 3)
    return hierarchy.linkage(data, method="average")


class TestGetMatrix:
    """Tests for the get_matrix function."""

    def test_get_matrix_dataframe(self, sample_dataframe):
        """Test get_matrix with DataFrame input."""
        result = get_matrix(sample_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dataframe.shape

    def test_get_matrix_dtm(self, sample_dtm):
        """Test get_matrix with DTM input."""
        result = get_matrix(sample_dtm)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "terms"

    def test_get_matrix_array(self, sample_array):
        """Test get_matrix with numpy array input."""
        result = get_matrix(sample_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_array.shape

    def test_get_matrix_list(self):
        """Test get_matrix with list input."""
        test_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = get_matrix(test_list)
        assert result == test_list

    def test_get_matrix_single_document_raises_error(self):
        """Test that get_matrix raises error for single document."""
        single_doc = pd.DataFrame([[1, 2, 3]])
        with pytest.raises(LexosException, match="must have more than one document"):
            get_matrix(single_doc)

    def test_get_matrix_sparse_dataframe(self, sample_dataframe):
        """Test get_matrix with sparse DataFrame."""
        # Create a mock sparse DataFrame that behaves like a real DataFrame
        mock_sparse_df = MagicMock(spec=pd.DataFrame)
        mock_sparse_df.sparse.to_dense.return_value = sample_dataframe
        mock_sparse_df.shape = sample_dataframe.shape

        # Mock hasattr to return True for the 'sparse' attribute
        with patch("builtins.hasattr") as mock_hasattr:
            mock_hasattr.return_value = True

            # Call get_matrix directly with our mock
            result = get_matrix(mock_sparse_df)

            # Verify the sparse.to_dense() method was called
            mock_sparse_df.sparse.to_dense.assert_called_once()

            # The result should be the densified dataframe
            assert result is sample_dataframe


class TestClustermap:
    """Tests for the Clustermap class."""

    def test_clustermap_initialization_default(self, sample_dataframe):
        """Test Clustermap initialization with default parameters."""
        with (
            patch("seaborn.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = MagicMock()
            mock_grid.figure = MagicMock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe)

            assert cm.metric == "euclidean"
            assert cm.method == "average"
            assert cm.figsize == (8, 8)
            assert cm.fig is not None

    def test_clustermap_with_labels(self, sample_dataframe):
        """Test Clustermap with custom labels."""
        labels = ["Label1", "Label2", "Label3", "Label4", "Label5"]

        with (
            patch("seaborn.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = MagicMock()
            mock_grid.figure = MagicMock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, labels=labels)
            assert cm.labels == labels

    def test_clustermap_hide_dendrograms(self, sample_dataframe):
        """Test hiding dendrograms."""
        with (
            patch("seaborn.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = MagicMock()
            mock_grid.figure = MagicMock()
            mock_grid.ax_col_dendrogram = MagicMock()
            mock_grid.ax_row_dendrogram = MagicMock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, hide_upper=True, hide_side=True)

            mock_grid.ax_col_dendrogram.remove.assert_called_once()
            mock_grid.ax_row_dendrogram.remove.assert_called_once()

    def test_clustermap_with_title(self, sample_dataframe):
        """Test Clustermap with title."""
        title = "Test Clustermap"

        with (
            patch("seaborn.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = MagicMock()
            mock_grid.figure = MagicMock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, title=title)

            mock_grid.figure.suptitle.assert_called_once_with(title, y=1.05)

    def test_clustermap_save(self, sample_dataframe, tmp_path):
        """Test saving clustermap to file."""
        with (
            patch("seaborn.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = MagicMock()
            mock_fig = MagicMock()
            mock_grid.figure = mock_fig
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe)

            save_path = tmp_path / "test_clustermap.png"
            cm.save(save_path, dpi=300)

            mock_fig.savefig.assert_called_once_with(save_path, dpi=300)

    def test_clustermap_invalid_linkage_matrix(self, sample_dataframe):
        """Test Clustermap with invalid linkage matrix."""
        invalid_linkage = np.array([[1, 2]])  # Invalid linkage matrix

        with patch("scipy.cluster.hierarchy.is_valid_linkage") as mock_valid:
            mock_valid.side_effect = TypeError("Invalid linkage")

            with pytest.raises(LexosException, match="Invalid `row_linkage` value"):
                Clustermap(dtm=sample_dataframe, row_linkage=invalid_linkage)


class TestCreateDendrogramTraces:
    """Tests for the _create_dendrogram_traces function."""

    def test_create_dendrogram_traces_basic(self, sample_linkage_matrix):
        """Test basic dendrogram trace creation."""
        traces, dendro_data = _create_dendrogram_traces(sample_linkage_matrix)

        assert isinstance(traces, list)
        assert len(traces) > 0
        assert all(isinstance(trace, go.Scatter) for trace in traces)
        assert isinstance(dendro_data, dict)
        assert "icoord" in dendro_data
        assert "dcoord" in dendro_data

    def test_create_dendrogram_traces_with_labels(self, sample_linkage_matrix):
        """Test dendrogram trace creation with labels."""
        labels = ["A", "B", "C", "D"]
        traces, dendro_data = _create_dendrogram_traces(
            sample_linkage_matrix, labels=labels
        )

        assert isinstance(traces, list)
        assert len(traces) > 0

    def test_create_dendrogram_traces_orientations(self, sample_linkage_matrix):
        """Test different dendrogram orientations."""
        orientations = ["top", "bottom", "left", "right"]

        for orientation in orientations:
            traces, dendro_data = _create_dendrogram_traces(
                sample_linkage_matrix, orientation=orientation
            )
            assert isinstance(traces, list)
            assert len(traces) > 0

    def test_create_dendrogram_traces_custom_styling(self, sample_linkage_matrix):
        """Test dendrogram trace creation with custom styling."""
        traces, dendro_data = _create_dendrogram_traces(
            sample_linkage_matrix, color="rgb(255,0,0)", line_width=2.0
        )

        assert isinstance(traces, list)
        assert len(traces) > 0

        # Check that styling is applied
        for trace in traces:
            assert trace.line.color == "rgb(255,0,0)"
            assert trace.line.width == 2.0


class TestPlotlyClusterGrid:
    """Tests for the PlotlyClusterGrid class."""

    def test_plotly_cluster_grid_initialization(self, sample_dataframe):
        """Test PlotlyClusterGrid initialization."""
        grid = PlotlyClusterGrid(sample_dataframe)

        assert isinstance(grid.data, pd.DataFrame)
        assert isinstance(grid.data2d, pd.DataFrame)
        assert grid.figsize == (800, 600)

    def test_plotly_cluster_grid_with_z_score(self, sample_dataframe):
        """Test PlotlyClusterGrid with z-score normalization."""
        grid = PlotlyClusterGrid(sample_dataframe, z_score=1)

        # Check that z-scoring was applied
        assert not np.allclose(grid.data2d.values, sample_dataframe.values)

    def test_plotly_cluster_grid_with_standard_scale(self, sample_dataframe):
        """Test PlotlyClusterGrid with standard scaling."""
        grid = PlotlyClusterGrid(sample_dataframe, standard_scale=1)

        # Check that standard scaling was applied
        assert not np.allclose(grid.data2d.values, sample_dataframe.values)

    def test_plotly_cluster_grid_z_score_and_standard_scale_error(
        self, sample_dataframe
    ):
        """Test that using both z_score and standard_scale raises error."""
        with pytest.raises(
            ValueError, match="Cannot perform both z-scoring and standard-scaling"
        ):
            PlotlyClusterGrid(sample_dataframe, z_score=1, standard_scale=1)

    def test_plotly_cluster_grid_with_mask(self, sample_dataframe):
        """Test PlotlyClusterGrid with mask."""
        mask = np.zeros(sample_dataframe.shape, dtype=bool)
        mask[0, 0] = True  # Mask first element

        grid = PlotlyClusterGrid(sample_dataframe, mask=mask)

        assert grid.mask is not None
        assert grid.mask.iloc[0, 0] == True

    def test_plotly_cluster_grid_invalid_mask_shape(self, sample_dataframe):
        """Test PlotlyClusterGrid with invalid mask shape."""
        invalid_mask = np.zeros((3, 3), dtype=bool)  # Wrong shape

        with pytest.raises(ValueError, match="Mask must have the same shape as data"):
            PlotlyClusterGrid(sample_dataframe, mask=invalid_mask)

    def test_z_score_method(self, sample_dataframe):
        """Test the _z_score static method."""
        z_scored = PlotlyClusterGrid._z_score(sample_dataframe, axis=1)

        # Check that mean is approximately 0 and std is approximately 1
        assert np.allclose(z_scored.mean().mean(), 0, atol=1e-10)
        assert np.allclose(z_scored.std().mean(), 1, atol=1e-10)

    def test_standard_scale_method(self, sample_dataframe):
        """Test the _standard_scale static method."""
        scaled = PlotlyClusterGrid._standard_scale(sample_dataframe, axis=1)

        # Check that values are between 0 and 1
        assert scaled.min().min() >= 0
        assert scaled.max().max() <= 1


class TestPlotlyClustermap:
    """Tests for the PlotlyClustermap class."""

    def test_plotly_clustermap_initialization(self, sample_dataframe):
        """Test PlotlyClustermap initialization."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2], [2, 3, 1.0, 3]])

            pcm = PlotlyClustermap(dtm=sample_dataframe)

            assert pcm.metric == "euclidean"
            assert pcm.method == "average"
            assert pcm.figsize == (700, 700)
            assert isinstance(pcm.fig, go.Figure)

    def test_plotly_clustermap_with_dtm(self, sample_dtm):
        """Test PlotlyClustermap with DTM input."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dtm)
            assert isinstance(pcm.fig, go.Figure)

    def test_plotly_clustermap_disable_clustering(self, sample_dataframe):
        """Test PlotlyClustermap with clustering disabled."""
        pcm = PlotlyClustermap(
            dtm=sample_dataframe, row_cluster=False, col_cluster=False
        )

        assert isinstance(pcm.fig, go.Figure)
        assert pcm.row_cluster == False
        assert pcm.col_cluster == False

    def test_plotly_clustermap_with_title(self, sample_dataframe):
        """Test PlotlyClustermap with custom title."""
        title = "Test Plotly Clustermap"

        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dataframe, title=title)
            assert pcm.title == title

    def test_plotly_clustermap_save_methods(self, sample_dataframe, tmp_path):
        """Test PlotlyClustermap save methods."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dataframe)

            # Test write_html
            html_path = tmp_path / "test.html"
            with patch.object(pcm.fig, "write_html") as mock_write_html:
                pcm.write_html(html_path)
                mock_write_html.assert_called_once_with(html_path)

            # Test write_image
            img_path = tmp_path / "test.png"
            with patch.object(pcm.fig, "write_image") as mock_write_image:
                pcm.write_image(img_path)
                mock_write_image.assert_called_once_with(img_path)

            # Test save (alias for write_image)
            with patch.object(pcm.fig, "write_image") as mock_write_image:
                pcm.save(img_path)
                mock_write_image.assert_called_once_with(img_path)

    def test_plotly_clustermap_export_methods(self, sample_dataframe):
        """Test PlotlyClustermap export methods."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dataframe)

            # Test to_html
            with patch.object(pcm.fig, "to_html") as mock_to_html:
                mock_to_html.return_value = "<html>test</html>"
                result = pcm.to_html()
                assert result == "<html>test</html>"
                mock_to_html.assert_called_once()

            # Test to_image
            with patch.object(pcm.fig, "to_image") as mock_to_image:
                mock_to_image.return_value = b"fake_image_data"
                result = pcm.to_image()
                assert result == b"fake_image_data"
                mock_to_image.assert_called_once()

    def test_plotly_clustermap_show(self, sample_dataframe):
        """Test PlotlyClustermap show method."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dataframe)

            with patch.object(pcm.fig, "show") as mock_show:
                pcm.show()
                mock_show.assert_called_once()

    def test_plotly_clustermap_with_annotations(self, sample_dataframe):
        """Test PlotlyClustermap with annotations enabled."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dataframe, annot=True, fmt=".3f")

            assert pcm.annot == True
            assert pcm.fmt == ".3f"

    def test_plotly_clustermap_dendrogram_ratio(self, sample_dataframe):
        """Test PlotlyClustermap with different dendrogram ratios."""
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            # Test with tuple
            pcm1 = PlotlyClustermap(dtm=sample_dataframe, dendrogram_ratio=(0.15, 0.25))
            assert pcm1.dendrogram_ratio == (0.15, 0.25)

            # Test with single value
            pcm2 = PlotlyClustermap(dtm=sample_dataframe, dendrogram_ratio=0.3)
            assert pcm2.dendrogram_ratio == 0.3


class TestIntegration:
    """Integration tests for the clustermap module."""

    def test_clustermap_and_plotly_clustermap_comparison(self, sample_dataframe):
        """Test that both implementations can handle the same data."""
        # Test Seaborn version
        with (
            patch("seaborn.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = MagicMock()
            mock_grid.figure = MagicMock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, title="Test")
            assert cm.fig is not None

        # Test Plotly version
        with patch.object(PlotlyClusterGrid, "_calculate_linkage") as mock_linkage:
            mock_linkage.return_value = np.array([[0, 1, 0.5, 2]])

            pcm = PlotlyClustermap(dtm=sample_dataframe, title="Test")
            assert isinstance(pcm.fig, go.Figure)

    def test_real_clustering_workflow(self):
        """Test a realistic clustering workflow with actual data."""
        # Create a small dataset with known structure
        np.random.seed(42)

        # Create two distinct clusters
        cluster1 = np.random.normal(0, 0.1, (3, 5))
        cluster2 = np.random.normal(1, 0.1, (3, 5))
        data = np.vstack([cluster1, cluster2])

        df = pd.DataFrame(
            data,
            index=[f"Doc{i + 1}" for i in range(6)],
            columns=[f"Term{i + 1}" for i in range(5)],
        )

        # Test PlotlyClusterGrid processing
        grid = PlotlyClusterGrid(df, z_score=1)
        assert grid.data2d.shape == df.shape

        # Test that we can create linkage matrices
        linkage_matrix = grid._calculate_linkage(grid.data2d.values)
        assert linkage_matrix.shape[0] == len(df) - 1  # n-1 merges for n observations
        assert linkage_matrix.shape[1] == 4  # Standard linkage matrix format
