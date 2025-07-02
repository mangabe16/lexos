"""test_kmeans.py.

Last updated: 2025-07-01
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend for testing
from lexos.dtm import DTM
from lexos.cluster.kmeans import KMeansCluster
from lexos.exceptions import LexosException

@pytest.fixture
def sample_data():
    """Fixture to create reproducible random sample data."""
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(10, 5))

def test_kmeans_cluster_runs(sample_data):
    """Test that clustering returns correct assignments length."""
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    assignments = clusterer()
    assert isinstance(assignments, np.ndarray)
    assert len(assignments) == len(sample_data)

def test_invalid_k_raises_exception(sample_data):
    """Test missing 'k' raises LexosException."""
    clusterer = KMeansCluster(dtm=sample_data)
    with pytest.raises(LexosException, match="Number of clusters 'k' must be specified"):
        clusterer()

def test_plot_2d_runs(sample_data):
    """Test plot_2d executes and returns a figure."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    fig = clusterer.plot_2d()
    assert fig is not None

def test_plot_3d_runs(sample_data):
    """Test plot_3d executes and returns a figure."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    fig = clusterer.plot_3d()
    assert fig is not None

def test_export_csv(tmp_path, sample_data):
    """Test CSV export after clustering."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    out_csv = tmp_path / "output.csv"
    clusterer.export_csv(str(out_csv))
    df = pd.read_csv(out_csv)
    assert "Document" in df.columns
    assert "Cluster" in df.columns

def test_export_csv_raises_without_clustering(tmp_path):
    """Test CSV export without clustering raises LexosException."""
    dtm = DTM(matrix=[[1, 2], [3, 4]], labels=["a", "b"])
    clusterer = KMeansCluster(dtm=dtm, k=2)
    with pytest.raises(LexosException, match="run clustering first"):
        clusterer.export_csv(tmp_path / "out.csv")

def test_export_csv_failure_handling(sample_data):
    """Test export_csv handles internal write failure properly."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    with patch("pandas.DataFrame.to_csv", side_effect=Exception("CSV write error")):
        with pytest.raises(LexosException, match="Failed to export CSV: CSV write error"):
            clusterer.export_csv("fake.csv")

def test_elbow_plot_runs(sample_data):
    """Test elbow plot runs without error."""
    clusterer = KMeansCluster(dtm=sample_data)
    clusterer.elbow_plot(show=False)

def test_elbow_plot_show_and_save(tmp_path, sample_data):
    """Test elbow plot save functionality."""
    clusterer = KMeansCluster(dtm=sample_data)
    path = tmp_path / "elbow.png"
    clusterer.elbow_plot(show=False, save_path=str(path))
    assert path.exists()

def test_plot_voronoi_runs(sample_data):
    """Test that Voronoi plot renders without error."""
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    clusterer()
    clusterer.plot_voronoi(show=False)

def test_plot_voronoi_show_and_save(tmp_path, sample_data):
    """Test Voronoi plot saving functionality."""
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    clusterer()
    path = tmp_path / "voronoi.png"
    clusterer.plot_voronoi(show=False, save_path=str(path))
    assert path.exists()

def test_plot_voronoi_raises_without_k(sample_data):
    """Test that plot_voronoi raises error if k is not set."""
    clusterer = KMeansCluster(dtm=sample_data)
    with pytest.raises(LexosException, match="Number of clusters 'k' must be specified"):
        clusterer.plot_voronoi(show=False)

def test_plot_2d_show_and_save(tmp_path, sample_data):
    """Test 2D plot saving functionality."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    path = tmp_path / "plot.png"
    clusterer.plot_2d(show=False, save_path=str(path))
    assert path.exists()

def test_plot_3d_show_and_save(tmp_path, sample_data):
    """Test 3D plot saving functionality."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    path = tmp_path / "plot3d.png"
    clusterer.plot_3d(show=False, save_path=str(path))
    assert path.exists()

def test_export_html_runs(tmp_path, sample_data):
    """Test exporting plot as HTML."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d()
    path = tmp_path / "plot.html"
    clusterer.export_html(str(path))
    assert path.exists()

def test_export_image_without_fig_raises(sample_data):
    """Test image export without figure raises LexosException."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.export_image("dummy.png")

def test_save_png_runs(tmp_path, sample_data):
    """Test saving plot as PNG."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d(show=False)
    path = tmp_path / "out.png"
    clusterer.save_png(str(path))
    assert path.exists()

def test_save_svg_runs(tmp_path, sample_data):
    """Test saving plot as SVG."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d(show=False)
    path = tmp_path / "out.svg"
    clusterer.save_svg(str(path))
    assert path.exists()

def test_save_png_without_plot(sample_data):
    """Test save_png without plotting raises error."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.save_png("no_plot.png")

def test_save_svg_without_plot(sample_data):
    """Test save_svg without plotting raises error."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.save_svg("no_plot.svg")

def test_export_html_without_plot_raises(sample_data, tmp_path):
    """Test that export_html raises error if no figure has been plotted."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    path = tmp_path / "out.html"
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.export_html(str(path))

def test_export_image_without_plot_raises(sample_data, tmp_path):
    """Test that export_image raises error if no figure has been plotted."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    path = tmp_path / "out.png"
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.export_image(str(path))

def test_export_image_runs(tmp_path, sample_data):
    """Test that export_image works after plotting a figure."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d(show=False)
    out_path = tmp_path / "plot.png"
    clusterer.export_image(str(out_path))
    assert out_path.exists()

def test_plot_3d_without_clustering_raises(sample_data):
    """Test that plot_3d raises if clustering is not yet run."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    with pytest.raises(LexosException, match="clustering before plotting"):
        clusterer.plot_3d()

@patch("plotly.graph_objects.Figure.show")
def test_plot_2d_show(mock_show, sample_data):
    """Test 2D plot show method triggers Plotly show call."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d(show=True)
    mock_show.assert_called_once()

def test_plot_3d_show(sample_data):
    """Test 3D plot show method triggers Plotly show call."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        clusterer.plot_3d(show=True)
        mock_show.assert_called_once()

def test_plot_voronoi_show(sample_data):
    """Test Voronoi plot show method triggers Plotly show call."""
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    clusterer()
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        clusterer.plot_voronoi(show=True)
        mock_show.assert_called_once()

def test_elbow_plot_show(sample_data):
    """Test elbow plot triggers Matplotlib show call."""
    clusterer = KMeansCluster(dtm=sample_data)
    with patch("matplotlib.pyplot.show") as mock_show:
        clusterer.elbow_plot(show=True)
        mock_show.assert_called_once()

def test_elbow_plot_invalid_k_range_triggers_exception(sample_data):
    """Test that elbow_plot raises LexosException when min_k > max_k."""
    clusterer = KMeansCluster(dtm=sample_data)
    k_range = range(11, 15)
    with pytest.raises(LexosException, match="Invalid k range"):
        clusterer.elbow_plot(k_range=k_range, show=False)

def test_elbow_plot_kmeans_fit_fails(sample_data):
    """Test elbow_plot handles internal KMeans fitting failure."""
    clusterer = KMeansCluster(dtm=sample_data)
    with patch("sklearn.cluster._kmeans.KMeans.fit", side_effect=ValueError("bad fit")):
        with pytest.raises(LexosException, match="Error fitting KMeans for k="):
            clusterer.elbow_plot(k_range=range(1, 3), show=False)

def test_elbow_plot_returns_knee(sample_data):
    """Test that elbow_plot returns the optimal k when return_knee=True."""
    clusterer = KMeansCluster(dtm=sample_data)
    result = clusterer.elbow_plot(show=False, return_knee=True)
    assert isinstance(result, int)
    assert result >= 1

def test_kmeans_fit_fails_with_invalid_k(sample_data):
    """Force kmeans to fail and hit the exception block."""
    clusterer = KMeansCluster(dtm=sample_data, k=0)  # invalid k
    with pytest.raises(LexosException, match="KMeans clustering failed"):
        clusterer()

def test_invalid_input_type():
    """Test instantiating with invalid input type raises ValidationError."""
    with pytest.raises(ValidationError, match="Input should be"):
        KMeansCluster(dtm="invalid", k=2)

def test_too_few_documents():
    """Test that clustering fails with fewer than 2 documents."""
    data = pd.DataFrame(np.random.rand(1, 5))
    clusterer = KMeansCluster(dtm=data, k=2)
    with pytest.raises(LexosException, match="at least 2 documents"):
        clusterer()

def test_plot_without_clustering():
    """Test that plotting without running clustering raises error."""
    data = np.array([[1, 2], [3, 4]])
    clusterer = KMeansCluster(dtm=data, k=2)
    with pytest.raises(LexosException, match="run clustering before plotting"):
        clusterer.plot_2d()

def test_kmeans_call_sets_attrs(sample_data):
    """Check that kwargs passed to __call__ get set."""
    clusterer = KMeansCluster()
    clusterer(dtm=sample_data, k=2, n_init=5, tol=1e-3)
    assert clusterer.k == 2
    assert clusterer.n_init == 5
    assert clusterer.tol == 1e-3

def test_set_attrs_directly(sample_data):
    """Test direct attribute setting using _set_attrs."""
    clusterer = KMeansCluster()
    clusterer._set_attrs(k=3, tol=1e-3, dtm=sample_data)
    assert clusterer.k == 3
    assert clusterer.tol == 1e-3
    assert clusterer.dtm.equals(sample_data)

def test_kmeans_unsupported_input_raises():
    """Simulate unsupported input and ensure LexosException is raised."""
    clusterer = KMeansCluster(k=2, dtm=pd.DataFrame([[1, 2], [3, 4]]))
    with patch.object(KMeansCluster, "_get_valid_matrix", side_effect=LexosException("Unsupported input")):
        with pytest.raises(LexosException, match="Unsupported input"):
            clusterer()

def test_get_valid_matrix_with_dtm():
    """Test that a valid DTM input returns correct matrix and sets labels."""
    data = [[1, 2], [3, 4]]
    labels = ["Doc1", "Doc2"]
    dtm = DTM(matrix=data, labels=labels)

    with patch.object(DTM, "to_df", return_value=pd.DataFrame(data, index=labels)):
        clusterer = KMeansCluster(dtm=dtm, k=2)
        matrix = clusterer._get_valid_matrix()

    assert isinstance(matrix, np.ndarray)
    assert clusterer.labels == labels
    assert matrix.shape == (2, 2)

def test_get_valid_matrix_with_dataframe():
    """Test that a pandas DataFrame input returns correct matrix."""
    df = pd.DataFrame([[5, 6], [7, 8]])
    clusterer = KMeansCluster(dtm=df, k=2)
    
    matrix = clusterer._get_valid_matrix()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)

def test_get_valid_matrix_with_ndarray():
    """Test that a NumPy ndarray input returns correct matrix."""
    arr = np.array([[9, 10], [11, 12]])
    clusterer = KMeansCluster(dtm=arr, k=2)

    matrix = clusterer._get_valid_matrix()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)

def test_get_valid_matrix_with_invalid_type():
    """Test that an unsupported input type raises LexosException."""
    clusterer = KMeansCluster(k=2)
    clusterer.dtm = "not_a_matrix" 
    with pytest.raises(LexosException, match="Unsupported input"):
        clusterer._get_valid_matrix()