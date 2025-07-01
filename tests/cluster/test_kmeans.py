import numpy as np
import pandas as pd
import pytest
from lexos.dtm import DTM
from pydantic import ValidationError
from lexos.cluster.kmeans.kmeans import KMeansCluster
from lexos.exceptions import LexosException
from pydantic import ValidationError
from unittest.mock import patch
from unittest.mock import MagicMock



@pytest.fixture
def sample_data():
    """Fixture for fake DTM data."""
    np.random.seed(42)
    data = np.random.rand(10, 5)  # 10 documents, 5 features
    return pd.DataFrame(data)

def test_kmeans_cluster_runs(sample_data):
    """Test basic clustering works and returns correct length."""
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    assignments = clusterer()
    
    assert isinstance(assignments, np.ndarray)
    assert len(assignments) == len(sample_data)

def test_invalid_k_raises_exception(sample_data):
    """Test that missing k raises an exception."""
    clusterer = KMeansCluster(dtm=sample_data)
    with pytest.raises(LexosException, match="Number of clusters 'k' must be specified"):
        clusterer()

def test_plot_2d_runs(sample_data):
    """Test plot_2d method does not crash."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    fig = clusterer.plot_2d()
    assert fig is not None

def test_plot_3d_runs(sample_data):
    """Test plot_3d method does not crash."""
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

def test_elbow_plot_runs(sample_data):
    """Test that elbow_plot runs without errors."""
    clusterer = KMeansCluster(dtm=sample_data)
    # We won't assert the plot, just make sure it runs
    clusterer.elbow_plot(show=False)

def test_plot_voronoi_runs(sample_data):
    """Test that voronoi plot renders without crash."""
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    clusterer()
    clusterer.plot_voronoi(show=False)

def test_kmeans_fit_fails_with_invalid_k(sample_data):
    """Force kmeans to fail and hit the exception block."""
    clusterer = KMeansCluster(dtm=sample_data, k=0)  # invalid k
    with pytest.raises(LexosException, match="KMeans clustering failed"):
        clusterer()
class DummyDTM:
    def __init__(self):
        self.labels = [f"Doc{i+1}" for i in range(5)]

    def to_df(self, transpose=True):
        return pd.DataFrame(np.random.rand(5, 5))

def test_kmeans_with_real_dtm():
    mat = [[1, 2], [3, 4]]
    labels = ["doc1", "doc2"]
    dtm = DTM(matrix=mat, labels=labels)

    with patch.object(DTM, "to_df", return_value=pd.DataFrame(mat, index=labels)):
        clusterer = KMeansCluster(dtm=dtm, k=2)
        result = clusterer()
        assert isinstance(result, np.ndarray)

def test_kmeans_with_numpy_array():
    """Test clustering with NumPy ndarray input."""
    data = np.random.rand(6, 4)
    clusterer = KMeansCluster(dtm=data, k=2)
    result = clusterer()
    assert len(result) == 6
def test_kmeans_unsupported_input_raises():
    """Simulate an unsupported input and trigger LexosException manually."""
    clusterer = KMeansCluster(k=2, dtm=pd.DataFrame([[1, 2], [3, 4]]))
    
    with patch.object(KMeansCluster, "_get_valid_matrix", side_effect=LexosException("Unsupported input")):
        with pytest.raises(LexosException, match="Unsupported input"):
            clusterer()

def test_kmeans_too_few_documents():
    """Ensure error is raised with fewer than 2 documents."""
    df = pd.DataFrame(np.random.rand(1, 5))
    clusterer = KMeansCluster(dtm=df, k=1)
    with pytest.raises(LexosException, match="Need at least 2 documents"):
        clusterer()
def test_kmeans_call_sets_attrs(sample_data):
    """Check that kwargs passed to __call__ get set."""
    clusterer = KMeansCluster()
    clusterer(dtm=sample_data, k=2, n_init=5, tol=1e-3)
    assert clusterer.k == 2
    assert clusterer.n_init == 5
    assert clusterer.tol == 1e-3
def test_set_attrs_directly(sample_data):
    clusterer = KMeansCluster()
    clusterer._set_attrs(k=3, tol=1e-3, dtm=sample_data)
    assert clusterer.k == 3
    assert clusterer.tol == 1e-3
    assert clusterer.dtm.equals(sample_data)

def test_export_image_without_figure_raises(sample_data):
    """Ensure error is raised when exporting without a generated plot."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.export_image("dummy.png", format="png")

@pytest.fixture
def sample_data():
    """Fixture to create random sample data."""
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(10, 5))

def test_export_html_runs(tmp_path, sample_data):
    """Test exporting plot as HTML."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d()
    path = tmp_path / "plot.html"
    clusterer.export_html(str(path))
    assert path.exists()
def test_export_csv_failure_handling(sample_data):
    """Ensure export_csv error is caught and re-raised as LexosException."""
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    
    with patch("pandas.DataFrame.to_csv", side_effect=Exception("CSV write error")):
        with pytest.raises(LexosException, match="Failed to export CSV: CSV write error"):
            clusterer.export_csv("fake.csv")
def test_kmeans_invalid_input_raises():
    with pytest.raises(ValidationError, match="Input should be"):
        KMeansCluster(dtm="invalid", k=2)
def test_export_image_no_fig_raises():
    clusterer = KMeansCluster(dtm=np.random.rand(3, 3), k=2)
    # clusterer() is NOT called, so plotly_fig stays None
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.export_image("dummy.png")
        
def test_save_png_runs(tmp_path):
    dtm = np.array([[1, 2], [3, 4]])
    clusterer = KMeansCluster(dtm=dtm, k=2)
    clusterer()
    clusterer.plot_2d(show=False)
    clusterer.save_png(tmp_path / "out.png")



def test_save_svg_runs(tmp_path):
    dtm = np.array([[1, 2], [3, 4]])
    clusterer = KMeansCluster(dtm=dtm, k=2)
    clusterer()
    clusterer.plot_2d(show=False)
    clusterer.save_svg(tmp_path / "out.svg")



def test_export_html_runs(tmp_path, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d()
    path = tmp_path / "plot.html"
    clusterer.export_html(path)
    assert path.exists()
def test_invalid_dtm_type():
    with pytest.raises(ValidationError):
        KMeansCluster(dtm="invalid type", k=2)


def test_less_than_two_documents():
    mat = np.array([[1, 2]])  # only 1 document
    with pytest.raises(LexosException, match="at least 2 documents"):
        clusterer = KMeansCluster(dtm=mat, k=2)
        clusterer()

def test_plot_without_clustering():
    mat = np.array([[1, 2], [3, 4]])
    clusterer = KMeansCluster(dtm=mat, k=2)
    with pytest.raises(LexosException, match="run clustering before plotting"):
        clusterer.plot_2d()





def test_export_image_raises_without_plot(tmp_path):
    mat = [[1, 2], [3, 4]]
    dtm = DTM(matrix=mat, labels=["doc1", "doc2"])
    kmeans = KMeansCluster(dtm=dtm, k=2)
    path = tmp_path / "no_image.png"
    with pytest.raises(LexosException, match="No figure available"):
        kmeans.export_image(str(path))


def test_export_csv_raises_without_clustering(tmp_path):
    mat = np.array([[1, 2], [3, 4]])
    clusterer = KMeansCluster(dtm=mat, k=2)
    with pytest.raises(LexosException, match="run clustering first"):
        clusterer.export_csv(tmp_path / "out.csv")



def test_export_csv_runs(tmp_path):
    mat = np.array([[1, 2], [3, 4]])
    clusterer = KMeansCluster(dtm=mat, k=2)
    clusterer()
    clusterer.export_csv(tmp_path / "out.csv")
    assert (tmp_path / "out.csv").exists()



def test_plot_2d_show_and_save(tmp_path, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    save_path = tmp_path / "plot.png"
    clusterer.plot_2d(show=False, save_path=str(save_path))
    assert save_path.exists()
def test_plot_3d_show_and_save(tmp_path, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    save_path = tmp_path / "plot3d.png"
    clusterer.plot_3d(show=False, save_path=str(save_path))
    assert save_path.exists()
def test_plot_voronoi_show_and_save(tmp_path, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    clusterer()
    save_path = tmp_path / "voronoi.png"
    clusterer.plot_voronoi(show=False, save_path=str(save_path))
    assert save_path.exists()
def test_elbow_plot_show_and_save(tmp_path, sample_data):
    clusterer = KMeansCluster(dtm=sample_data)
    save_path = tmp_path / "elbow.png"
    clusterer.elbow_plot(show=False, save_path=str(save_path))
    assert save_path.exists()
@patch("plotly.graph_objects.Figure.show")
def test_plot_2d_show(mock_show, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d(show=True)
    mock_show.assert_called_once()



@patch("plotly.graph_objects.Figure.show")
def test_plot_2d_show(mock_show, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_2d(show=True)
    mock_show.assert_called_once()

@patch("plotly.graph_objects.Figure.show")
def test_plot_3d_show(mock_show, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=2)
    clusterer()
    clusterer.plot_3d(show=True)
    mock_show.assert_called_once()

@patch("plotly.graph_objects.Figure.show")
def test_plot_voronoi_show(mock_show, sample_data):
    clusterer = KMeansCluster(dtm=sample_data, k=3)
    clusterer()
    clusterer.plot_voronoi(show=True)
    mock_show.assert_called_once()
@patch("matplotlib.pyplot.show")
def test_elbow_plot_show(mock_show, sample_data):
    clusterer = KMeansCluster(dtm=sample_data)
    clusterer.elbow_plot(show=True)
    mock_show.assert_called_once()
def test_save_png_without_plot():
    clusterer = KMeansCluster(dtm=np.random.rand(3, 3), k=2)
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.save_png("fake.png")

def test_save_svg_without_plot():
    clusterer = KMeansCluster(dtm=np.random.rand(3, 3), k=2)
    with pytest.raises(LexosException, match="No figure available"):
        clusterer.save_svg("fake.svg")
def test_export_csv_without_clustering(tmp_path):
    dtm = DTM(matrix=[[1, 2], [3, 4]], labels=["a", "b"])
    clusterer = KMeansCluster(dtm=dtm, k=2)
    with pytest.raises(LexosException, match="run clustering first"):
        clusterer.export_csv(tmp_path / "out.csv")
