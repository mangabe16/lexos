import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from lexos.cluster.kmeans.kmeans import KMeansCluster
from lexos.dtm import DTM


class MockVectorizer:
    terms_list = [f"f{i}" for i in range(5)]


def make_mock_dtm(n_docs=10, n_terms=5) -> DTM:
    """Create a mock DTM object with minimal dependencies."""
    X = np.random.rand(n_docs, n_terms)
    dtm = DTM()
    dtm.doc_term_matrix = csr_matrix(X)
    dtm.labels = [f"Doc{i + 1}" for i in range(n_docs)]
    dtm.vectorizer = MockVectorizer()
    return dtm


def test_kmeans_cluster_assignment():
    dtm = make_mock_dtm()
    kmeans = KMeansCluster(dtm=dtm, k=3)
    assignments = kmeans()
    assert len(assignments) == len(dtm.labels)
    assert set(assignments).issubset(set(range(3)))


def test_plot_2d_generation():
    dtm = make_mock_dtm()
    kmeans = KMeansCluster(dtm=dtm, k=3)
    kmeans()
    fig = kmeans.plot_2d()
    assert fig is not None


def test_plot_3d_generation():
    dtm = make_mock_dtm()
    kmeans = KMeansCluster(dtm=dtm, k=3)
    kmeans()
    fig = kmeans.plot_3d()
    assert fig is not None


def test_elbow_plot_executes():
    dtm = make_mock_dtm()
    kmeans = KMeansCluster(dtm=dtm)
    fig = kmeans.elbow_plot(k_range=range(1, 5), show=False)
    assert fig is not None


def test_export_csv(tmp_path):
    dtm = make_mock_dtm()
    kmeans = KMeansCluster(dtm=dtm, k=3)
    kmeans()
    output_path = tmp_path / "clusters.csv"
    kmeans.export_csv(str(output_path))
    df = pd.read_csv(output_path)
    assert df.shape[0] == len(dtm.labels)
    assert "Cluster" in df.columns


def test_save_png_and_svg(tmp_path):
    dtm = make_mock_dtm()
    kmeans = KMeansCluster(dtm=dtm, k=3)
    kmeans()
    kmeans.plot_2d()
    png_path = tmp_path / "plot.png"
    svg_path = tmp_path / "plot.svg"
    kmeans.save_png(str(png_path))
    kmeans.save_svg(str(svg_path))
    assert png_path.exists()
    assert svg_path.exists()
