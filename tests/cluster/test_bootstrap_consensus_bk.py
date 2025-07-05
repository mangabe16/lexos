"""test_bootstrap_consensus.py.

Last Update: March 6, 2025
"""

from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib as mpl  # added
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy
from Bio import Phylo
from matplotlib.figure import Figure
from pydantic import ValidationError
from scipy.cluster.hierarchy import linkage

from lexos.cluster.bootstrap_consensus_bk import BCT
from lexos.dtm import DTM
from lexos.exceptions import LexosException

nlp = spacy.load("en_core_web_sm")
mpl.use("Agg")  # added

# Fixtures


@pytest.fixture
def sample_dtm():
    """Create a sample DTM for testing.

    Returns:
        DTM: Sample DTM with test data
    """
    dtm = DTM()
    dtm(
        docs=[
            nlp("kitten alert"),
            nlp("term1"),
            nlp("Term3"),
            nlp("10term"),
            nlp("2term"),
        ],
        labels=["doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    return dtm


@pytest.fixture
def default_bct():
    """Create a BCT instance with default values.

    Returns:
        BCT: Bootstrap Consensus Tree instance with default settings
    """
    return BCT()


@pytest.fixture
def bct_instance():
    """Create a base BCT instance.

    Returns:
        BCT: Bootstrap Consensus Tree instance
    """
    return BCT()


@pytest.fixture
def sample_data():
    """Create sample data for linkage matrix creation.

    Returns:
        tuple: (data array, labels, linkage matrix)
    """
    data = np.array([[0, 0], [1, 1], [4, 5]])
    labels = ["A", "B", "C"]
    Z = linkage(data, "average")
    return data, labels, Z


@pytest.fixture
def sample_bct():
    """Create a sample BCT instance.

    Returns:
        BCT: Bootstrap Consensus Tree instance with default settings
    """
    return BCT(distance_metric="euclidean", linkage_method="average")


@pytest.fixture
def sample_data2():
    """Create sample data for testing.

    Returns:
        tuple: (labels, sample_dtm)
    """
    labels = ["doc1", "doc2", "doc3"]
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sample_dtm = pd.DataFrame(data, columns=["term1", "term2", "term3"], index=labels)
    return labels, sample_dtm


@pytest.fixture
def bct_instance2(sample_dtm):
    """Create a BCT instance with sample data.

    Args:
        sample_dtm: Sample DTM fixture

    Returns:
        BCT: Configured BCT instance
    """
    return BCT(doc_term_matrix=sample_dtm, iterations=10, replace="without")


@pytest.fixture
def sample_trees():
    """Create sample trees for testing.

    Returns:
        list: List of Phylo trees
    """
    newick_str = "(doc1:0.1,doc2:0.2);"
    return [Phylo.read(StringIO(newick_str), "newick") for _ in range(3)]


@pytest.fixture
def mock_consensus_tree():
    """Create a mock consensus tree.

    Returns:
        Phylo.Tree: Mocked consensus tree
    """
    tree = Phylo.read(StringIO("(A, (B, C), (D, E))"), "newick")
    return tree


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_figure():
    """Create a mock matplotlib figure.

    Returns:
        Mock: Mocked matplotlib figure
    """
    fig = plt.figure()
    return fig


@pytest.fixture
def bct_with_figure(mock_figure):
    """Create BCT instance with a mock figure.

    Args:
        mock_figure: Mock figure fixture

    Returns:
        BCT: BCT instance with mock figure
    """
    bct = BCT()
    bct.fig = Mock()
    bct.fig.savefig = Mock()
    return bct


# Tests


def test_bct_default_initialization(default_bct):
    """Test BCT initialization with default values."""
    assert default_bct.distance_metric == "euclidean"
    assert default_bct.linkage_method == "average"
    assert default_bct.cutoff == 0.5
    assert default_bct.iterations == 100
    assert default_bct.replace == "without"
    assert default_bct.text_color == "rgb(0, 0, 0)"
    assert default_bct.showfig is False
    assert default_bct.fig is None


def test_bct_custom_initialization():
    """Test BCT initialization with custom values."""
    custom_bct = BCT(
        distance_metric="cosine",
        linkage_method="complete",
        cutoff=0.7,
        iterations=200,
        replace="with",
        text_color="rgb(255, 0, 0)",
        showfig=True,
    )

    assert custom_bct.distance_metric == "cosine"
    assert custom_bct.linkage_method == "complete"
    assert custom_bct.cutoff == 0.7
    assert custom_bct.iterations == 200
    assert custom_bct.replace == "with"
    assert custom_bct.text_color == "rgb(255, 0, 0)"
    assert custom_bct.showfig is True


def test_bct_with_dtm(sample_dtm):
    """Test BCT initialization with DTM."""
    bct = BCT(doc_term_matrix=sample_dtm)
    assert isinstance(bct.doc_term_matrix, DTM)
    assert bct.doc_term_matrix is sample_dtm


def test_bct_with_doc_labels():
    """Test BCT initialization with document labels."""
    labels = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    bct = BCT(doc_labels=labels)
    assert bct.doc_labels == labels


def test_bct_with_label_dict():
    """Test BCT initialization with label dictionary."""
    label_dict = {0: "doc1", 1: "doc2", 2: "doc3", 3: "doc4", 4: "doc5"}
    bct = BCT(doc_labels=label_dict)
    assert bct.doc_labels == label_dict


@pytest.mark.parametrize(
    "text_color", ["rgb(0, 0, 0)", "rgb(255, 255, 255)", "rgb(128, 128, 128)"]
)
def test_bct_text_color_validation(text_color):
    """Test text color validation.

    Args:
        text_color: RGB color string to test
    """
    bct = BCT(text_color=text_color)
    assert bct.text_color == text_color


@pytest.mark.parametrize(
    "invalid_color",
    [
        "rgb(256, 0, 0)",  # Invalid RGB value
        "dirt",  # Invalid string
    ],
)
def test_bct_invalid_text_color(invalid_color):
    """Test invalid text color handling.

    Args:
        invalid_color: Invalid color string to test
    """
    with pytest.raises(LexosException):
        BCT(text_color=invalid_color)


def test_bct_invalid_text_color_dtype():
    """Test invalid data type color handling."""
    with pytest.raises(ValidationError):
        BCT(text_color=123)


def test_doc_term_matrix_property_with_valid_dtm(sample_dtm):
    """Test _doc_term_matrix property with valid DTM.

    Args:
        sample_dtm: Fixture providing sample DTM
    """
    bct = BCT(doc_term_matrix=sample_dtm)
    result = bct._doc_term_matrix

    # Check return type
    assert isinstance(result, pd.DataFrame)

    # Check transposition
    expected = sample_dtm.to_df().T
    pd.testing.assert_frame_equal(result, expected)

    # Check index and columns
    assert list(result.index) == ["doc1", "doc2", "doc3", "doc4", "doc5"]
    assert sorted(result.columns.tolist()) == sorted(
        ["kitten", "alert", "term1", "Term3", "10term", "2term"]
    )


def test_doc_term_matrix_property_with_no_dtm():
    """Test _doc_term_matrix property when no DTM is provided."""
    bct = BCT()
    with pytest.raises(LexosException) as exc_info:
        _ = bct._doc_term_matrix
    assert str(exc_info.value) == "No document term matrix found."


def test_doc_term_matrix_property_preserves_values(sample_dtm):
    """Test that _doc_term_matrix property preserves matrix values.

    Args:
        sample_dtm: Fixture providing sample DTM
    """
    bct = BCT(doc_term_matrix=sample_dtm)
    result = bct._doc_term_matrix

    # Check specific values after transposition
    assert result.loc["doc1", "kitten"] == 1
    assert result.loc["doc1", "alert"] == 1
    assert result.loc["doc2", "term1"] == 1
    assert result.loc["doc2", "Term3"] == 0


def test_document_label_map_empty():
    """Test _document_label_map with no labels provided."""
    bct = BCT()
    assert bct._document_label_map == {}


def test_document_label_map_with_dict():
    """Test _document_label_map with dictionary input."""
    label_dict = {0: "doc_a", 1: "doc_b", 2: "doc_c"}
    bct = BCT(doc_labels=label_dict)
    assert bct._document_label_map == label_dict


def test_document_label_map_with_int_list():
    """Test _document_label_map with list of integers."""
    int_list = [0, 1, 2]
    bct = BCT(doc_labels=int_list)
    expected = {0: "doc1", 1: "doc2", 2: "doc3"}
    assert bct._document_label_map == expected


def test_document_label_map_with_str_list():
    """Test _document_label_map with list of strings."""
    str_list = ["doc_a", "doc_b", "doc_c"]
    bct = BCT(doc_labels=str_list)
    expected = {0: "doc_a", 1: "doc_b", 2: "doc_c"}
    assert bct._document_label_map == expected


@pytest.mark.parametrize(
    "labels,expected",
    [
        ([0, 1], {0: "doc1", 1: "doc2"}),
        (["a", "b"], {0: "a", 1: "b"}),
        ({1: "test"}, {1: "test"}),
        ([], {}),
    ],
)
def test_document_label_map_various_inputs(labels, expected):
    """Test _document_label_map with various input types.

    Args:
        labels: Input labels to test
        expected: Expected output mapping
    """
    bct = BCT(doc_labels=labels)
    assert bct._document_label_map == expected


def test_document_label_map_modification():
    """Test _document_label_map behavior when labels are modified."""
    bct = BCT(doc_labels=["a", "b"])
    initial_map = bct._document_label_map
    bct.doc_labels = ["c", "d"]
    assert bct._document_label_map != initial_map
    assert bct._document_label_map == {0: "c", 1: "d"}


def test_linkage_to_newick_basic(sample_data):
    """Test basic Newick tree conversion."""
    _, labels, Z = sample_data
    newick = BCT.linkage_to_newick(Z, labels)

    # Check string type
    assert isinstance(newick, str)
    # Check Newick format requirements
    assert newick.endswith(";")
    assert newick.count("(") == newick.count(")")
    # Check all labels are present
    for label in labels:
        assert label in newick


def test_linkage_to_newick_leaf_nodes(sample_data):
    """Test leaf node handling in Newick conversion."""
    _, labels, Z = sample_data
    newick = BCT.linkage_to_newick(Z, labels)

    # Each label should be followed by a colon and distance
    for label in labels:
        assert f"{label}:" in newick


def test_linkage_to_newick_distances():
    """Test distance calculations in Newick conversion."""
    # Create simple test data with known distances
    data = np.array([[0], [2], [10]])
    labels = ["A", "B", "C"]
    Z = linkage(data, "single")

    newick = BCT.linkage_to_newick(Z, labels)

    # Check that distances are present
    assert ":" in newick
    assert "," in newick


@pytest.mark.parametrize(
    "data,labels",
    [
        (np.array([[0], [1]]), ["A", "B"]),
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), ["A", "B", "C", "D"]),
        (np.array([[0], [1], [2], [3], [4]]), ["A", "B", "C", "D", "E"]),
    ],
)
def test_linkage_to_newick_various_sizes(data, labels):
    """Test Newick conversion with different input sizes.

    Args:
        data: Input data array
        labels: Node labels
    """
    Z = linkage(data, "average")
    newick = BCT.linkage_to_newick(Z, labels)

    # Basic format checks
    assert isinstance(newick, str)
    assert newick.endswith(";")
    # Check all labels are present
    for label in labels:
        assert label in newick


def test_linkage_to_newick_structure():
    """Test hierarchical structure in Newick output."""
    data = np.array([[0], [1], [10]])
    labels = ["A", "B", "C"]
    Z = linkage(data, "single")

    newick = BCT.linkage_to_newick(Z, labels)

    # Check basic structure elements
    assert newick.startswith("(")
    assert "," in newick
    assert ")" in newick
    # Should have n-1 commas for n labels
    assert newick.count(",") == len(labels) - 1


def test_linkage_to_newick_recursive_build():
    """Test recursive tree building with nested structures."""
    data = np.array([[0], [1], [2], [10]])
    labels = ["A", "B", "C", "D"]
    Z = linkage(data, "complete")

    newick = BCT.linkage_to_newick(Z, labels)

    # Check proper nesting
    assert newick.count("(") > 1
    assert newick.count(")") > 1
    # Parentheses should be balanced
    assert newick.count("(") == newick.count(")")


def test_get_newick_tree_return_type(sample_bct, sample_data2):
    """Test return type of _get_newick_tree method."""
    labels, sample_dtm = sample_data2
    result = sample_bct._get_newick_tree(labels, sample_dtm)

    assert isinstance(result, Phylo.BaseTree.Tree)


def test_get_newick_tree_labels(sample_bct, sample_data2):
    """Test if labels are preserved in the tree."""
    labels, sample_dtm = sample_data2
    tree = sample_bct._get_newick_tree(labels, sample_dtm)

    # Get terminal node names from tree
    tree_labels = [terminal.name for terminal in tree.get_terminals()]

    # Check if all input labels are present in tree
    assert set(labels) == set(tree_labels)


def test_get_newick_tree_different_metrics(sample_data2):
    """Test tree generation with different distance metrics."""
    labels, sample_dtm = sample_data2
    metrics = ["euclidean", "cosine", "cityblock"]

    for metric in metrics:
        bct = BCT(distance_metric=metric)
        tree = bct._get_newick_tree(labels, sample_dtm)
        assert isinstance(tree, Phylo.BaseTree.Tree)


def test_get_newick_tree_different_linkage(sample_data2):
    """Test tree generation with different linkage methods."""
    labels, sample_dtm = sample_data2
    methods = ["single", "complete", "average"]

    for method in methods:
        bct = BCT(linkage_method=method)
        tree = bct._get_newick_tree(labels, sample_dtm)
        assert isinstance(tree, Phylo.BaseTree.Tree)


def test_get_newick_tree_single_sample():
    """Test handling of single sample."""
    labels = ["doc1"]
    data = [[1, 2, 3]]
    sample_dtm = pd.DataFrame(data, columns=["term1", "term2", "term3"], index=labels)
    bct = BCT()

    with pytest.raises(ValueError):
        bct._get_newick_tree(labels, sample_dtm)


def test_get_newick_tree_branch_lengths(sample_bct, sample_data2):
    """Test if tree contains valid branch lengths."""
    labels, sample_dtm = sample_data2
    tree = sample_bct._get_newick_tree(labels, sample_dtm)

    # Check if all branches have valid lengths
    for clade in tree.find_clades():
        if clade != tree.root:
            assert hasattr(clade, "branch_length")
            assert clade.branch_length is not None
            assert isinstance(clade.branch_length, (int, float))


def test_get_bootstrap_trees_basic(bct_instance2):
    """Test basic bootstrap tree generation."""
    trees = bct_instance2._get_bootstrap_trees()

    # Check return type and length
    assert isinstance(trees, list)
    assert len(trees) == bct_instance2.iterations
    assert all(isinstance(tree, Phylo.BaseTree.Tree) for tree in trees)


def test_get_bootstrap_trees_labels(bct_instance2):
    """Test if labels are preserved in bootstrap trees."""
    trees = bct_instance2._get_bootstrap_trees()
    expected_labels = set(bct_instance2._doc_term_matrix.index)

    for tree in trees:
        tree_labels = {terminal.name for terminal in tree.get_terminals()}
        assert tree_labels == expected_labels


def test_get_bootstrap_trees_no_dtm():
    """Test error handling when no DTM is provided."""
    bct = BCT()
    with pytest.raises(LexosException, match="No document term matrix found."):
        bct._get_bootstrap_trees()


def test_get_bootstrap_trees_sampling(bct_instance2):
    """Test if sampling works correctly."""
    np.random.seed(42)  # For reproducibility
    trees = bct_instance2._get_bootstrap_trees()

    # Get unique column sets to verify sampling
    original_cols = set(bct_instance2._doc_term_matrix.columns)

    # Mock sampling to verify 80% fraction
    sample = bct_instance2._doc_term_matrix.sample(
        axis=1,
        frac=0.8,
        replace=bct_instance2.replace,
        random_state=np.random.RandomState(42),
    )
    # np.ceil ensures the same rounding as pd.DataFrame.sample
    assert len(sample.columns) == int(np.ceil(0.8 * len(original_cols)))


@pytest.mark.parametrize("iterations", [1, 5, 20])
def test_get_bootstrap_trees_iterations(sample_dtm, iterations):
    """Test different numbers of iterations.

    Args:
        sample_dtm: Sample DTM fixture
        iterations: Number of iterations to test
    """
    bct = BCT(doc_term_matrix=sample_dtm, iterations=iterations)
    trees = bct._get_bootstrap_trees()
    assert len(trees) == iterations


@pytest.mark.parametrize("replace", ["with", "without"])
def test_get_bootstrap_trees_replacement(sample_dtm, replace):
    """Test different replacement strategies.

    Args:
        sample_dtm: Sample DTM fixture
        replace: Replacement strategy to test
    """
    bct = BCT(doc_term_matrix=sample_dtm, replace=replace)
    trees = bct._get_bootstrap_trees()
    assert all(isinstance(tree, Phylo.BaseTree.Tree) for tree in trees)


def test_get_bootstrap_consensus_tree_basic(sample_dtm, sample_trees):
    """Test basic consensus tree generation."""
    bct = BCT(doc_term_matrix=sample_dtm, iterations=5)

    with patch.object(bct, "_get_bootstrap_trees", return_value=sample_trees):
        consensus_tree = bct._get_bootstrap_consensus_tree()

        assert isinstance(consensus_tree, Phylo.BaseTree.Tree)
        assert consensus_tree.count_terminals() > 0


@pytest.mark.parametrize("cutoff", [0.3, 0.5, 0.7])
def test_get_bootstrap_consensus_tree_cutoffs(sample_dtm, cutoff, sample_trees):
    """Test consensus tree with different cutoff values.

    Args:
        sample_dtm: Sample DTM fixture
        cutoff: Cutoff value to test
    """
    bct = BCT(doc_term_matrix=sample_dtm, cutoff=cutoff, iterations=5)

    with patch.object(bct, "_get_bootstrap_trees", return_value=sample_trees):
        consensus_tree = bct._get_bootstrap_consensus_tree()
        assert isinstance(consensus_tree, Phylo.BaseTree.Tree)


def test_get_bootstrap_consensus_tree_labels(sample_dtm, sample_trees):
    """Test if labels are preserved in consensus tree."""
    bct = BCT(doc_term_matrix=sample_dtm, iterations=5)
    expected_labels = set(["doc1", "doc2", "doc3"])

    with patch.object(bct, "_get_bootstrap_trees", return_value=sample_trees):
        consensus_tree = bct._get_bootstrap_consensus_tree()
        tree_labels = {terminal.name for terminal in consensus_tree.get_terminals()}
        assert tree_labels.issubset(expected_labels)


def test_get_bootstrap_consensus_tree_branch_lengths(sample_dtm, sample_trees):
    """Test if consensus tree has valid branch lengths."""
    bct = BCT(doc_term_matrix=sample_dtm, iterations=5)

    with patch.object(bct, "_get_bootstrap_trees", return_value=sample_trees):
        consensus_tree = bct._get_bootstrap_consensus_tree()

        for clade in consensus_tree.find_clades():
            if clade != consensus_tree.root:
                assert hasattr(clade, "branch_length")
                assert isinstance(clade.branch_length, (int, float))


def test_get_bootstrap_consensus_tree_fig_basic(sample_dtm, mock_consensus_tree):
    """Test basic figure generation."""
    bct = BCT(doc_term_matrix=sample_dtm)

    with patch.object(
        bct, "_get_bootstrap_consensus_tree", return_value=mock_consensus_tree
    ):
        fig = bct._get_bootstrap_consensus_tree_fig()

        assert isinstance(fig, Figure)
        # NOTE: Tight layout is not applied with test data
        assert plt.gcf().get_size_inches()[0] == 6.4
        assert plt.gcf().get_size_inches()[1] > 0


def test_get_bootstrap_consensus_tree_fig_color(sample_dtm, mock_consensus_tree):
    """Test color handling in figure generation."""
    test_color = "rgb(100,150,200)"
    bct = BCT(doc_term_matrix=sample_dtm, text_color=test_color)

    with patch.object(
        bct, "_get_bootstrap_consensus_tree", return_value=mock_consensus_tree
    ):
        fig = bct._get_bootstrap_consensus_tree_fig()

        # Check if color was properly normalized
        expected_color = tuple(x / 255 for x in [100, 150, 200])
        assert fig.axes[0].xaxis.label.get_color() == expected_color
        assert fig.axes[0].yaxis.label.get_color() == expected_color


def test_get_bootstrap_consensus_tree_fig_spines(sample_dtm, mock_consensus_tree):
    """Test spine visibility settings."""
    bct = BCT(doc_term_matrix=sample_dtm)

    with patch.object(
        bct, "_get_bootstrap_consensus_tree", return_value=mock_consensus_tree
    ):
        fig = bct._get_bootstrap_consensus_tree_fig()
        ax = fig.axes[0]

        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["bottom"].get_visible()
        assert ax.spines["left"].get_visible()


@pytest.mark.parametrize("showfig", [True, False])
def test_get_bootstrap_consensus_tree_fig_show_option(
    sample_dtm, showfig, mock_consensus_tree
):
    """Test showfig parameter behavior.

    Args:
        sample_dtm: Sample DTM fixture
        showfig: Boolean to test show behavior
    """
    bct = BCT(doc_term_matrix=sample_dtm, showfig=showfig)

    with patch.object(
        bct, "_get_bootstrap_consensus_tree", return_value=mock_consensus_tree
    ):
        fig = bct._get_bootstrap_consensus_tree_fig()
        assert isinstance(fig, Figure)


def test_get_bootstrap_consensus_tree_fig_labels(sample_dtm, mock_consensus_tree):
    """Test figure labels and title."""
    bct = BCT(doc_term_matrix=sample_dtm)

    with patch.object(
        bct, "_get_bootstrap_consensus_tree", return_value=mock_consensus_tree
    ):
        fig = bct._get_bootstrap_consensus_tree_fig()

        assert fig.axes[0].get_xlabel() == "Branch Length"
        assert fig.axes[0].get_ylabel() == "Documents"
        assert fig.axes[0].get_title() == "Bootstrap Consensus Tree Result"


def test_call_basic_functionality(sample_dtm):
    """Test basic call functionality with default parameters."""
    bct = BCT()
    fig = bct(doc_term_matrix=sample_dtm)
    assert isinstance(fig, Figure)
    assert isinstance(bct.fig, Figure)
    assert bct.doc_term_matrix == sample_dtm


def test_call_custom_parameters(sample_dtm):
    """Test call with custom parameters."""
    params = {
        "doc_term_matrix": sample_dtm,
        "distance_metric": "cosine",
        "linkage_method": "complete",
        "cutoff": 0.7,
        "iterations": 50,
        "replace": "with",
        "text_color": "rgb(100,150,200)",
        "showfig": True,
    }

    bct = BCT()
    with patch.object(
        BCT, "_get_bootstrap_consensus_tree_fig", return_value=mock_figure
    ):
        fig = bct(**params)

        # Verify all parameters were set correctly
        for key, value in params.items():
            assert getattr(bct, key) == value


@pytest.mark.parametrize("showfig", [True, False])
def test_call_show_behavior(sample_dtm, showfig):
    """Test figure display behavior.

    Args:
        sample_dtm: Sample DTM fixture
        showfig: Boolean to test show behavior
    """
    bct = BCT()
    with patch("matplotlib.pyplot.ioff") as mock_ioff:
        with patch.object(
            BCT, "_get_bootstrap_consensus_tree_fig", return_value=mock_figure
        ):
            bct(doc_term_matrix=sample_dtm, showfig=showfig)

            if not showfig:
                mock_ioff.assert_called_once()
            else:
                mock_ioff.assert_not_called()


def test_call_figure_generation(sample_dtm):
    """Test that figure is generated and stored."""
    bct = BCT()
    mock_fig = mock_figure

    with patch.object(BCT, "_get_bootstrap_consensus_tree_fig", return_value=mock_fig):
        bct(doc_term_matrix=sample_dtm)
        assert bct.fig == mock_fig


def test_call_no_dtm():
    """Test call behavior without DTM."""
    bct = BCT()
    with patch.object(
        BCT, "_get_bootstrap_consensus_tree_fig", return_value=mock_figure
    ):
        # Should not raise an error as DTM is optional
        bct()
        assert bct.doc_term_matrix is None


def test_save_valid_path(bct_with_figure, tmp_path):
    """Test saving figure with valid path.

    Args:
        bct_with_figure: BCT instance with mock figure
        tmp_path: pytest temporary path fixture
    """
    save_path = Path(tmp_path) / "test.png"
    bct_with_figure.save(save_path)
    bct_with_figure.fig.savefig.assert_called_once_with(save_path)


def test_save_string_path(bct_with_figure, tmp_path):
    """Test saving figure with string path.

    Args:
        bct_with_figure: BCT instance with mock figure
        tmp_path: pytest temporary path fixture
    """
    save_path = str(tmp_path / "test.png")
    bct_with_figure.save(save_path)
    bct_with_figure.fig.savefig.assert_called_once_with(save_path)


def test_save_invalid_path(bct_with_figure):
    """Test saving figure with invalid path raises exception.

    Args:
        bct_with_figure: BCT instance with mock figure
        invalid_path: Invalid path value to test
    """
    invalid_path = ""
    with pytest.raises(LexosException, match="You must provide a valid path."):
        bct_with_figure.save(invalid_path)


def test_show_with_figure(bct_with_figure):
    """Test show method with existing figure."""
    with patch("matplotlib.pyplot.ion") as mock_ion:
        result = bct_with_figure.show()

        mock_ion.assert_called_once()
        assert result == bct_with_figure.fig


def test_show_without_figure():
    """Test show method without figure raises exception."""
    bct = BCT()

    with pytest.raises(
        LexosException, match="You must call the instance before showing the figure."
    ):
        bct.show()
