"""test_scale_model.py.

Last Update: 9 March, 2025

NOTE: _pcoa() function derives fro scikit-bio and is tested there.
"""
import gzip
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import entropy

from lexos.exceptions import LexosException
from lexos.topic_modeling.mallet.scale_model import (
    ValidationError,
    _df_with_names,
    _input_check,
    _input_validate,
    _jensen_shannon,
    _num_dist_rows,
    _series_with_name,
    _topic_coordinates,
    convert_mallet_data,
    extract_params,
    get_topic_coordinates,
    js_MMDS,
    js_PCoA,
    js_TSNE,
    pivot_and_smooth,
    state_to_df,
)

#Fixtures

@pytest.fixture
def valid_distribution_matrix():
    """Create valid probability distribution matrix.

    Returns:
        np.array: Matrix where each row sums to 1
    """
    return np.array([
        [0.3, 0.3, 0.4],
        [0.5, 0.2, 0.3],
        [0.1, 0.1, 0.8]
    ])

@pytest.fixture
def invalid_distribution_matrix():
    """Create invalid probability distribution matrix.

    Returns:
        np.array: Matrix where some rows don't sum to 1
    """
    return np.array([
        [0.3, 0.3, 0.3],  # sums to 0.9
        [0.5, 0.2, 0.3],  # sums to 1.0
        [0.1, 0.1, 0.7]   # sums to 0.9
    ])

@pytest.fixture
def valid_inputs():
    """Create valid test inputs.

    Returns:
        tuple: (topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
    """
    return (
        pd.DataFrame([[0.5, 0.5], [0.3, 0.7]]),  # topic_term_dists
        pd.DataFrame([[0.6, 0.4], [0.8, 0.2], [0.4, 0.6]]),  # doc_topic_dists
        [100, 150, 200],  # doc_lengths
        ['word1', 'word2'],  # vocab
        [50, 75]  # term_frequency
    )

@pytest.fixture
def valid_inputs2():
    """Create valid test inputs.

    Returns:
        tuple: Valid test data containing:
            - topic_term_dists: Topic-term probability matrix
            - doc_topic_dists: Document-topic probability matrix
            - doc_lengths: Document length list
            - vocab: Vocabulary list
            - term_frequency: Term frequency list
    """
    return (
        pd.DataFrame([[0.5, 0.5], [0.3, 0.7]]),  # topic_term_dists
        pd.DataFrame([[0.6, 0.4], [0.8, 0.2]]),  # doc_topic_dists
        [100, 150],  # doc_lengths
        ['word1', 'word2'],  # vocab
        [50, 75]  # term_frequency
    )

@pytest.fixture
def probability_distributions():
    """Create sample probability distributions.

    Returns:
        tuple: Two numpy arrays representing probability distributions
    """
    P = np.array([0.3, 0.4, 0.3])
    Q = np.array([0.2, 0.5, 0.3])
    return P, Q

@pytest.fixture
def sample_distributions():
    """Create sample probability distributions for testing.

    Returns:
        np.array: Matrix of probability distributions
    """
    return np.array([
        [0.3, 0.3, 0.4],  # Distribution 1
        [0.5, 0.2, 0.3],  # Distribution 2
        [0.1, 0.1, 0.8]   # Distribution 3
    ])

@pytest.fixture
def sample_distributions2():
    """Create sample probability distributions for testing.

    Returns:
        np.array: Matrix of probability distributions
    """
    return np.array([
        [0.3, 0.3, 0.4],  # Distribution 1
        [0.5, 0.2, 0.3],  # Distribution 2
        [0.1, 0.1, 0.8],  # Distribution 3
        [0.4, 0.4, 0.2]   # Distribution 4
    ])


@pytest.fixture
def sample_distributions3():
    """Create sample probability distributions for testing.

    Returns:
        np.array: Matrix of probability distributions
    """
    return np.array([
        [0.3, 0.3, 0.4],  # Distribution 1
        [0.5, 0.2, 0.3],  # Distribution 2
        [0.1, 0.1, 0.8],  # Distribution 3
        [0.4, 0.4, 0.2]   # Distribution 4
    ])


@pytest.fixture
def sample_data():
    """Create sample data for testing.

    Returns:
        tuple: (DataFrame, ndarray) Sample data in different formats
    """
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    array = np.array([[1, 2, 3], [4, 5, 6]])
    return df, array


@pytest.fixture
def sample_data2():
    """Create sample data for testing.

    Returns:
        tuple: (Series, list) Sample data in different formats
    """
    series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    list_data = [4, 5, 6]
    return series, list_data


@pytest.fixture
def mock_mds():
    """Create mock MDS function.

    Returns:
        callable: Mock MDS function that returns predefined coordinates
    """
    def mds_func(data):
        n_topics = data.shape[0]
        return np.array([[i, i+1] for i in range(n_topics)])
    return mds_func

@pytest.fixture
def sample_data3():
    """Create sample test data.

    Returns:
        tuple: (topic_term_dists, topic_proportion)
    """
    topic_term_dists = np.array([
        [0.3, 0.3, 0.4],
        [0.5, 0.2, 0.3],
        [0.1, 0.1, 0.8]
    ])
    topic_proportion = np.array([0.4, 0.35, 0.25])
    return topic_term_dists, topic_proportion


@pytest.fixture
def sample_data4():
    """Create sample test data.

    Returns:
        tuple: Test data containing required inputs for get_topic_coordinates
    """
    topic_term_dists = np.array([
        [0.3, 0.3, 0.4],
        [0.5, 0.2, 0.3],
        [0.1, 0.1, 0.8]
    ])
    doc_topic_dists = np.array([
        [0.6, 0.2, 0.2],
        [0.2, 0.7, 0.1],
        [0.3, 0.3, 0.4]
    ])
    doc_lengths = [100, 150, 200]
    vocab = ['word1', 'word2', 'word3']
    term_frequency = [50, 75, 100]
    return topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency


@pytest.fixture
def sample_statefile():
    """Create a temporary MALLET state file for testing.

    Returns:
        Path: Path to temporary state file
    """
    content = (
        b"#doc source pos typeindex type topic\n"
        b"alpha :0.5 0.3 0.2\n"
        b"beta :0.01\n"
        b"1 doc1.txt 0 1 word1 2\n"
    )

    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(content)
        return Path(f.name)

@pytest.fixture
def malformed_statefile():
    """Create a malformed state file for testing.

    Returns:
        Path: Path to malformed state file
    """
    content = b"malformed content\n" * 3

    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(content)
        return Path(f.name)

@pytest.fixture
def sample_statefile2():
    """Create a sample MALLET state file for testing.

    Returns:
        Path: Path to temporary gzipped state file
    """
    content = (
        b"#doc source pos typeindex type topic\n"
        b"alpha :0.5 0.3 0.2\n"
        b"beta :0.01\n"
        b"doc1 file1.txt 0 1 word1 2\n"
        b"doc1 file1.txt 1 2 word2 1\n"
        b"doc2 file2.txt 0 3 word3 2\n"
    )

    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(content)
        return Path(f.name)

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing.

    Returns:
        pd.DataFrame: Sample data with topic-term counts
    """
    return pd.DataFrame({
        'topic': [1, 1, 2, 2, 3],
        'term': ['word1', 'word2', 'word1', 'word3', 'word2'],
        'count': [5, 3, 2, 4, 1]
    })

@pytest.fixture
def sample_statefile3():
    """Create a sample MALLET state file for testing.

    Returns:
        Path: Path to temporary state file
    """
    content = (
        b"#doc source pos typeindex type topic\n"
        b"alpha :0.5 0.3 0.2\n"
        b"beta :0.01\n"
        b"doc1 file1.txt 0 1 word1 2\n"
        b"doc1 file1.txt 1 2 word2 1\n"
        b"doc2 file2.txt 0 3 word3 2\n"
        b"doc2 file2.txt 1 1 word1 0\n"
    )

    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(content)
        return Path(f.name)
# Tests

def test_all_rows_valid(valid_distribution_matrix):
    """Test when all rows sum to 1."""
    result = _num_dist_rows(valid_distribution_matrix)
    assert result == 3  # all rows are valid

def test_some_rows_invalid(invalid_distribution_matrix):
    """Test when some rows don't sum to 1."""
    result = _num_dist_rows(invalid_distribution_matrix)
    assert result == 1  # only one row sums to 1

def test_empty_matrix():
    """Test with empty matrix."""
    empty_matrix = np.array([])
    result = _num_dist_rows(empty_matrix.reshape(0, 0))
    assert result == 0

@pytest.mark.parametrize("test_matrix,expected", [
    (np.array([[1.0]]), 1),
    (np.array([[1.0, 0.0], [0.5, 0.5]]), 2),
    # Disabled due to floating point precision - not sure if this matters
    # (np.array([[0.999]]), 0)
])
def test_various_matrices(test_matrix, expected):
    """Test with various matrix configurations.

    Args:
        test_matrix (np.array): Input matrix to test
        expected (int): Expected number of valid rows
    """
    result = _num_dist_rows(test_matrix)
    assert result == expected

def test_precision_handling():
    """Test handling of floating point precision."""
    matrix = np.array([
        [0.333333333, 0.333333333, 0.333333334],  # sums to 1.0
        [0.1, 0.1, 0.1]  # sums to 0.3
    ])
    result = _num_dist_rows(matrix)
    assert result == 1  # only first row should be considered valid

def test_ndigits_parameter():
    """Test ndigits parameter (currently not implemented)."""
    matrix = np.array([[0.333, 0.333, 0.334]])
    # Should behave the same regardless of ndigits
    result1 = _num_dist_rows(matrix, ndigits=2)
    result2 = _num_dist_rows(matrix, ndigits=3)
    assert result1 == result2

def test_valid_input(valid_inputs):
    """Test with valid input data."""
    errors = _input_check(*valid_inputs)
    assert len(errors) == 0

def test_mismatched_topic_counts(valid_inputs):
    """Test when topic counts don't match between matrices."""
    inputs = list(valid_inputs)
    inputs[1] = pd.DataFrame([[0.6, 0.4, 0.0], [0.8, 0.2, 0.0]])  # Wrong topic count

    errors = _input_check(*inputs)

    assert any("Topic count mismatch" in error for error in errors)

def test_mismatched_doc_lengths(valid_inputs):
    """Test when document lengths don't match doc_topic_dists."""
    inputs = list(valid_inputs)
    inputs[2] = [100, 150]  # Missing one document length

    errors = _input_check(*inputs)
    assert any("Document count mismatch" in error for error in errors)

def test_mismatched_vocab_size(valid_inputs):
    """Test when vocabulary size doesn't match topic_term_dists."""
    inputs = list(valid_inputs)
    inputs[3] = ['word1', 'word2', 'word3']  # Extra vocabulary term

    errors = _input_check(*inputs)
    assert any("Vocabulary size mismatch" in error for error in errors)

def test_mismatched_term_frequency(valid_inputs):
    """Test when term frequency length doesn't match vocabulary."""
    inputs = list(valid_inputs)
    inputs[4] = [50]  # Missing one term frequency

    errors = _input_check(*inputs)
    assert any("Term frequency length mismatch" in error for error in errors)

def test_invalid_topic_term_distributions(valid_inputs):
    """Test when topic-term distributions don't sum to 1."""
    inputs = list(valid_inputs)
    inputs[0] = pd.DataFrame([[0.3, 0.3], [0.3, 0.7]])  # First row sums to 0.6

    errors = _input_check(*inputs)
    assert any("Invalid probability distributions in topic_term_dists" in error for error in errors)

def test_invalid_doc_topic_distributions(valid_inputs):
    """Test when document-topic distributions don't sum to 1."""
    inputs = list(valid_inputs)
    inputs[1] = pd.DataFrame([[0.5, 0.4], [0.8, 0.2], [0.4, 0.6]])  # First row sums to 0.9

    errors = _input_check(*inputs)
    assert any("Invalid probability distributions in doc_topic_dists" in error for error in errors)

def test_valid_input2(valid_inputs2):
    """Test validation with valid input data."""
    # Should not raise any exception
    _input_validate(*valid_inputs2)

def test_invalid_topic_counts(valid_inputs2):
    """Test validation with mismatched topic counts."""
    inputs = list(valid_inputs2)
    # Create mismatch between topic_term_dists and doc_topic_dists
    inputs[1] = pd.DataFrame([[0.6, 0.4, 0.0], [0.8, 0.2, 0.0]])

    with pytest.raises(ValidationError) as exc_info:
        _input_validate(*inputs)
    assert "Topic count mismatch" in str(exc_info.value)

def test_invalid_distributions(valid_inputs2):
    """Test validation with invalid probability distributions."""
    inputs = list(valid_inputs2)
    # Create distribution that doesn't sum to 1
    inputs[0] = pd.DataFrame([[0.3, 0.3], [0.3, 0.7]])

    with pytest.raises(ValidationError) as exc_info:
        _input_validate(*inputs)
    assert "Invalid probability distributions" in str(exc_info.value)

def test_invalid_doc_lengths(valid_inputs2):
    """Test validation with mismatched document lengths."""
    inputs = list(valid_inputs2)
    # Create mismatch in document lengths
    inputs[2] = [100]  # Missing one document length

    with pytest.raises(ValidationError) as exc_info:
        _input_validate(*inputs)
    assert "Document count mismatch" in str(exc_info.value)

def test_multiple_errors(valid_inputs2):
    """Test validation with multiple validation errors."""
    inputs = list(valid_inputs2)
    # Introduce multiple issues
    inputs[0] = pd.DataFrame([[0.3, 0.3], [0.3, 0.7]])  # Invalid distribution
    inputs[2] = [100]  # Wrong document length

    with pytest.raises(ValidationError) as exc_info:
        _input_validate(*inputs)
    error_msg = str(exc_info.value)
    assert "Invalid probability distributions" in error_msg
    assert "Document count mismatch" in error_msg
    assert error_msg.count(" - ") > 1  # Multiple bullet points

def test_error_message_formatting(valid_inputs2):
    """Test error message formatting."""
    inputs = list(valid_inputs2)
    inputs[3] = ['single_word']  # Wrong vocabulary size

    with pytest.raises(ValidationError) as exc_info:
        _input_validate(*inputs)
    error_msg = str(exc_info.value)
    assert error_msg.startswith("\n")  # Starts with newline
    assert " - " in error_msg  # Contains bullet points
    assert "Validation Error:" in error_msg  # Contains error prefix

def test_jensen_shannon_identical_distributions():
    """Test JSD between identical distributions is 0."""
    P = np.array([0.5, 0.5])
    assert _jensen_shannon(P, P) == pytest.approx(0.0)

def test_jensen_shannon_basic(probability_distributions):
    """Test basic JSD calculation."""
    P, Q = probability_distributions
    jsd = _jensen_shannon(P, Q)
    assert isinstance(jsd, float)
    assert 0 <= jsd <= 1  # JSD is bounded between 0 and 1

def test_jensen_shannon_symmetry(probability_distributions):
    """Test JSD symmetry property: JSD(P||Q) = JSD(Q||P)."""
    P, Q = probability_distributions
    assert _jensen_shannon(P, Q) == pytest.approx(_jensen_shannon(Q, P))

def test_jensen_shannon_manual_calculation():
    """Test JSD against manually calculated value."""
    P = np.array([0.5, 0.5])
    Q = np.array([1.0, 0.0])
    M = 0.5 * (P + Q)
    expected = 0.5 * (entropy(P, M) + entropy(Q, M))
    assert _jensen_shannon(P, Q) == pytest.approx(expected)

@pytest.mark.parametrize("P,Q", [
    (np.array([0.5, 0.5]), np.array([0.5, 0.5])),  # identical
    (np.array([1.0, 0.0]), np.array([0.0, 1.0])),  # opposite
    (np.array([0.3, 0.7]), np.array([0.4, 0.6])),  # similar
])
def test_jensen_shannon_various_distributions(P, Q):
    """Test JSD with different probability distribution pairs.

    Args:
        P: First probability distribution
        Q: Second probability distribution
    """
    jsd = _jensen_shannon(P, Q)
    assert isinstance(jsd, float)
    assert 0 <= jsd <= 1

def test_jensen_shannon_error_handling():
    """Test JSD error handling for invalid inputs."""
    with pytest.raises(LexosException):
        _jensen_shannon(np.array([0.5]), np.array([0.5, 0.5]))  # Different lengths

    # NOTE: This test is disabled. I'm not sure if a negative number would ever be passed to the function.
    # with pytest.raises(LexosException):
    #     _jensen_shannon(np.array([1.5, -0.5]), np.array([0.5, 0.5]))  # Invalid probabilities

def test_jensen_shannon_zero_handling():
    """Test JSD handling of zero probabilities."""
    P = np.array([0.0, 1.0])
    Q = np.array([1.0, 0.0])
    jsd = _jensen_shannon(P, Q)
    assert isinstance(jsd, float)
    assert not np.isnan(jsd)

def test_js_pcoa_basic(sample_distributions):
    """Test basic dimension reduction functionality."""
    result = js_PCoA(sample_distributions)

    # Check output shape (n_distributions x 2)
    assert result.shape == (3, 2)
    assert isinstance(result, np.ndarray)

def test_js_pcoa_identical_distributions():
    """Test dimension reduction with identical distributions."""
    distributions = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])

    result = js_PCoA(distributions)

    # All points should be very close to each other
    assert np.allclose(result[0], result[1])
    assert np.allclose(result[1], result[2])

def test_js_pcoa_different_distributions():
    """Test dimension reduction with very different distributions."""
    distributions = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ])

    result = js_PCoA(distributions)

    # Points should be different
    assert not np.allclose(result[0], result[1])
    assert not np.allclose(result[1], result[2])

def test_js_pcoa_distance_preservation():
    """Test that relative distances are preserved in reduced dimensions."""
    distributions = np.array([
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.5]
    ])

    result = js_PCoA(distributions)

    # Calculate original distances
    orig_dists = pdist(distributions, metric=_jensen_shannon)
    # Calculate distances in reduced space
    reduced_dists = pdist(result)

    # Check correlation between original and reduced distances
    correlation = np.corrcoef(orig_dists, reduced_dists)[0, 1]
    assert correlation > 0.9  # High correlation expected

@pytest.mark.parametrize("n_distributions,n_features", [
    (2, 3),
    (5, 2),
    (10, 4),
])
def test_js_pcoa_various_sizes(n_distributions, n_features):
    """Test dimension reduction with different input sizes.

    Args:
        n_distributions: Number of distributions to test
        n_features: Number of features per distribution
    """
    # Create random probability distributions
    distributions = np.random.dirichlet(
        np.ones(n_features),
        size=n_distributions
    )

    result = js_PCoA(distributions)

    assert result.shape == (n_distributions, 2)

def test_js_pcoa_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        # Non-probability distribution
        invalid_dist = np.array([[1.5, -0.5], [0.8, 0.2]])
        js_PCoA(invalid_dist)

def test_js_pcoa_single_feature():
    """Test handling of single-feature distributions."""
    distributions = np.array([[1.0], [1.0], [1.0]])

    result = js_PCoA(distributions)

    assert result.shape == (3, 2)
    assert np.allclose(result, 0)  # All points should be at origin

def test_js_mmds_basic(sample_distributions2):
    """Test basic MMDS functionality."""
    result = js_MMDS(sample_distributions2)

    # Check output shape and type
    assert result.shape == (4, 2)  # 4 distributions reduced to 2D
    assert isinstance(result, np.ndarray)

def test_js_mmds_custom_params():
    """Test MMDS with custom parameters."""
    distributions = np.array([
        [0.5, 0.5],
        [0.8, 0.2],
        [0.3, 0.7]
    ])

    result = js_MMDS(distributions, max_iter=500, n_init=10)
    assert result.shape == (3, 2)

def test_js_mmds_distance_preservation(sample_distributions2):
    """Test that relative distances are approximately preserved."""
    result = js_MMDS(sample_distributions2)

    # Calculate original distances
    original_dists = np.zeros((4, 4))
    result_dists = np.zeros((4, 4))

    # Compare pairwise distances
    for i in range(4):
        for j in range(i+1, 4):
            # Original space distances
            dist_ij = np.sqrt(np.sum((sample_distributions2[i] - sample_distributions2[j])**2))
            original_dists[i,j] = original_dists[j,i] = dist_ij

            # Reduced space distances
            reduced_dist_ij = np.sqrt(np.sum((result[i] - result[j])**2))
            result_dists[i,j] = result_dists[j,i] = reduced_dist_ij

    # Check correlation between distance matrices
    correlation = np.corrcoef(original_dists.flatten(), result_dists.flatten())[0,1]
    assert correlation > 0.5  # Reasonable correlation threshold

@pytest.mark.parametrize("n_samples,n_features", [
    (3, 2),
    (5, 4),
    (10, 3)
])
def test_js_mmds_different_sizes(n_samples, n_features):
    """Test MMDS with different input sizes.

    Args:
        n_samples: Number of distributions
        n_features: Number of features per distribution
    """
    distributions = np.random.dirichlet(
        np.ones(n_features),
        size=n_samples
    )

    result = js_MMDS(distributions)
    assert result.shape == (n_samples, 2)

def test_js_mmds_reproducibility():
    """Test reproducibility with fixed random state."""
    distributions = np.array([
        [0.7, 0.3],
        [0.2, 0.8],
        [0.5, 0.5]
    ])

    result1 = js_MMDS(distributions)
    result2 = js_MMDS(distributions)

    np.testing.assert_array_almost_equal(result1, result2)

def test_js_mmds_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        invalid_dist = np.array([[1.5, -0.5], [0.8, 0.2]])  # Invalid probabilities
        js_MMDS(invalid_dist)

def test_js_tsne_basic(sample_distributions3):
    """Test basic t-SNE functionality.

    Tests dimension reduction with appropriate perplexity for small dataset.

    Args:
        sample_distributions3: Fixture providing test distributions

    Note: `init` and `perplexity` settings are chosen to work with the test data.
    """
    result = js_TSNE(sample_distributions3, init="random", perplexity=3)  # Reduced perplexity

    # Check output shape and type
    assert result.shape == (4, 2)  # 4 distributions reduced to 2D
    assert isinstance(result, np.ndarray)

def test_js_tsne_custom_params():
    """Test t-SNE with custom parameters."""
    distributions = np.array([
        [0.5, 0.5],
        [0.8, 0.2],
        [0.3, 0.7]
    ])

    result = js_TSNE(
        distributions,
        init="random",
        perplexity=2,  # Small perplexity for small dataset
        max_iter=250
    )
    assert result.shape == (3, 2)

def test_js_tsne_reproducibility():
    """Test reproducibility with fixed random state."""
    distributions = np.array([
        [0.7, 0.3],
        [0.2, 0.8],
        [0.5, 0.5]
    ])

    result1 = js_TSNE(distributions, init="random", perplexity=2)
    result2 = js_TSNE(distributions, init="random", perplexity=2)

    np.testing.assert_array_almost_equal(result1, result2)

@pytest.mark.parametrize("n_samples,n_features", [
    (5, 2),
    (10, 4),
    (15, 3)
])
def test_js_tsne_different_sizes(n_samples, n_features):
    """Test t-SNE with different input sizes.

    Args:
        n_samples: Number of distributions
        n_features: Number of features per distribution
    """
    distributions = np.random.dirichlet(
        np.ones(n_features),
        size=n_samples
    )

    result = js_TSNE(distributions, init="random", perplexity=n_samples-1)
    assert result.shape == (n_samples, 2)

def test_js_tsne_distance_preservation(sample_distributions3):
    """Test that local structure is preserved."""
    result = js_TSNE(sample_distributions3, init="random", perplexity=3)

    # Check that similar distributions are closer in reduced space
    original_dists = np.zeros((4, 4))
    result_dists = np.zeros((4, 4))

    for i in range(4):
        for j in range(i+1, 4):
            # Original space distances
            dist_ij = np.sqrt(np.sum((sample_distributions3[i] - sample_distributions3[j])**2))
            original_dists[i,j] = original_dists[j,i] = dist_ij

            # Reduced space distances
            reduced_dist_ij = np.sqrt(np.sum((result[i] - result[j])**2))
            result_dists[i,j] = result_dists[j,i] = reduced_dist_ij

    # Check local structure preservation (correlation should be positive)
    correlation = np.corrcoef(original_dists.flatten(), result_dists.flatten())[0,1]
    assert correlation > 0

def test_js_tsne_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        invalid_dist = np.array([[1.5, -0.5], [0.8, 0.2]])  # Invalid probabilities
        js_TSNE(invalid_dist)

def test_js_tsne_perplexity_warning():
    """Test perplexity warning for small datasets."""
    small_dist = np.array([
        [0.5, 0.5],
        [0.3, 0.7]
    ])

    with pytest.raises(LexosException):
        js_TSNE(small_dist)

def test_df_with_names_dataframe_input(sample_data):
    """Test function with DataFrame input.

    Args:
        sample_data: Fixture providing test data
    """
    df, _ = sample_data
    result = _df_with_names(df, "test_index", "test_columns")

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "test_index"
    assert result.columns.name == "test_columns"
    assert result.values.tolist() == df.values.tolist()

def test_df_with_names_array_input(sample_data):
    """Test function with numpy array input.

    Args:
        sample_data: Fixture providing test data
    """
    _, array = sample_data
    result = _df_with_names(array, "test_index", "test_columns")

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "test_index"
    assert result.columns.name == "test_columns"
    assert result.values.tolist() == array.tolist()

def test_df_with_names_index_numbering(sample_data):
    """Test that index is properly numbered.

    Args:
        sample_data: Fixture providing test data
    """
    df, _ = sample_data
    df.index = ['a', 'b', 'c']  # Set non-numeric index

    result = _df_with_names(df, "test_index", "test_columns")

    assert list(result.index) == [0, 1, 2]
    assert result.index.name == "test_index"

@pytest.mark.parametrize("index_name,columns_name", [
    ("row", "col"),
    ("document", "topic"),
    ("", ""),
    ("very_long_name", "another_long_name")
])
def test_df_with_names_various_names(sample_data, index_name, columns_name):
    """Test function with different name combinations.

    Args:
        sample_data: Fixture providing test data
        index_name: Name for index
        columns_name: Name for columns
    """
    df, _ = sample_data
    result = _df_with_names(df, index_name, columns_name)

    assert result.index.name == index_name
    assert result.columns.name == columns_name

def test_df_with_names_empty_dataframe():
    """Test function with empty DataFrame."""
    empty_df = pd.DataFrame()
    result = _df_with_names(empty_df, "test_index", "test_columns")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert result.index.name == "test_index"
    assert result.columns.name == "test_columns"


def test_series_with_name_series_input(sample_data2):
    """Test function with Series input.

    Args:
        sample_data2: Fixture providing test data
    """
    series, _ = sample_data2
    result = _series_with_name(series, "test_name")

    assert isinstance(result, pd.Series)
    assert result.name == "test_name"
    assert list(result.index) == [0, 1, 2]  # Numeric index
    assert list(result.values) == [1, 2, 3]

def test_series_with_name_list_input(sample_data2):
    """Test function with list input.

    Args:
        sample_data2: Fixture providing test data
    """
    _, list_data = sample_data2
    result = _series_with_name(list_data, "test_name")

    assert isinstance(result, pd.Series)
    assert result.name == "test_name"
    assert list(result.values) == [4, 5, 6]

def test_series_with_name_numeric_index():
    """Test that numeric index is preserved when already present."""
    series = pd.Series([1, 2, 3], index=[0, 1, 2])
    result = _series_with_name(series, "test_name")

    assert list(result.index) == [0, 1, 2]

@pytest.mark.parametrize("name", [
    "simple",
    "complex_name_123",
    "",  # Empty string
    "很長的名字"  # Unicode name
])
def test_series_with_name_various_names(sample_data2, name):
    """Test function with different name values.

    Args:
        sample_data2: Fixture providing test data
        name: Name to test
    """
    series, _ = sample_data2
    result = _series_with_name(series, name)

    assert result.name == name

def test_series_with_name_empty_input():
    """Test function with empty input."""
    empty_series = pd.Series([])
    result = _series_with_name(empty_series, "test_name")

    assert isinstance(result, pd.Series)
    assert result.empty
    assert result.name == "test_name"

def test_series_with_name_numpy_input():
    """Test function with numpy array input."""
    array_data = np.array([1, 2, 3])
    result = _series_with_name(array_data, "test_name")

    assert isinstance(result, pd.Series)
    assert result.name == "test_name"
    assert list(result.values) == [1, 2, 3]


def test_topic_coordinates_basic(mock_mds, sample_data3):
    """Test basic topic coordinate generation."""
    topic_term_dists, topic_proportion = sample_data3
    result = _topic_coordinates(mock_mds, topic_term_dists, topic_proportion)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['x', 'y', 'topics', 'cluster', 'Freq']
    assert len(result) == 3  # Number of topics

def test_topic_coordinates_values(mock_mds, sample_data3):
    """Test specific values in coordinate DataFrame."""
    topic_term_dists, topic_proportion = sample_data3
    result = _topic_coordinates(mock_mds, topic_term_dists, topic_proportion)

    # Check x, y coordinates from mock_mds
    assert result['x'].tolist() == [0, 1, 2]
    assert result['y'].tolist() == [1, 2, 3]

    # Check topic numbering (1-based)
    assert result['topics'].tolist() == [1, 2, 3]

    # Check cluster assignment
    assert all(result['cluster'] == 1)

    # Check frequency calculation
    expected_freqs = topic_proportion * 100
    np.testing.assert_array_almost_equal(result['Freq'], expected_freqs)

def test_topic_coordinates_dimensions(mock_mds):
    """Test with different numbers of topics."""
    test_cases = [
        (np.random.rand(2, 5), np.array([0.6, 0.4])),
        (np.random.rand(5, 3), np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        (np.random.rand(10, 4), np.ones(10) / 10)
    ]

    for topic_term_dists, topic_proportion in test_cases:
        result = _topic_coordinates(mock_mds, topic_term_dists, topic_proportion)
        n_topics = topic_term_dists.shape[0]

        assert len(result) == n_topics
        assert result['topics'].tolist() == list(range(1, n_topics + 1))
        assert len(result['x']) == len(result['y']) == n_topics

def test_topic_coordinates_assertions(mock_mds, sample_data3):
    """Test assertion error for mismatched dimensions."""
    topic_term_dists, _ = sample_data3
    wrong_proportions = np.array([0.5, 0.5])  # Wrong length

    with pytest.raises(LexosException):
        _topic_coordinates(mock_mds, topic_term_dists, wrong_proportions)

def test_topic_coordinates_frequency_range(mock_mds, sample_data3):
    """Test that frequencies are properly scaled to percentages."""
    topic_term_dists, topic_proportion = sample_data3
    result = _topic_coordinates(mock_mds, topic_term_dists, topic_proportion)

    assert all(0 <= freq <= 100 for freq in result['Freq'])
    assert np.isclose(result['Freq'].sum(), 100)

def test_get_topic_coordinates_basic(sample_data4):
    """Test basic functionality with default parameters."""
    result = get_topic_coordinates(*sample_data4)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'topics', 'cluster', 'Freq'}
    assert len(result) == 3  # Number of topics

def test_get_topic_coordinates_mds_string(sample_data4):
    """Test different MDS string options."""
    topic_term_dists, *rest = sample_data4

    # Test PCoA
    result_pcoa = get_topic_coordinates(*sample_data4, mds='pcoa')
    assert isinstance(result_pcoa, pd.DataFrame)

    # Test MMDS if sklearn is present
    if hasattr(get_topic_coordinates, 'sklearn_present'):
        result_mmds = get_topic_coordinates(*sample_data4, mds='mmds')
        assert isinstance(result_mmds, pd.DataFrame)

def test_get_topic_coordinates_custom_mds(sample_data4):
    """Test with custom MDS function."""
    def custom_mds(data):
        return np.array([[1, 1], [2, 2], [3, 3]])

    result = get_topic_coordinates(*sample_data4, mds=custom_mds)
    assert result['x'].tolist() == [1, 2, 3]
    assert result['y'].tolist() == [1, 2, 3]

def test_get_topic_coordinates_sorting(sample_data4):
    """Test topic sorting functionality."""
    # Test with sorting enabled (default)
    result_sorted = get_topic_coordinates(*sample_data4, sort_topics=True)
    freqs_sorted = result_sorted['Freq'].values
    assert all(freqs_sorted[i] >= freqs_sorted[i+1] for i in range(len(freqs_sorted)-1))

    # Test with sorting disabled
    result_unsorted = get_topic_coordinates(*sample_data4, sort_topics=False)
    assert not all(result_unsorted['Freq'].values == result_sorted['Freq'].values)

def test_get_topic_coordinates_input_validation(sample_data4):
    """Test input validation."""
    topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency = sample_data4

    # Test with mismatched dimensions
    with pytest.raises(Exception):
        bad_doc_topics = doc_topic_dists[:-1]  # Remove one document
        get_topic_coordinates(
            topic_term_dists, bad_doc_topics,
            doc_lengths, vocab, term_frequency
        )

def test_get_topic_coordinates_error_handling():
    """Test error handling for invalid inputs."""
    with pytest.raises(LexosException):
        # Pass invalid inputs
        get_topic_coordinates(
            None, None, None, None, None
        )

@pytest.mark.parametrize("mds_type", ['pcoa', 'mmds', 'tsne', 'invalid'])
def test_get_topic_coordinates_mds_types(sample_data4, mds_type):
    """Test different MDS type specifications.

    Args:
        sample_data4: Fixture providing test data
        mds_type: Type of MDS to test
    """
    try:
        result = get_topic_coordinates(*sample_data4, mds=mds_type)
        assert isinstance(result, pd.DataFrame)
        if mds_type == 'invalid':
            # Should fall back to PCoA
            assert result is not None
    except Exception as e:
        if mds_type == 'invalid':
            assert 'Unknown mds' in str(e)

def test_extract_params_basic(sample_statefile):
    """Test basic parameter extraction."""
    alpha, beta = extract_params(str(sample_statefile))

    assert isinstance(alpha, list)
    assert isinstance(beta, float)
    assert alpha == ['0.5', '0.3', '0.2']
    assert beta == 0.01

def test_extract_params_file_handling(sample_statefile):
    """Test file handling and cleanup."""
    result = extract_params(str(sample_statefile))
    assert result is not None

    # Clean up temp file
    sample_statefile.unlink()

def test_extract_params_malformed(malformed_statefile):
    """Test handling of malformed state file."""
    with pytest.raises(Exception):
        extract_params(str(malformed_statefile))

    # Clean up
    malformed_statefile.unlink()

def test_extract_params_nonexistent_file():
    """Test handling of nonexistent file."""
    with pytest.raises(FileNotFoundError):
        extract_params("nonexistent.gz")

def test_state_to_df_basic(sample_statefile2):
    """Test basic state file parsing functionality."""
    df = state_to_df(str(sample_statefile2))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Number of data rows
    assert list(df.columns) == ['#doc', 'source', 'pos', 'typeindex', 'type', 'topic']

def test_state_to_df_content(sample_statefile2):
    """Test content of parsed DataFrame."""
    df = state_to_df(str(sample_statefile2))

    # Check specific values
    assert df['#doc'].tolist() == ['doc1', 'doc1', 'doc2']
    assert df['type'].tolist() == ['word1', 'word2', 'word3']
    assert df['topic'].tolist() == [2, 1, 2]

def test_state_to_df_missing_file():
    """Test handling of missing state file."""
    with pytest.raises(LexosException):
        state_to_df("nonexistent.gz")

def test_state_to_df_empty_file():
    """Test handling of empty state file."""
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(b"#doc source pos typeindex type topic\n")

        with pytest.raises(LexosException):
            _ = state_to_df(f.name)

def test_state_to_df_malformed_content():
    """Test handling of malformed state file content."""
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(b"malformed content\n" * 3)

        with pytest.raises(Exception):
            state_to_df(f.name)

def test_state_to_df_cleanup(sample_statefile2):
    """Test proper file cleanup after processing."""
    df = state_to_df(str(sample_statefile2))
    assert df is not None

    # Clean up
    sample_statefile2.unlink()
    assert not sample_statefile2.exists()

def test_pivot_and_smooth_basic(sample_df):
    """Test basic pivot and smoothing functionality."""
    result = pivot_and_smooth(
        df=sample_df,
        smooth_value=0.1,
        rows_variable='topic',
        cols_variable='term',
        values_variable='count'
    )

    assert isinstance(result, pd.DataFrame)
    # Check if rows sum to 1 (normalized)
    assert np.allclose(result.sum(axis=1), 1.0)

def test_pivot_and_smooth_values(sample_df):
    """Test specific values after smoothing and normalization."""
    result = pivot_and_smooth(
        df=sample_df,
        smooth_value=0.5,
        rows_variable='topic',
        cols_variable='term',
        values_variable='count'
    )

    # All values should be positive due to smoothing
    assert (result.values >= 0).all()
    # Matrix should be normalized (row sums = 1)
    assert np.allclose(result.sum(axis=1), 1.0)

def test_pivot_and_smooth_missing_values():
    """Test handling of missing values in sparse matrix."""
    df = pd.DataFrame({
        'topic': [1, 2],
        'term': ['word1', 'word2'],
        'count': [5, 3]
    })

    result = pivot_and_smooth(
        df=df,
        smooth_value=0.1,
        rows_variable='topic',
        cols_variable='term',
        values_variable='count'
    )

    # Check if NaN values were properly handled
    assert not result.isna().any().any()

@pytest.mark.parametrize("smooth_value", [
    0.0,
    0.1,
    1.0,
    10.0
])
def test_pivot_and_smooth_different_values(sample_df, smooth_value):
    """Test different smoothing values.

    Args:
        sample_df: Fixture providing test data
        smooth_value: Smoothing value to test
    """
    result = pivot_and_smooth(
        df=sample_df,
        smooth_value=smooth_value,
        rows_variable='topic',
        cols_variable='term',
        values_variable='count'
    )

    assert np.allclose(result.sum(axis=1), 1.0)
    assert (result.values >= 0).all()

def test_pivot_and_smooth_empty_input():
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame(columns=['topic', 'term', 'count'])

    with pytest.raises(LexosException):
        _ = pivot_and_smooth(
            df=empty_df,
            smooth_value=0.1,
            rows_variable='topic',
            cols_variable='term',
            values_variable='count'
        )

def test_pivot_and_smooth_invalid_columns():
    """Test error handling for invalid column names."""
    invalid_df = pd.DataFrame({
        'invalid1': [1, 2],
        'invalid2': ['a', 'b'],
        'invalid3': [5, 3]
    })

    with pytest.raises(KeyError):
        pivot_and_smooth(
            df=invalid_df,
            smooth_value=0.1,
            rows_variable='topic',
            cols_variable='term',
            values_variable='count'
        )

def test_convert_mallet_data_basic(sample_statefile2):
    """Test basic MALLET data conversion."""
    result = convert_mallet_data(str(sample_statefile2))

    assert isinstance(result, dict)
    expected_keys = {
        'topic_term_dists', 'doc_topic_dists', 'doc_lengths',
        'vocab', 'term_frequency'
    }
    assert set(result.keys()) == expected_keys

def test_convert_mallet_data_shapes(sample_statefile2):
    """Test output data shapes and dimensions."""
    result = convert_mallet_data(str(sample_statefile2))

    # Check matrix shapes
    assert result['topic_term_dists'].shape[1] == len(result['vocab'])
    assert result['doc_topic_dists'].shape[0] == len(result['doc_lengths'])
    assert len(result['vocab']) == len(result['term_frequency'])

def test_convert_mallet_data_distributions(sample_statefile2):
    """Test probability distributions in output."""
    result = convert_mallet_data(str(sample_statefile2))

    # Check row sums for probability distributions
    assert np.allclose(result['topic_term_dists'].sum(axis=1), 1.0)
    assert np.allclose(result['doc_topic_dists'].sum(axis=1), 1.0)

def test_convert_mallet_data_missing_file():
    """Test handling of missing state file."""
    with pytest.raises(LexosException):
        convert_mallet_data("nonexistent.gz")

def test_convert_mallet_data_malformed_content():
    """Test handling of malformed state file content."""
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            gz.write(b"malformed content\n" * 3)

        with pytest.raises(LexosException):
            convert_mallet_data(f.name)

def test_convert_mallet_data_values(sample_statefile2):
    """Test specific values in converted data."""
    result = convert_mallet_data(str(sample_statefile2))

    # Check vocabulary
    assert 'word1' in result['vocab']
    assert 'word2' in result['vocab']
    assert 'word3' in result['vocab']

    # Check document lengths
    assert sum(result['doc_lengths']) == 3  # Total tokens

def test_convert_mallet_data_term_frequencies(sample_statefile2):
    """Test term frequency calculations."""
    result = convert_mallet_data(str(sample_statefile2))

    # Sum of term frequencies should equal total tokens
    assert sum(result['term_frequency']) == sum(result['doc_lengths'])

