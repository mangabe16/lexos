# Lexos Cluster Module: Document Clustering Visualizations

The `lexos.cluster` module provides tools for visualizing hierarchical clustering of text documents. It enables developers to analyze and represent linguistic relationships within a corpus, primarily utilizing a Document-Term Matrix (DTM) as input. This module offers static (Matplotlib/Seaborn) and interactive (Plotly) visualization options, and includes methods for assessing the robustness of cluster results.

---

## Core Classes

### `ClusterMap` (`clustermap.py`)

This class generates **static clustermaps** using `seaborn` and `matplotlib`. It combines a heatmap of document-term frequencies with dendrograms on both axes, allowing for the visualization of clustering relationships between documents and terms simultaneously.

* **Backend:** `seaborn.clustermap`.
* **Metrics & Methods:** Supports standard `metric`s (e.g., "euclidean", "cosine") and linkage `method`s (e.g., "average", "ward") for hierarchical clustering.
* **Scaling:** Offers `z_score` and `standard_scale` for data normalization before clustering.
* **Customization:** Extensive options for `cmap` (color map), `figsize`, `linewidths`, and control over dendrogram visibility (`hide_upper`, `hide_side`).
* **Output:** Returns a `matplotlib.figure.Figure` object; supports saving to common image formats (`.png`, `.pdf`, etc.) via `save()`.

### `PlotlyClustermap` (`plotly_clustermap.py`)

This class generates **interactive Plotly Clustermaps**, providing a web-based visualization that allows for zooming, panning, and hovering to inspect data points. It is designed for rich, exploratory data analysis directly within web environments or Jupyter Notebooks.

* **Backend:** `plotly.figure_factory.create_dendrogram` (internal use for dendrograms) and custom Plotly figure construction.
* **Metrics & Methods:** Utilizes `scipy.spatial.distance.pdist` for `metric`s and `scipy.cluster.hierarchy.linkage` for `method`s.
* **Interactivity:** Generated figures are interactive HTML by default.
* **Customization:** Configurable `colorscale` for the heatmap, `width`/`height`, and `title`.
* **Output:** Returns a `plotly.graph_objects.Figure` object; supports saving as interactive HTML (`.html`) or static images (`.png`, `.svg`, etc.) using `write_html()` and `write_image()`.

### `PlotlyDendrogram` (`plotly_dendrogram.py`)

This class generates **interactive Plotly Dendrograms**, focusing specifically on the hierarchical tree structure of document clusters. It's ideal for a clear view of how documents group based on similarity.

* **Backend:** `plotly.figure_factory.create_dendrogram`.
* **Metrics & Methods:** Same underlying `scipy` functions for `metric`s and `method`s as `PlotlyClustermap`.
* **Orientation:** Supports `orientation` (`bottom`, `left`, `top`, `right`) for dendrogram layout.
* **Truncation:** Features `truncate_mode` (e.g., "lastp", "level") to simplify complex dendrograms by showing only key mergers.
* **Cluster Coloring:** `color_threshold` allows visual differentiation of clusters based on their merge distance.
* **Output:** Returns a `plotly.graph_objects.Figure` object; supports saving as interactive HTML or static images via `write_html()` and `write_image()`.

### `BCT` (Bootstrap Consensus Tree) (`bootstrap_consensus.py`)

This class generates a **bootstrap consensus tree**, a static visualization that shows the robustness of the clusters found. It resamples the data multiple times, performs hierarchical clustering on each resampled dataset, and then creates a consensus tree showing the clusters that appear most consistently.

* **Backend:** `Bio.Phylo` for tree manipulation and `matplotlib` for plotting.
* **Bootstrapping:** Implements a bootstrapping process to assess the stability of clusters. Key parameters include `iterations` (number of resampling iterations) and `cutoff` (the consensus threshold for displaying clusters).
* **Metrics & Methods:** Uses `scipy.cluster.hierarchy.linkage` for clustering, allowing for different `distance_metric` and `linkage_method` choices.
* **Output:** Returns a `matplotlib.figure.Figure` object.  Branch lengths represent the bootstrap support values.
* **Note:** This class uses `biopython` for working with phylogenetic trees.

### `Dendrogram` (`dendrogram.py`)

This class generates **static dendrograms** using `scipy.cluster.hierarchy` and `matplotlib`. It provides a traditional dendrogram visualization of hierarchical clustering.

* **Backend:** `scipy.cluster.hierarchy.dendrogram` and `matplotlib`.
* **Metrics & Methods:** Uses `scipy.spatial.distance.pdist` for distance calculation and `scipy.cluster.hierarchy.linkage` for the clustering method. Offers various `metric` and `method` options.
* **Customization:** Allows for control over the dendrogram's appearance, including `orientation`, leaf label rotation (`leaf_rotation`), and font size (`leaf_font_size`).
* **Output:** Returns a `matplotlib.figure.Figure` object.

---

## Prerequisites and Installation

To effectively use and contribute to this module, ensure you have the following installed:

* **Python 3.8+**
* **Core Libraries:**

    ```bash
    pip install matplotlib pandas scipy pydantic numpy spacy plotly seaborn biopython
    ```
* **spaCy Language Model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```
* **Static Image Export for Plotly (Optional):** For `PlotlyClustermap.write_image()` and `PlotlyDendrogram.write_image()`, you'll need `kaleido`:

    ```bash
    pip install kaleido
    ```
    If you are developing the `lexos` package, it's recommended to install it in editable mode from the project root:

    ```bash
    pip install -e .
    ```

---

## Usage

All classes operate on a Document-Term Matrix (DTM) derived from text data. Refer to the detailed Jupyter Notebook tutorials for comprehensive examples and parameter explanations:

* `plotly_clustermap_tutorial.ipynb`
* `plotly_dendrogram_tutorial.ipynb`

Here's a minimal example for `PlotlyClustermap`:

```python
import spacy
from lexos.dtm import DTM
from lexos.cluster.plotly_clustermap import PlotlyClustermap # Assuming this is the Plotly one

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Example documents and labels
docs = ["Document one content.", "Document two content.", "Another document."]
labels = ["Doc1", "Doc2", "Doc3"]

# Create DTM
dtm_instance = DTM()
dtm_instance(docs=[nlp(doc) for doc in docs], labels=labels)

# Generate and show Clustermap
clustermap = PlotlyClustermap()
clustermap(dtm=dtm_instance, labels=labels, showfig=True, title="My Plotly Clustermap")
# To save as HTML: clustermap.fig.write_html("my_clustermap.html")

```

Here's a minimal example for `BCT`:

```python
from lexos.dtm import DTM
from lexos.cluster.bootstrap_consensus import BCT
import spacy

nlp = spacy.load("en_core_web_sm")

docs = ["Document one content.", "Document two content.", "Another document."]
labels = ["Doc1", "Doc2", "Doc3"]

dtm_instance = DTM()
dtm_instance(docs=[nlp(doc) for doc in docs], labels=labels)

bct = BCT(doc_term_matrix=dtm_instance, iterations=50, showfig=True) # Reduced iterations for speed
# bct.save("bootstrap_tree.png") # To save the figure
```

### Development and Testing

This module includes a robust test suite to ensure functionality and stability. All tests currently pass, and the module maintains approximately 98% code coverage. Please note that I have deleted tests for code requiring kaleido (see issue #19).

To run the test suite and generate a detailed coverage report (this requires `uv` for dependency management in the project root):

```bash
uv run pytest --cov=src/lexos/cluster --cov-report=html tests/cluster
```