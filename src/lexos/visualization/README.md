
---

# README for Lexos Visualization Suite

## Overview

The Lexos Visualization Suite provides a set of modular tools for analyzing and visualizing textual data. These tools are built to support humanities scholars, linguists, and data scientists in exploring patterns across documents using both traditional and interactive visual methods.

This suite integrates with the broader Lexos ecosystem and is compatible with data structures such as token lists, frequency dictionaries, `spaCy` Doc objects, Pandas DataFrames, and Lexos Document-Term Matrices (DTMs). Each visualization function is accompanied by interactive Jupyter notebooks for demonstration and experimentation.

---

## Key Features

* **Word Cloud Generation**
  Create highly customizable word clouds from raw strings, tokenized text, frequency dictionaries, or tabular data. Supports optional masking (e.g., circular clouds), file export, and display suppression.

* **Multicloud Grids**
  Visualize and compare multiple documents or slices of a long text using a grid layout of word clouds. Labeling and layout options allow for intuitive comparisons between sections or sources.

* **Interactive Plotly Word Clouds**
  Build zoomable, browser-rendered word clouds with fine-tuned layout control and color palettes using Plotly.

* **Bubble Charts**
  Explore word frequency and token spacing visually through bubble charts that map size and position to textual metrics.

* **Preprocessing Utilities**
  Convert between different data types (e.g., `list[str]`, `spaCy.Doc`, DataFrame, DTM) to ensure compatibility with the visualizations. Includes intelligent type detection and validation.

---

## Contents

| Notebook                 | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `cloud.ipynb`            | Introduces `wordcloud()` and `multicloud()`, with examples from strings, Docs, DTMs |
| `plotly_wordcloud.ipynb` | Demonstrates browser-based word cloud generation using Plotly                       |
| `bubbleChart.ipynb`      | Shows how to construct bubble charts from token position and frequency data         |
| `processors.ipynb`       | Details how to convert between raw text, tokens, and tabular formats                |

---

## Dependencies

These notebooks assume that the Lexos development environment is properly configured. The following Python libraries are required:

* `lexos` (internal package)
* `spaCy` (`en_core_web_sm` model)
* `matplotlib`
* `wordcloud`
* `plotly`
* `pandas`
* `numpy`
* `pydantic`

For installation instructions, refer to the root-level Lexos `README.md`.

---

## Sample Usage

```python
from lexos.visualization.cloud import wordcloud

# Generate and display a basic word cloud with a circular mask
wordcloud("This is an example string for a word cloud.", round=100)
```

For multicloud comparisons:

```python
from lexos.visualization.cloud import multicloud

texts = [["alpha", "beta", "beta"], ["gamma", "delta", "delta", "delta"]]
multicloud(texts, labels=["Text A", "Text B"], ncols=2)
```

---

## Running Tests

This repository includes a set of unit tests designed to validate the functionality and robustness of the Lexos Visualization modules. The test suite ensures compatibility across input types, proper handling of edge cases, and consistent visual outputs.

To run the tests associated with the visualization components, execute the following command from the project root within the active virtual environment:

```bash
# Run all tests for the visualization module
uv run pytest tests/visualization/
```

This will execute tests for `cloud.py`, `bubbleviz.py`, `plotly_wordcloud.py`, and any supporting processor functions. Code coverage is actively monitored and maintained across these modules.

---
