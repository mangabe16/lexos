# Rolling Windows Module

## Overview

The Rolling Windows module is a powerful text analysis tool that tracks how patterns change throughout documents. It creates a "moving spotlight" that slides through your text, counting specific words or patterns in each section to reveal temporal dynamics and narrative structures.

### What Does Rolling Windows Do?

Imagine reading a novel and tracking how often "love" appears versus "conflict" as the story progresses. Rolling Windows automates this process, creating visual timelines that show:

- **Thematic Evolution**: How central themes rise and fall throughout a text
- **Character Arcs**: When characters appear most prominently in narratives  
- **Stylistic Patterns**: Changes in writing style, punctuation usage, or linguistic features
- **Emotional Journeys**: Tracking sentiment-related words to map emotional trajectories
- **Structural Analysis**: Identifying patterns in document organization

The module generates frequency plots that help researchers, students, and analysts understand how language patterns evolve within texts.

## Key Components

### 1. The `Windows` Class

The foundation of the module, this class segments your text into overlapping "windows" - chunks of text that slide through your document one step at a time.

```python
from lexos.rolling_windows import Windows

windows = Windows()
rw = windows(input=doc, n=100, window_type="tokens", output="strings")
```

**Key Parameters:**
- `n`: Size of each window (e.g., 100 tokens)
- `window_type`: How to measure windows ("tokens", "characters", or "lines")
- `output`: Format of results ("strings" or "tokens")

> **Developer Note:**  
> The `Windows` class yields a generator for efficient memory usage. Convert it to a list only if needed.  
> 
> **Accepted input types:**  
> - `str` (plain text)  
> - spaCy `Doc` objects  
> - `list` of strings or tokens  

### 2. Calculator Classes

Calculators analyze the patterns within each window. The module includes several calculator types:

#### `Averages` Calculator
Calculates the average frequency of patterns across windows, normalizing for window size.

```python
from lexos.rolling_windows.calculators import Averages

averages = Averages(
    patterns=["love", "death"],
    windows=rw,
    mode="exact"
)
```

#### `Counts` Calculator  
Provides raw occurrence counts without normalization.

#### `Ratios` Calculator
Computes the ratio between two patterns (e.g., positive vs. negative words).

**Search Modes:**  
A search mode determines how the Rolling Windows module matches your specified patterns within each window of text. It controls whether the search looks for exact string matches, uses regular expressions for flexible pattern matching, applies linguistic rules with spaCy, or detects multi-word phrases. Choosing the right search mode allows you to tailor the analysis to your research question—whether you need simple keyword counts, advanced pattern recognition, or linguistic feature extraction.
- `"exact"`: Precise string matching
- `"regex"`: Regular expression patterns (e.g., `"^love.*"` for words starting with "love")
- `"spacy_rule"`: Advanced linguistic pattern matching
- `"multi_token"`: Phrase detection

> **IMPORTANT TIP:**  
> After creating a calculator, use the `.to_df()` method to convert results into a pandas DataFrame for further analysis or plotting.

### 3. Plotter Classes

Visualize your results with two plotting options:

#### `SimplePlotter`
Generates high-quality static plots suitable for publications using Matplotlib.

```python
from lexos.rolling_windows.plotters import SimplePlotter

plotter = SimplePlotter(title="Word Frequencies Over Time")
plotter.plot(averages.to_df())
```

#### `PlotlyPlotter`
Generates interactive web-based visualizations with hover tooltips and zoom capabilities.

```python
from lexos.rolling_windows.plotters import PlotlyPlotter

interactive_plotter = PlotlyPlotter()
interactive_plotter.plot(averages.to_df(), show_plot=True)
```

## How It Works

1. **Text Preparation**: Load your text and optionally preprocess it (lowercase, remove punctuation)
2. **Window Creation**: Segment text into overlapping windows of specified size
3. **Pattern Analysis**: Count occurrences of your search terms in each window
4. **Calculation**: Compute averages, counts, or ratios based on your needs
5. **Visualization**: Generate plots showing pattern frequencies across the document

> **TIP:**  
> For a detailed, step-by-step walkthrough—including code examples and explanations—see the accompanying tutorial Jupyter notebook.


## Practical Applications

### Literary Analysis
Track character mentions, themes, or stylistic elements throughout novels:
```python
# Track protagonist vs antagonist presence
patterns = ["elizabeth", "wickham"]
```

### Historical Documents
Analyze changing terminology or concepts over time:
```python
# Track evolution of political terms
patterns = ["liberty", "freedom", "rights"]
```

### Linguistic Research
Study language features and their distribution:
```python
# Analyze punctuation patterns
patterns = ["!", "?", "..."]
```

### Content Analysis
Examine emotional or thematic content in texts:
```python
# Sentiment tracking
patterns = ["happy", "sad", "angry", "peaceful"]
```

## Dependencies

The Rolling Windows module requires:
- `spacy` (with language model, e.g., `en_core_web_sm`)
- `pandas` 
- `numpy`
- `matplotlib` (for static plots)
- `plotly` (for interactive plots)
- `pydantic` (for data validation)

> **Developer Note:**  
> Install the spaCy language model separately:  
> 
> ```bash
> python -m spacy download en_core_web_sm
> ```

## Quick Start Example

```python
import spacy
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators import Averages
from lexos.rolling_windows.plotters import SimplePlotter

# Load text and create spaCy doc
nlp = spacy.load("en_core_web_sm")
text = "Your text here..."
doc = nlp(text)

# Create 100-token windows
windows = Windows()
rw = windows(input=doc, n=100, window_type="tokens")

# Calculate pattern frequencies
calc = Averages(patterns=["love", "war"], windows=rw)

# Generate visualization
plotter = SimplePlotter(title="Love vs War")
plotter.plot(calc.to_df())
```

## Advanced Features

### Milestone Markers
Add vertical lines to mark important sections (chapters, scenes, etc.):

```python
milestones = {"Chapter 1": 0, "Chapter 2": 500, "Chapter 3": 1000}
plotter = SimplePlotter(show_milestones=True, milestone_labels=milestones)
```

### Custom Window Alignment
Control how character-based windows snap to token boundaries:

```python
windows = Windows()
rw = windows(input=doc, n=500, window_type="characters", 
             alignment_mode="contract")  # or "expand", "strict"
```

### Export Options
Save results in various formats:
- PNG/SVG (static plots)
- HTML (interactive plots)
- CSV (raw data)

## Testing

The module includes comprehensive test coverage (100% as of 6/2/2025):

```bash
# Run all tests
uv run pytest tests/rolling_windows/

# Generate coverage report
uv run pytest --cov=src/lexos/rolling_windows tests/rolling_windows/
```
