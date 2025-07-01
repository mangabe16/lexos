# Rolling Windows

## Overview

!!! important
    This page is currently under construction.

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

!!! note "Developer's Note"
    The `Windows` class yields a generator for efficient memory usage. Convert it to a list only if needed.

    **Accepted input types:**

    - `str` (plain text)
    - spaCy `Doc` objects
    - `list` of strings or tokens

    **Important: Windows Are Consumed!**
    After using a windows object once (like converting to a list), you need to create fresh windows for each calculator. This is why you'll need to create new windows for each analysis.

### 2. Calculator Classes

Calculators analyze the patterns within each window. The module includes several calculator types:

## Choosing Your Analysis Type

| Calculator | Use When | What it measures | Output |
|------------|----------|------------------|--------|
| `Counts` | You need raw occurrence numbers | Total number of matches per window | Raw frequencies |
| `Averages` | Comparing texts of different lengths or standardizing | Matches per unit (normalized) | Frequency rates |
| `Ratios` | Comparing balance between exactly 2 patterns | Relative proportion | Values from 0.0 to 1.0 |

### `Counts` Calculator

Provides raw occurrence counts without normalization.

```python
from lexos.rolling_windows.calculators import Counts

counter = Counts()
counter(
    patterns=["love", "death"],
    windows=rw,
    mode="exact",
    case_sensitive=False
)
```

### `Averages` Calculator

Calculates the average frequency of patterns across windows, normalizing for window size.

```python
from lexos.rolling_windows.calculators import Averages

averages = Averages()
averages(
    patterns=["love", "death"],
    windows=rw,
    mode="exact",
    case_sensitive=False
)
```

**When to use Averages instead of Counts:**

- Comparing texts of different lengths
- Comparing different window sizes
- Creating standardized measurements
- Academic/scientific analysis

### `Ratios` Calculator

Computes the ratio between two patterns (e.g., positive vs. negative words).

```python
from lexos.rolling_windows.calculators import Ratios

ratio_calc = Ratios()
ratio_calc(
    patterns=["positive", "negative"],  # Exactly 2 patterns required
    windows=rw,
    mode="exact",
    case_sensitive=False
)
```

**Understanding Ratios:**

- **0.0** = Only the second pattern appears
- **0.5** = Both patterns appear equally
- **1.0** = Only the first pattern appears
- Values closer to 0 favor the second pattern
- Values closer to 1 favor the first pattern

## Search Modes

A search mode determines how the Rolling Windows module matches your specified patterns within each window of text. It controls whether the search looks for exact string matches, uses regular expressions for flexible pattern matching, applies linguistic rules with spaCy, or detects multi-word phrases.

### Available Search Modes

- `"exact"`: Precise string matching
- `"regex"`: Regular expression patterns
- `"spacy_rule"`: Advanced linguistic pattern matching
- `"multi_token"`: Phrase detection

### Pattern Matching Examples

| Pattern Type | Example | Mode | Use Case |
|--------------|---------|------|----------|
| Exact match | `["love", "hate"]` | `"exact"` | Simple word counting |
| Word starts with | `[r"\bsh\w*"]` | `"regex"` | Words starting with "sh" |
| Word ends with | `[r".*ing$"]` | `"regex"` | Words ending in "-ing" |
| Numbers | `[r"\d+"]` | `"regex"` | Numeric content |
| Capitalized words | `[r"\b[A-Z]\w*"]` | `"regex"` | Proper nouns, sentence starts |
| Multi-word phrases | `["sherlock holmes"]` | `"exact"` | Exact phrase detection |
| All proper nouns | `[[{"POS": "PROPN"}]]` | `"spacy_rule"` | Linguistic analysis |
| All verbs | `[[{"POS": "VERB"}]]` | `"spacy_rule"` | Grammatical patterns |

!!! important "Important Tip"
    After creating a calculator, use the `.to_df()` method to convert results into a pandas DataFrame for further analysis or plotting.

    **SpaCy Requirements:**
    - spaCy patterns require `window_type="tokens"` and `output="tokens"`
    - Regex patterns use raw strings (e.g., `r"\bsh\w*"`)
    - Use `case_sensitive=False` for case-insensitive matching

### 3. Plotter Classes

Visualize your results with two plotting options:

#### `SimplePlotter`

Generates high-quality static plots suitable for publications using Matplotlib.

**Best for:** Reports, publications, presentations

```python
from lexos.rolling_windows.plotters import SimplePlotter

plotter = SimplePlotter(title="Word Frequencies Over Time")
plotter.plot(averages.to_df())
```

#### `PlotlyPlotter`

Generates interactive web-based visualizations with hover tooltips and zoom capabilities.

**Best for:** Exploration, web presentation, detailed analysis with hover information

```python
from lexos.rolling_windows.plotters import PlotlyPlotter

interactive_plotter = PlotlyPlotter()
interactive_plotter.plot(averages.to_df(), show_plot=True)
```

**Interactive Features:**

- **Hover** over points to see exact values
- **Zoom** in/out with mouse wheel or zoom controls
- **Pan** by clicking and dragging
- **Toggle** lines on/off by clicking legend items
- **Download** plot as PNG using the camera icon

## Window Size Guidelines

| Text Type | Window Type | Suggested Size | Reasoning |
|-----------|-------------|----------------|-----------|
| Short story | characters | 200-500 | Captures local patterns |
| Novel/Book | characters | 500-2000 | Balances detail and trends |
| Short text | tokens | 20-50 | Enough words for patterns |
| Novel/Book | tokens | 50-200 | Captures thematic shifts |
| Poetry | tokens (lines) | 5-20 | Respects verse structure |

## How It Works

1. **Text Preparation**: Load your text and optionally preprocess it (lowercase, remove punctuation)
2. **Window Creation**: Segment text into overlapping windows of specified size
3. **Pattern Analysis**: Count occurrences of your search terms in each window
4. **Calculation**: Compute averages, counts, or ratios based on your needs
5. **Visualization**: Generate plots showing pattern frequencies across the document

!!! note "Tip"
    For a detailed, step-by-step walkthrough—including code examples and explanations—see the accompanying tutorial Jupyter notebook.

## Quick Start Example

```python
import spacy
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators import Averages
from lexos.rolling_windows.plotters import SimplePlotter

# Load text and create spaCy doc
nlp = spacy.load("en_core_web_sm")
with open("your_text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Basic text cleaning (optional)
text = raw_text.lower()
doc = nlp(text)

# Create 100-token windows
windows = Windows()
rw = windows(input=doc, n=100, window_type="tokens", output="strings")

# Calculate pattern frequencies
calc = Averages()
calc(patterns=["love", "war"], windows=rw, mode="exact", case_sensitive=False)

# Generate visualization
plotter = SimplePlotter(title="Love vs War")
plotter.plot(calc.to_df())

# Save results
calc.to_df().to_csv("analysis_results.csv")
plotter.save("analysis_plot.png")
```

## Complete Workflow Example

```python
# 1. Load and prepare text
with open("A_Scandal_in_Bohemia.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# 2. Create windows (1000 characters each)
windows = Windows()
analysis_windows = windows(input=text, n=1000, window_type="characters", output="strings")

# 3. Calculate average frequencies of 'a' and 'e'
calculator = Averages()
calculator(patterns=["a", "e"], windows=analysis_windows, mode="exact", case_sensitive=False)
results = calculator.to_df()

# 4. Create visualization
plotter = SimplePlotter(title="Letter Frequencies: 'a' vs 'e'")
plotter(df=results)

# 5. Save everything
plotter.save("analysis.png")
results.to_csv("analysis_data.csv")
```

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

## Troubleshooting Common Issues

**Problem:** "Windows are consumed" error
**Solution:** Create fresh windows for each calculator call

**Problem:** spaCy patterns don't work
**Solution:** Ensure `window_type="tokens"` and `output="tokens"` for spaCy rules

**Problem:** No matches found
**Solution:** Check case sensitivity, try `case_sensitive=False`

**Problem:** Regex not working
**Solution:** Use raw strings (`r"pattern"`) and escape special characters

**Problem:** Memory issues with large texts
**Solution:** Reduce window size or limit text length (e.g., `text[:1000000]`)

## Dependencies

The Rolling Windows module requires:

- `spacy` (with language model, e.g., `en_core_web_sm`)
- `pandas`
- `numpy`
- `matplotlib` (for static plots)
- `plotly` (for interactive plots)
- `pydantic` (for data validation)

!!! note "Developer's Note"
    Install the spaCy language model separately:

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Testing

The module includes comprehensive test coverage (100% as of 6/2/2025):

```bash
# Run all tests
uv run pytest tests/rolling_windows/

# Generate coverage report
uv run pytest --cov=src/lexos/rolling_windows tests/rolling_windows/
```
