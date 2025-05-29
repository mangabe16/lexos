# README for Rolling Windows module

## Overview
This module provides rolling windows analysis for tracking how patterns change throughout texts. It divides documents into overlapping segments and analyzes each segment for specified patterns, revealing trends and narrative arcs in literary works, speeches, or any long-form text.

## Features
- Create character-based or token-based windows from text
- Support for multiple input formats (strings, spaCy docs, token lists)
- Pattern counting and frequency analysis with multiple search modes
- Integration with spaCy for advanced linguistic analysis
- Static and interactive visualization with milestone support
- Comprehensive calculator suite for statistical analysis

## Dependencies
The Rolling Windows module is part of lexos. Follow the lexos README to install the virtual environment and all necessary libraries.
- numpy
- pandas 
- spacy
- matplotlib
- plotly
- scipy
- pydantic
- lexos

## Quick Start

```python
import spacy
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators import Averages
from lexos.rolling_windows.plotters import SimplePlotter

# Load text and create windows
nlp = spacy.load("en_core_web_sm")
doc = nlp("Your text here...")
windows = Windows()

# Analyze emotional patterns
analysis_windows = windows(input=doc, n=50, window_type="tokens", output="strings")
averages = Averages(patterns=["love", "fear", "joy"], windows=analysis_windows)

# Visualize results
plotter = SimplePlotter(title="Emotional Analysis")
plotter(df=averages.to_df())
```

## Core Components

### Windows
Creates overlapping text segments for analysis:
- **Character windows**: Fixed character counts
- **Token windows**: Fixed word/token counts  
- **Span windows**: Custom spaCy spans

### Calculators
Analyze patterns within windows:
- **Counts**: Raw pattern occurrences
- **Averages**: Average frequency per window

### Plotters
Visualize rolling windows data:
- **SimplePlotter**: Static matplotlib plots
- **PlotlyPlotter**: Interactive web-based plots

## Search Modes
- **exact**: Precise string matching
- **regex**: Regular expression patterns
- **spacy_rule**: Linguistic rule matching
- **multi_token**: Multi-word phrase matching

## Running Tests
This repository contains a comprehensive test suite with 100% coverage:

```bash
# Run all tests in the rolling windows module
uv run pytest tests/rolling_windows/

# Run with coverage report
uv run pytest --cov=src/lexos/rolling_windows --cov-report=html tests/rolling_windows/
```