# Rolling Windows

The `rolling_windows` module provides classes for calculating and visualizing statistical frequencies of terms over sliding windows.

The main module is [rolling_windows](rolling_windows.py).

The `rolling_windows` module has three built-in calculator classes, [counts](calculators/counts.py), [averages](calculators/averages.py), and [ratios](calculators/ratios.py). Custom calculators should inherit from [base_plotter](calculators/base_calculator.py).

The `rolling_windows` module has two built-in plotter classes, [simple_plotter](plotters/simple_plotter.py) and [plotly_plotter](plotters/plotly_plotter.py). Custom plotters should inherit from [base_plotter](plotters/base_plotter.py).
