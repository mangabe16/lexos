# Running Tests and Checking Coverage

We use **pytest** for testing and **coverage.py** to measure test coverage.

To run all tests, start from the project root and run:

```bash
uv run pytest
```

Note that this can take a while, as it runs all tests in the `tests` directory.

To run tests for a specific module (e.g. `dtm`), use the following command:

```bash
uv run pytest tests/dtm
```

To run tests for a specific file, use:

```bash
uv run pytest tests/dtm/test_dtm.py
```

To run a specific function, use:

```bash
uv run pytest tests/dtm/test_dtm.py::test_function_name
```

To run tests with coverage, you can use the following command:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

This displays a coverage report in your terminal.

!!! Important
    The `--cov=src` option specifies the source directory to measure coverage for. You can adjust this to target specific modules or directories as needed.

To generate an HTML coverage report, use:

```bash
uv run pytest --cov=src/lexos/dtm --cov-report=html tests/dtm
```

After running, open `htmlcov/index.html` in your browser to inspect coverage. As with the terminal report, you can adjust the `--cov` option to target specific modules or directories.
