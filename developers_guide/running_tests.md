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

## Checking Coverage

To run tests with coverage, you can use the following command:

```bash
uv run pytest --cov src src --cov-report=term-missing
```

This displays a coverage report in your terminal.

> [!IMPORTANT]
> In the command above, `src` refers to the source directory of the test file. For some reason, the pytest-cov plugin generates a report for _all_ Lexos modules unless you specify the path to the test file's folder _twice_ (e.g. `tests/scrubber`). Doing this will generate a coverage report for all test files in the folder, which is at least better. So far, attempts to specify the path to a single file has always generated a report containing all test files in the folder. This is an ongoing issue with pytest-cov.

To generate an HTML coverage report, something like use:

```bash
uv run pytest --cov tests/dtm tests/dtm --cov-report=html
```

After running, open `htmlcov/index.html` in your browser to inspect coverage. As with the terminal report, you can adjust the `--cov` option to target specific modules or directories.
