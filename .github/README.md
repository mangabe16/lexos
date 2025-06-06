# README

The file `ci_workflow.yml` is a template for a GitHub Actions workflow that runs on pull requests to the main branch. It sets up a Python environment, installs `uv`, installs the project with all dependencies, lints with Ruff, and then runs tests using pytest.

When all tests pass, we can put a badge in the project README file to show the status of the tests. The badge will look like this: ![Tests](https://github.com/scottkleinman/uv_lexos/actions/workflows/ci_workflow.yml/badge.svg).
