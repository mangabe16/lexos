# README

The file `ci_workflow.yml` is a template for a GitHub Actions workflow that runs on pull requests to the main branch. It sets up a Python environment, installs `uv`, installs the project with all dependencies, and then runs tests using pytest.

The workflow is currently not operational. To activate it, the `ci_workflow.yml` file needs to be placed in the `.github/workflows` directory of the repository. It should work once the change is pushed to the remote repo.

However, it should be checked to see if the workflow is correct. Also, the caching and linting procedures are not yet enabled; see the comments inside the workflow file for more details.

When all tests pass, we can put a badge in the project README file to show the status of the tests. The badge will look like this: ![Tests](https://github.com/scottkleinman/uv_lexos/actions/workflows/ci_workflow.yml/badge.svg).
