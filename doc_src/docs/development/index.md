# Contribute to Lexos

Thanks for your interest in contributing to Lexos. This page will give you a quick overview of our contribution guidelines and how to proceed.

## Issues and Bug Reports

First, [search the GitHub issues](https://github.com/scottkleinman/lexos/issues) to see if the issue has already been reported. If so, please leave a comment on the existing issue.

If are just looking for help using Lexos, please post you question on the [GitHub Discussions board](https://github.com/scottkleinman/lexos/discussions).

### Submitting Issues

When opening an issue, use a **descriptive title** and include your **environment** (operating system, Python version, Lexos version). If you are fixing a bug with a pull request, please refer to your pull request when reporting the bug in your issue on GitHub.

When reporting issues, follow [GitHub Markdown conventions](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax), especially for code and code blocks.

Please consider adding a GitHub label to your post to help us categorize the issue.

## Contributing to the Code Base or Documentation

The Lexos source code and documentation currently reside in the same repository under the `lexos/src` and `lexos/doc_src` folders respectively. To work with them, you will need to fork and clone the repository and work with it in a local development environment. When you are ready, you can submit the code as a pull request to the main Lexos repository.

### Fork and Clone the Repository

Start by forking the project on GitHub to your own account. Then clone your fork locally. On the command line, run

```bash
git clone https://github.com/your-username/lexos.git
cd lexos
```

You can also clone the repository with a preferred `git` client or code editor.

### Set Up Your Development Environment

To make changes to the Lexos source code or documentation, you will need to have a development environment consisting of a Python 3.12+, (preferably) [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies and your virtual environment, and [git](https://git-scm.com) installed.

📖 **For more detailed information, see [Setting Up Your Development Environment](setup.md).**

### Fixing Bugs

When fixing a bug, first create an [issue](https://github.com/scottkleinman/lexos/issues) if one does not already exist.

Next, add a test to the relevant file in the [`lexos/tests`](lexos/tests) folder. Then add a [pytestmark](https://docs.pytest.org/en/6.2.x/example/markers.html#working-with-custom-markers), `@pytest.mark.issue(NUMBER)`, to reference the issue number.

```python
# Assume you're fixing Issue #1234
@pytest.mark.issue(1234)
def test_issue1234():
    ...
```

Test for the bug you're fixing, and make sure the test fails. Next, add and commit your test file. Finally, fix the bug, make sure your test passes and reference the issue number in your pull request description.

📖 **For more information on how to add tests, see the [Lexos Tests](tests.md) page.**

### Contributing Features

When contributing new features, it is important to understand the architecture of the Lexos Library. New features should be added to existing modules or should be created in new modules, as appropriate for the type of feature. All new features must be accompanied by tests and documentation as outlined on the [Lexos Tests](lexos/tests.md) page.

📖 **For more information on how to create new features, see the [Creating Features](creating-features.md) page.**

### Updating the Documentation Website

Documentation for Lexos takes three forms:

1. The User Guide: Readable web pages with code samples describing the use of Lexos modules
2. The API docs: Technical documentation for Lexos classes and functions auto-generated from their docstrings
3. Tutorials: Jupyter notebooks (and accompanying sample data) guiding users through a workflow

Contributions for all three are welcome. If you design a new feature or module, you should submit new documentation to accompany it (or make pull requests for changes to the current documentation, if appropriate).

📖 **For instructions on how to edit, build, and run the [documentation website](https://scottkleinman.github.io/lexos/) locally see the [Lexos Documentation](documentation.md) page.**

### Submitting Your Contribution

Start by committing your changes. Make sure you write clear, descriptive commit messages.

An example using the command line would be

```bash
git add .
git commit -m "Fix bug in tokenizer and update docs"
```

However, you may use the `git` client of your choice.

1. **Push to Your Fork**

   ```bash
   git push origin fix-bug-in-tokenizer
   ```

2. **Open a Pull Request**

   - Go to the [Lexos repo](https://github.com/scottkleinman/lexos/) and open a pull request from your branch.
   - Fill out the description field by describing your changes.

## Code Conventions

All Python code must be compatible with Python 3.12+ and follow the project's code conventions.

📖 **The Lexos code conventions are described on the separate [Code Conventions](code-conventions.md) page.**

## Adding Tests

Lexos uses the [pytest](http://doc.pytest.org/) framework for testing. For more info on this, see the [pytest documentation](http://docs.pytest.org/en/latest/contents.html). Tests for Lexos modules and classes live in their own directories of the same name. For example, tests for the `Tokenizer` class can be found in [`/lexos/tests/tokenizer`](lexs/tests/tokenizer). To be interpreted and run, all test files and test functions need to be prefixed with `test_`.

📖 **For more guidelines and information on how to add tests, check out the separate [Lexos Tests](tests.md) page.**

## Code of Conduct

Lexos adheres to the [Contributor Covenant Code of Conduct](http://contributor-covenant.org/version/1/4/). By participating, you are expected to uphold this code.
