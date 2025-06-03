# Lexos API: Developer Setup Guide

This guide provides step-by-step instructions for setting up your local development environment to contribute to the Lexos API project. We use **uv** for dependency management and follow a centralized Git workflow.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting the Code](#getting-the-code)
- [Setting Up Your Python Environment with uv](#setting-up-your-python-environment-with-uv)
- [Installing spaCy Models](#installing-spacy-models)
- [Running Tests & Checking Coverage](#running-tests--checking-coverage)
- [Code Quality with Ruff (Linter & Formatter)](#code-quality-with-ruff-linter--formatter)
- [Git Workflow Basics](#git-workflow-basics)

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**  
    While `uv` can install Python, having a recent version pre-installed is often smoother.
- **Git**  
    For version control.

---

## Getting the Code

We use a centralized Git workflow, meaning everyone clones and pushes directly to the main repository (assuming you have write permission).

1. **Clone the Repository**  
     Open your terminal (PowerShell on Windows) and run:

     ```bash
     git clone https://github.com/scottkleinman/uv_lexos.git
     ```

2. **Navigate into the Project Directory**

     ```bash
     cd uv_lexos
     ```

---

## Setting Up Your Python Environment with uv

**uv** is our fast and efficient tool for managing project dependencies and virtual environments.

1. **Install uv Globally**  
     If you haven't already, install uv according to the [official documentation](https://astral.sh/uv/).

     - For **Windows (PowerShell)**:

         ```powershell
         irm https://astral.sh/uv/install.ps1 | iex
         ```

     - For **macOS/Linux (Bash/Zsh)**:

         ```bash
         curl -LsSf https://astral.sh/uv/install.sh | sh
         ```

2. **Verify Installation**  
     Close and reopen your terminal, then run:

     ```bash
     uv --version
     ```

3. **Create Virtual Environment and Install Dependencies**  
     From the `uv_lexos` project root:

     ```bash
     uv venv
     uv pip sync pyproject.toml
     ```

     This creates a `.venv` directory and installs all dependencies listed in `pyproject.toml`.

4. **Activate Your Virtual Environment**  
     You must activate your virtual environment every time you start a new terminal session.

     - **Windows (PowerShell):**

         ```powershell
         .venv\Scripts\activate
         ```

     - **macOS/Linux:**

         ```bash
         source .venv/bin/activate
         ```

     Your terminal prompt should now show `(lexos)` or `(.venv)` at the beginning.

---

## Installing spaCy Models

Our project relies on spaCy for Natural Language Processing. While spaCy itself is installed via uv, the language models need to be downloaded separately.

From your activated virtual environment in the project root, run:

```bash
uv run python -m spacy download xx_sent_ud_sm
uv run python -m spacy download en_core_web_sm
```

---

## Running Tests & Checking Coverage

We use **pytest** for testing and **coverage.py** to measure test coverage.

- **Run All Tests:**

    ```bash
    uv run pytest
    ```

- **Run Tests for a Specific Module (e.g., dtm):**

    ```bash
    uv run pytest tests/dtm
    ```

- **Run All Tests with Code Coverage:**

    ```bash
    uv run pytest --cov=src --cov-report=term-missing
    ```

    This displays a coverage report in your terminal.

- **Run Tests for a Specific Module with HTML Coverage Report:**

    ```bash
    uv run pytest --cov=src/lexos/dtm --cov-report=html tests/dtm
    ```

    After running, open `htmlcov/index.html` in your browser to inspect coverage.

---

## Code Quality with Ruff (Linter & Formatter)

**Ruff** is used for fast linting and formatting.

- **Install Ruff:**

    ```bash
    uv add ruff
    ```

- **Lint Your Code:**

    ```bash
    uv run ruff check .
    ```

- **Auto-Fix Linting Issues:**

    ```bash
    uv run ruff check . --fix
    ```

- **Format Your Code:**

    ```bash
    uv run ruff format .
    ```

---

## Git Workflow Basics

We follow a centralized workflow where everyone pushes to the main repository directly (requiring write permissions).

- **Stay Updated (Pull main into dev):**

    ```bash
    git checkout dev
    git pull origin main
    git push origin dev
    ```

- **Create a New Feature/Fix Branch:**

    ```bash
    git checkout dev
    git pull origin dev
    git checkout -b feature/your-awesome-feature  # or fix/your-bug-fix
    ```

- **Commit Your Changes:**

    ```bash
    git status
    git add .  # or git add <specific-files>
    git commit -m "feat: Add awesome feature"  # Use conventional commit messages
    ```

- **Push Your Branch:**

    ```bash
    git push -u origin feature/your-awesome-feature
    ```

    > **Note:** This requires write access to the `scottkleinman/uv_lexos` repository.

- **Open a Pull Request (PR):**  
    Once your work is ready and pushed, go to the GitHub repository in your browser. GitHub will prompt you to create a PR from your new branch to the `dev` branch. Fill out the description, assign reviewers, and submit.

- **Delete Local and Remote Branches (After Merge):**

    ```bash
    git checkout dev
    git pull origin dev
    git branch -d feature/your-awesome-feature
    git push origin --delete feature/your-awesome-feature
    ```

---
