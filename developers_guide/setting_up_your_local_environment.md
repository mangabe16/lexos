# Setting Up Your Local Development Environment

This guide provides step-by-step instructions for setting up your local development environment to contribute to the Lexos API project. We use **uv** for dependency management and follow a centralized Git workflow.

---

## Prerequisites

Before you begin, ensure you have the following installed:

### Python 3.12+

You can install Python using **uv** with `uv python install 3.12`. If you already have it installed using a distribution like Anaconda, **uv** will detect that installation.

### Git

**Git** is used for version control. If you don't have it installed, you can download it from [git-scm.com](https://git-scm.com/).

### Visual Studio Code (VS Code)

We recommend using [Visual Studio Code](https://code.visualstudio.com/) as your code editor. It has excellent support for Python and Git, and you can install extensions for code linting and formatting.

---

## Getting the Code

We use a centralized Git workflow, meaning everyone clones the main repository on GitHub (assuming you have write permission), makes their edits in branches locally, and then makes pull requests for merger into the main branch.

### Clone the Repository

Open your terminal (PowerShell on Windows) and run:

```bash
git clone https://github.com/scottkleinman/uv_lexos.git
```

Alternatively, if you are using VS Code or a client like GitHub Desktop, go to the GitHub repository page, click on the green "Code" button, and copy the HTTPS URL. Use this URL with your client's clone feature to clone the repository.

### Navigate into the Project Directory

Use whatever path leads to the `uv_lexos` directory.

```bash
cd uv_lexos
```

---

## Setting Up Your Python Environment with **uv**

**uv** is our fast and efficient tool for managing project dependencies and virtual environments.

### Install **uv** Globally

If you haven't already, install **uv** according to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/), follow these steps:

**For Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If you have Anaconda installed, you can run the command in a new Anaconda Prompt.

**For macOS/Linux (Bash/Zsh):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Some Mac users have reported issues with the above command. **uv** appears to install correctly, but you can't run **uv** commands. If you encounter this problem, you can try homebrew instead (you may need to install homebrew first):

```bash
brew install uv
```

### Verify the Installation

Close and reopen your terminal, then run:

```bash
uv --version
```

or

```bash
uv --help
```

### Create a Virtual Environment and Install the Project Dependencies

From the `uv_lexos` project root:

```bash
uv venv
uv sync
```

This creates a `.venv` directory and installs all dependencies listed in `pyproject.toml`.

---

### Installing spaCy Models

Our project relies on spaCy for Natural Language Processing. While spaCy itself is installed via **uv**, the language models need to be downloaded in a separate process.

Recent updates to `pyproject.toml` attempt to automatically download the default language models (multilingual and English) when you run `uv sync`. This feature is still experimental. If it does not work, you may need to manually download the models. From your activated virtual environment in the project root, run:

```bash
uv run python -m spacy download xx_sent_ud_sm
uv run python -m spacy download en_core_web_sm
```

---

### Activate the Virtual Environment

**uv** commands will intelligently activate the virtual environment when you run them. However, for other commands (like `python` or `pip`), you need to activate the virtual environment manually. So it's a god idea to do this every time you start a new terminal session.

**Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
 ```

Your terminal prompt should now show `(uv_lexos)` or `(.venv)` at the beginning.

---

## Setting up VS Code for Python Development

We recommend the following VS Code extensions for Python development:

- Even Better TOML: For better syntax highlighting and formatting of `pyproject.toml`.
- GitLens: Enhances Git capabilities in VS Code.
- Jupyter: If you plan to work with Jupyter notebooks.
- Markdownlint: For linting Markdown files.
- Pylance: Provides rich type information and IntelliSense for Python.
- Python: Official extension for Python development.
- Ruff: For linting and formatting Python code (see below for further instructions).

You can install these extensions from the VS Code marketplace or by searching for them in the Extensions view (Ctrl+Shift+X).

> [!IMPORTANT]
> The Lexos repo has a file called `.vscode.json`, which contains a path to a Windows Python exectuable in your virtual environment. If you are on Windows, you need to change that path to the appropriate one on your computer. If you are on a Mac, you will need to do the same, but you will need to provide a valid posix (Mac or Linux) path to Python (which will be a `.bin` file) in the `bin` folder of your virtual environment. Once you have done this, your Jupyter notebooks should work correctly in VS Code from anywhere within the project folder.

---

## Code Quality with Ruff (Linter & Formatter)

**Ruff** is used for fast linting and formatting.

### Install Ruff

```bash
uv add ruff
```

### Lint Your Code

```bash
uv run ruff check .
```

### Auto-Fix Linting Issues

```bash
uv run ruff check . --fix
```

### Format Your Code

```bash
uv run ruff format .
```

If you are using VSCode, you can set up **Ruff** as the default formatter. Add the following to your `settings.json` (in the command pallette, type `Preferences: Open Settings (JSON)`):

```json
{
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "ruff.linting.enabled": true,
    "ruff.linting.run": "onType"
}
```

You may wish to comment out `editor.formatOnSave` if you want to manually format your code. Then install the **Ruff** extension from the VSCode marketplace. You can automatically format your code by running the command palette (Ctrl+Shift+P) and selecting "Format Document" or "Format Selection" or by right-clicking and selecting these options.

## Markdown Linting

It is recommended that you install the **markdownlint** extension in VS Code for linting Markdown files when producing doumentation.
