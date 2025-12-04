# Lexos Documentation

Documentation for Lexos takes three forms:

1. The **User Guide**: Readable web pages with code samples describing the use of Lexos modules
2. The **API docs**: Technical documentation for Lexos classes and functions auto-generated from their docstrings
3. **Tutorials**: Jupyter notebooks (and accompanying sample data) guiding users through a workflow

Contributions for all three are welcome. If you design a new feature or module, you should submit new documentation to accompany it (or make pull requests for changes to the current documentation, if appropriate).

If you contribute a new feature to Lexos, you do not have to produce a tutorial for that feature, but it will be greatly appreciated.

## The Documentation Website

The documentation website is static website generated with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Each page is a Markdown file, which is converted to HTML when the site is built.

To preview changes to the documentation, serve it locally with

```bash
cd lexos/doc_src
uv run mkdocs serve
```

This will start a local server and automatically build a `docs` folder in the project root to contain the built website. If you do not want to serve the site, you can call `uv run mkdocs build`. However, in most case you will want to serve it to observe your changes in a web browser.

If you make a new page, you must add it to the `doc_src/docs/mkdocs.yml` configuration. If the page is under an `overview.md` page, check to see if the `overview.md` page has discussion or a table of contents where you might want to link to the new page. Note that the `mkdocs.yml` file is very easy to corrupt, so **be careful**.

Whether you make a change to an existing page or add a new one, your text should follow [GitHub Markdown conventions](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax), especially for code and code blocks. Markdown text should be formatted following  the default Markdown linting rules of of [Markdownlint](https://github.com/DavidAnson/markdownlint), except as specified in the `doc_src/.markdownlint.json` file. To see examples, you may find it helpful to review the current documentation files in [`doc_src/docs`](doc_src/docs ) or via `mkdocs serve`.

!!! note
    It is recommended that you install the Markdownlint extension in VS Code for linting Markdown files when producing documentation.

Before you make a pull request, check that the site builds properly in your local environment and make sure that your content does not contain any Markdown linting errors Lexos uses the default Markdown linting rules of of [Markdownlint](https://github.com/DavidAnson/markdownlint), except as specified in the `doc_src/.markdownlint.json` file. If you are using VS Code, the Markdownlint extension will show you any errors.

## The User Guide

The User Guide is intended to provide an entry-level introduction to the major features of Lexos. Pages in the User Guide are primarily intended to provide user-friendly overviews of the Lexos modules without being too exhaustive or too technical. It is acceptable to provide more technical explanations or notes for developers in admonitions (see below), but these should be relatively infrequent. Whether you are considering contributing to existing User Guide pages or adding a new one, use the existing pages as guides for the appropriate content, tone, and technical depth. For instance, you do not necessarily need to give an account of every parameter available in a given function, just those most likely to be used by an entry-level user. You can assume that the user has some familiarity with the Python programming language, but it may be worthwhile to define some terms or explain certain concepts.

Where possible, provide code samples in code blocks. Sample code should follow the conventions described on the [Code Conventions](code-conventions.md) page. If your code generates visualizations, provide links to static images in `.png` format. Typically, your page would be in a folder along with accompanying images.

User Guide pages should follow the Markdown principles noted above under **The Documentation Website**. Since User Guide pages are mostly written description, they should be well-edited and follow established standards for published writing. The Lexos documentation does not follow a specific style guide, but we recommend [The Chicago Manual of Style, 17th Edition](https://www.chicagomanualofstyle.org/) if you are in need of guidance but what written convention to adopt. This obviously only applies to documents in English. At present, the Lexos documentation does not have any pages in other languages, but we can imagine adding sections in other languages if users contribute them.

## The API Documentation

Each API documentation is meant primarily for developers, as it is highly technical, but it is also the only portion of the documentation that describes the full functionality of all Lexos features. For instance, a User Guide page or a Tutorial may describe only the major parameters of a function or method &mdash; those most likely to be used or most relevant to the workflow being discussed. If the User Guide or a tutorial does not mention a possible configuration or customization of a function, it is worth checking the API Documentation to see if the function has a parameter to do what you want.

Unlike the User Guide, the API documentation is mostly generated automatically from the type hints and docstrings in the Python source code. This information is converted to HTML with [`mkdocstrings`](https://mkdocstrings.github.io/) when you build the documentation website.

Each module should have its own folder, the name of which should correspond to the name of the module. Inside, there should be an `index.md` page, which is the starting point for the module's API documentation. The `index.md` file should contain a brief Markdown description of the module and a link to any other pages in the module's API documentation (these should be additional Markdown files). API pages should follow the Markdown principles noted above under **The Documentation Website**.

Each Markdown file in the module's API documentation should contain a [`mkdocstrings`](https://mkdocstrings.github.io/) template like the following:

```yaml
  ::: lexos.module
      handler: python
      selection:
        members:
          - MyClass
          - MyClass.my_method
          - my_function
```

The only exception is the `index.md` file, which only needs this template if it is the only Markdown file in the API's documentation.

In the template, replace `module` in `lexos.module` with the name of the module. Each class, method, property, or function in the source code listed under `members` will be read by `mkdocstrings` and its type hints and docstrings will be used to generate the documentation in HTML format. A short introduction (in Markdown) may be placed above the template, an further explanations can be added below, if necessary. Members will be listed in the documentation output in the order in which they are listed in the template. If you wish to provide further discussion between members, you can also document them as a series of smaller templates like the one below with discussion in between, instead of useing the `mkdocstrings` selection syntax shown above.

```yaml
### ::: lexos.cutter.TextCutter

    rendering:
      show_root_heading: true
      heading_level: 3
```

To preview changes to the documentation, serve it locally with

  ```bash
  uv run mkdocs serve
  ```

To create direct links to individual classes, properties, and methods anywhere in the documentation, use syntax like the following:

- [\`BaseLoader.data\`](base_loader/#lexos.io.base_loader.BaseLoader.data)
- [\`BaseLoader.load_dataset\`](base_loader/#lexos.io.base_loader.BaseLoader.load_dataset)

If you create API documentation for a new module, be sure to add it to the HTML table in `doc_src/docs/api/index.md`. When you add another row, make sure that you edit the `row-even` and `row-odd` class names so that the table striping alternates in the generated output.

## The Tutorials

The User Guide is the beginner's entry-point into using Lexos, but there is no substitute for hands-on experience. So, as part of the "documentation" offer a series of Jupyter notebooks with executable code where the user can try out Lexos features. Notebooks may or may not come with sample datasets. If they do, the dataset should be compressed to a zip file in a subfolder inside the tutorial's folder. This allows the user to download both the tutorial notebook and the data to run locally. If you create a new tutorial, make sure to add it to the table of contents in `docs_src/docs/tutorials/index.md`.

Tutorials should be aimed at entry-level users and their Markdown narrative should follow the same principles as outlined for the User Guide (except that sample code should mostly be in executable Python cells). All code blocks and Python cells should follow the conventions described on the [Code Conventions](code-conventions.md) page.

## Submitting Changes

Start by committing your changes. Make sure you write clear, descriptive commit messages.

An example using the command line would be
     ```bash
     git add .
     git commit -m "New documentation page about a fancy new feature"
     ```

However, you may use the `git` client of your choice.

1. **Push to Your Fork**

   ```bash
   git push origin new-module-doc
   ```

2. **Open a Pull Request**

   - Go to the original repo and open a pull request from your branch.
   - Fill out the pull request form describing the new module.

3. **Review and Collaboration**

   - Respond to feedback from maintainers.
   - Make requested changes and push updates.
   - Once approved, your changes will be merged!
