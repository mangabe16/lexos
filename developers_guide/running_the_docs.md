# Running the Documentation Website

We use [MkDocs](https://www.mkdocs.org/) and Material for [MkDocs](https://squidfunk.github.io/mkdocs-material/) to build the documentation website. You must make build the website anew every time you build it. The website's HTML files are saved into a `uv/docs` folder which is ignored by git when you push code to GitHub. This prevents the hundreds of files generated for the website from being pushed to GitHub. As a result, you can only view the documentation website locally. When the site goes public, the `uv/docs` folder will be served using GitHub pages.

## Viewing the Website

To view the website, enter the following commands:

```bash
cd uv_lexos/doc_src
uv run mkdocs serve
```

The command line will provide you with a prompt to go to the website on a localhost (VS Code may also give you a prompt), or you can simply enter the displayed url in your browser. Hit `Control/Command + C` in your terminal to stop the localhost process when you are done.

## Editing the Website

At present, there is no complete guide to the procedures for editing the website. To understand how the site works, please consult the [MkDocs](https://www.mkdocs.org/) and Material for [MkDocs](https://squidfunk.github.io/mkdocs-material/) documentation. However, the basic procedure is that documentation files are authored in Markdown format and saved in the `doc_src/docs` folder. If you are serving the website when you edit a Markdown file, the website will update automatically a few seconds after you save the file. You can also rebuild the website without viewing it using `uv run mkdocs build`.

Your best bet for understanding how to format Markdown documents is to look for models in the existing documents. Note that the API documentation employs shortcodes using the Autodocstrings extension. These shortcodes automatically build the API text using the docstrings in the source code. However, you can also write Markdown around these shortcodes in order to provide further explanatory material.
