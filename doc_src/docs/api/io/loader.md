# loader

## The `Loader` Class

The `Loader` class is the main class for loading files in various formats. It tries to be "smart" detecting the format as well as can be done so that you can use a common interface to load content regardless of source.

::: lexos.io.loader
    handler: python
    selection:
      members:
        - Loader
        - Loader.__init__
        - Loader._get_mime_type
        - Loader._load_docx_file
        - Loader._load_pdf_file
        - Loader._load_text_file
        - Loader._load_zip_file
        - Loader.load_dataset
        - Loader.load
        - Loader.loads
