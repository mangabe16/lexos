# Data Loader

## The `DataLoader` Class

The `DataLoader` class is the main class for loading datasets in various formats. It tries to be "smart" detecting the format as well as can be done so that you can use a common interface to load content regardless of source.

::: lexos.io.data_loader
    handler: python
    selection:
      members:
        - DataLoader
        - DataLoader.__init__
        - DataLoader.load
        - DataLoader.save
        - DataLoader.to_csv
        - DataLoader.to_json
        - DataLoader.to_excel
