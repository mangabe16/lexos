# base_loader

## The BaseLoader Class

The `BaseLoader` class is the base class for all loaders in the IO module. Additional modules should inherit from BaseLoader.

::: lexos.io.base_loader
    handler: python
    selection:
      members:
        - BaseLoader
        - BaseLoader.__iter__
        - BaseLoader.data
        - BaseLoader.df
        - BaseLoader.records
        - BaseLoader.load_dataset
        - BaseLoader.dedupe
        - BaseLoader.reset
        - BaseLoader.show_duplicates
        - BaseLoader.to_csv
        - BaseLoader.to_excel
        - BaseLoader.to_json
