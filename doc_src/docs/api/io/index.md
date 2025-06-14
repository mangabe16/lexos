# IO

!!! warning "For Documentation Developers"
    This is an example of how you would link to an individual class property or method.

    - [`BaseLoader.data`](base_loader/#lexos.io.base_loader.BaseLoader.data)
    - [`BaseLoader.load_dataset`](base_loader/#lexos.io.base_loader.BaseLoader.load_dataset)

The IO module manages input and output functions. It contains two main loader modules: [`loader`](loader.md) and [`data_loader`](data_loader.md). The `loader` provides an interface for loading texts in a variety of formats, whether from local files or from urls. The `data_loader` module provides method for loading or downloading large numbers of texts that are generally stored in a single file. Both modules inherit from a [`BaseLoader`](base_loader.md) class, which provides common functionality for loading texts.
