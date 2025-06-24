# Scrubber

## Overview

The Scrubber module provides a flexible, pipeline-based system for text cleaning and normalization as part of the Lexos project. It enables users to preprocess text by applying a customizable sequence of "scrubber components" (pipes) to remove, replace, or normalize elements such as punctuation, digits, whitespace, and more.

## Features

- Modular pipeline for text scrubbing
- Built-in registry of reusable scrubber components
- Easy addition and removal of pipeline components
- Support for custom components and configuration
- Integration with other Lexos modules
- Batch processing of texts via generator interface
- Robust error handling

## Submodules

### [Normalize](normalize.md)

The **normalize** submodule contains functions to normalize all [bullet points](normalize/#lexos.scrubber.normalize.bullet_points), [hyphenated words](normalize/#lexos.scrubber.normalize.hyphenated_words), [letters](normalize/#lexos.scrubber.normalize.lower_case) (to lowercase), [quotation marks](normalize/#lexos.scrubber.normalize.quotation_marks), [repeating characters](normalize/#lexos.scrubber.normalize.repeated_chars), [unicode](normalize/#lexos.scrubber.normalize.unicode), and [whitespace](normalize/#lexos.scrubber.normalize.whitespace) by replacing them with more standardized characters.

### [Pipeline](pipeline.md)

The **pipeline** submodule allows the user to create a pipeline which calls functions from the other submodules in a specific order.

### [Registry](registry.md)

The **registry** submodule contains functions [`get_component`](registry/#lexos.scrubber.registry.get_component) and [`get_components`](registry/#lexos.scrubber.registry.get_component) to get one component from a string, or multiple from a tuple, respectively.

### [Remove](remove.md)

The **remove** submodule contains functions to remove [accents](remove/#lexos.scrubber.remove.accents), all [brackets](remove/#lexos.scrubber.remove.brackets) ( ) [ ] { } and the text within them, [digits](remove/#lexos.scrubber.remove.digits), [newlines](remove/#lexos.scrubber.remove.newlines), given regex [patterns](remove/#lexos.scrubber.remove.patterns), [Project Gutenberg headers](remove/#lexos.scrubber.remove.project_gutenberg_headers), [punctuation](remove/#lexos.scrubber.remove.punctuation), [tabs](remove/#lexos.scrubber.remove.tabs), and [tags](remove/#lexos.scrubber.remove.tags).

### [Replace](replace.md)

The **replace** submodule contains functions which replace [currency symbols](replace/#lexos.scrubber.replace.currency_symbols), [digits](replace/#lexos.scrubber.replace.digits), [emails](replace/#lexos.scrubber.replace.emails), [emojis](replace/#lexos.scrubber.replace.emojis), [hashtags](replace/#lexos.scrubber.replace.hashtags), given regex [patterns](replace/#lexos.scrubber.replace.pattern), [phone numbers](replace/#lexos.scrubber.replace.phone_numbers), [punctuation](replace/#lexos.scrubber.replace.punctuation), [special characters](replace/#lexos.scrubber.replace.special_characters), [urls](replace/#lexos.scrubber.replace.urls), and [user handles](replace/#lexos.scrubber.replace.user_handles) with a string of the form `_TYPE_`.

### [Resources](resources.md)

The **resources** submodule contains the [HTMLTextExtractor](resources/#lexos.scrubber.resources.HTMLTextExtractor) class, a subclass of [html.parser.HTMLParser](https://docs.python.org/3/library/html.parser.html). 

### [Scrubber](scrubber.md)

The **scrubber** submodule contains the main logic for the Scrubber module. It contains the [`Pipe`](scrubber/#lexos.scrubber.scrubber.Pipe) dataclass and the [`Scrubber`](scrubber/#lexos.scrubber.scrubber.Scrubber) class. The Pipe class contains only a call method and the Scrubber class contains an initialization method along with methods [`add_pipe`](scrubber/#lexos.scrubber.scrubber.Scrubber.add_pipe), [`pipe`](scrubber/#lexos.scrubber.scrubber.Scrubber.pipe), [`remove_pipe`](scrubber/#lexos.scrubber.scrubber.Scrubber.remove_pipe), [`reset`](scrubber/#lexos.scrubber.scrubber.Scrubber.reset), and [`scrub`](scrubber/#lexos.scrubber.scrubber.Scrubber.scrub). The Scrubber class also contains the attribute [`pipes`](scrubber/#lexos.scrubber.scrubber.Scrubber.pipes) which returns a list of the pipeline components. The submodule also includes the function [`scrub`](scrubber/#lexos.scrubber.scrubber.scrub) which takes in the text to scrub, the pipeline, and the optional factory and returns the scrubbed text

### [Tags](tags.md)

The **tags** submodule uses [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for several functions to [remove attributes](tags/#lexos.scrubber.tags.remove_attribute), [remove comments](tags/#lexos.scrubber.tags.remove_comments), [remove doctypes](tags/#lexos.scrubber.tags.remove_doctype), [remove elements](tags/#lexos.scrubber.tags.remove_element), [remove tags](tags/#lexos.scrubber.tags.remove_tag), [replace attributes](tags/#lexos.scrubber.tags.replace_attribute), and [replace tags](tags/#lexos.scrubber.tags.replace_tag) in HTML and XML files.

### [Utils](utils.md)

The **utils** submodule contains the function [`get_tags`](utils/#lexos.scrubber.utils.get_tags).

