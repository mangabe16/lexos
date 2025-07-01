# Tags

## Overview

The `tag` component of Scrubber is a submodule containing functions for transforming HTML and XML content (if you simply wish to remove all tags, using `remove.tags` is more efficient). These markup languages wrap content into **elements** indicated by angular brackets. Each element can further contain **attributes**, the **value** of which is contained within quotation marks. For example, the markup `<span id="1">John Smith</span>` indicates that the content "John Smith" is a "span" element with an "id" attribute, the value of which is "1". The functions in Scrubber's tag component allow you to manipulate the elements, attributes, and content by choosing a **selector** (usually the name of an element) and providing filters if only certain occurrences of this selector should be changed. This allows for some fairly nuanced transformations to be applied to the markup. For instance, in many cases Scrubber could be used to transform XML content into HTML markup for presentation on the web. This page offers an overview of the component functions. For a fuller description of their usage, see the [API documentation](api/scrubber/remove/).

Here is a list of the functions available in the `replace` component:

- `remove_attribute`: Removes all instances of an attribute in a specified element.
- `remove_comments`: Removes all comments from the text.
- `remove_doctype`: Removes the HTML document type declaration or XML declaration from the text.
- `remove_element`: Removes a specified HTML or XML element from the text, including the tag's content.
- `remove_tag`: Remove a tag from the text but retain the tag's content.
- `replace_attribute`: Replaces the value of an attribute with another value.
- `replace_tag`: Replaces a tag with anther tag.

Each function has parameters that allow you to specify how the function should behave. For example, you can specify which elements to target, whether to include or exclude certain attributes, and how to handle the content of the elements. The functions can be used individually or in combination to achieve complex transformations on your text. See the [API documentation](api/scrubber/remove/) for more details on the parameters available for each function.

!!! note
    In order to keep the functions simple, each function call can perform only a single transformation. To perform multiple transformations, it is necessary to call functions multiple times (or construct a pipeline that calls the functions iteratively). You may have to get to know your text's markup fairly well or inspect the results after each transformation in order to achieve the desired effects. When calling `tags` functions multiple times, the order in which the functions are called can make a considerable difference in the output..

## Example Using Direct Import

```python
from lexos.scrubber.tags import remove_element, replace_tag
text = "<p>Hello World</p><span>Hello, world!</span>"
scrubbed_text = remove_element(text, "span", "p") # Remove <p>
scrubbed_text = replace_tag(scrubbed_text, "span", "p") # Replace <span> with <p>
print(scrubbed_text)  # Output: "<p>Hello, world!</p>"
```

## Example Using in a Pipeline

The example below shows how the components can used in a Scrubber pipeline.

```python
from functools import partial
from lexos.scrubber import Scrubber
from lexos.scrubber.replace import remove_element, replace_tag

# Create partial functions for specific transformations
remove_p_by_class_value = partial(remove_element, attribute="class", value="remove")
replace_span = partial("span", "p")

scrubber = Scrubber()
scrubber.add_pipe([remove_p_by_class_value, replace_span])
text = "<p>Hello world</p><p class="remove">Hello World</p><p class="change">Hello, world!</p>"
scrubbed_text = scrubber.scrub(text)
print(scrubbed_text)  # Output: "<span>Hello world</span><span class="change">Hello, world!</span>"
```

The partial functions are created for clarity so that you can see what each function does. In practice, you could also use the functions directly in the pipeline without creating partials.
