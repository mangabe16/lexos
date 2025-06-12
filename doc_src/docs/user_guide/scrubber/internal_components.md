# Internal Components

Using Scrubber components is made easier by an understanding of the underlying architecture and resources used by the individual functions. This page provides some additional for information to help you understand Scrubber's inner workings.

## The Registry

 Lexos ships with a **registry** of default functions which can be imported and referenced by name for easy access. Custom functions can be added to the registry by registering them as in the following example:

```python
#  Define the custom function
def title_case(text: str) -> str:
    """Our custom function to convert text to title case."""
    return text.title()

# Register the custom function
scrubber_components.register("title_case", func=title_case)
```

The registry is implemented using the Python <a href="https://github.com/explosion/catalogue" target="_blank">catalogue</a> library. See the `catalogue` documentation for further details about how to work with the registry.

## Pipeline

TBD.

## Resources

Many Scrubber functions use internal resources such as mappings of Unicode code points, regex values, or even helper functions. Inspecting the code in the [API Documentation](/api/scrubber/normalize/) can help you understand whether what Scrubber is doing under the hood is your desired behaviour or whether you need to modify Scrubber with a custom function.
