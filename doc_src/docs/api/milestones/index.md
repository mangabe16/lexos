
## Sample API Doc Generated with Pydantic and mkdocstrings

This is a sample API doc generated with Pydantic and mkdocstrings. The API doc is generated from the Pydantic models and includes information about the models, their fields, and their types. The doc also includes examples of how to use the models and their methods.

`griffe_pydantic` is still using the `description` keyword to populate the docstring. The field name becomes a link; when clicked, it goes down to an entry with the description prose. If you use `json_schema_extra`, there is no link and no entry. Why did I ever decide to use `json_schema_extra`? I think I must have misread [https://docs.pydantic.dev/latest/migration/#changes-to-pydanticfield](https://docs.pydantic.dev/latest/migration/#changes-to-pydanticfield) and thought that `description` was going away. I think I was wrong. I think `description` is still the way to go.

```python

### ::: lexos.milestones.token_milestones.TokenMilestones
    rendering:
      show_root_heading: true
      heading_level: 3
