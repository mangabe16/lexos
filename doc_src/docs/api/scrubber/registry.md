# Registry

The registry component of Scrubber maintains a catalogue of registered functions that can be imported individually as needed. The registry enables the functions to be referenced by name using string values. The code registry is created and accessed using the <a href="https://github.com/explosion/catalogue" target="_blank">catalogue</a> library by Explosion.

::: lexos.scrubber.registry
  handler: python
  selection:
    members:
      - get_component
      - get_components
