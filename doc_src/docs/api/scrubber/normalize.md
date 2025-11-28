# Normalize

The collection of "Normalize" functions take notations which are not standardized (such as - or * or ~ for a bullet point) and replaces them all with the same, normalized notation.

::: lexos.scrubber.normalize
  handler: python
  selection:
    members:
      - bullet_points
      - hyphenated_words
      - lower_case
      - quotation_marks
      - repeating_chars
      - unicode
      - whitespace
