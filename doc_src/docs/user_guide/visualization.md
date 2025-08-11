# Visualization

## Overview

!!! important
    This page is currently under construction. In particular, images need to be added to demonstrate each of the code samples.

The Lexos `visualization` module provides a set of modular tools for visualizing the frequency of terms in textual data. Currently, the primary tool is the "word cloud", a cover term for a number of variant types of charts that display terms scaled according to their frequency. There are two basic types: traditional word clouds and packed circle charts, or bubble charts.

There are two methods of generating word cloud variants: pure Python approaches that produce static images and Javascript versions that generate charts in the web browser with interactive features. Each of these will be discussed below.

## Word Clouds

Word clouds display each submitted term in your text(s), scaled according to its frequency and laid out in a compact display so that you can easily "eyeball" which terms are most frequent. To produce a basic word cloud, import the `WordCloud` class and submit a simple text.

```python
from lexos.visualization.cloud import WordCloud

text = "This is a sample text to demonstrate how to produce a word cloud."
wc = WordCloud(data=text, title="My Word Cloud")
wc.show()
```

Notice that you can optionally supply a title with the `title` keyword. The last line (`wc.show()`) will display the word cloud.

The `WordCloud` class has a number of other useful parameters for modifying the appearance of the cloud. For instance, you can adjust the height and width of the image (in pixels), limit the number of terms that appear, or apply a round mask to make the word cloud appear circular.

```python
wc = WordCloud(data=text, title="My Word Cloud", height=200, width=200, limit=10, round=100)
```

!!! note
    Under the hood, word clouds are produced using the Python <code><a href="https://amueller.github.io/word_cloud/" target="_blank">WordCloud</a></code> and <code><a href="https://matplotlib.org/" target="_blank">matplotlib</a></code> libraries. Additional parameters can be passed to these libraries using the `opts` (`WordCloud`) and `figure_opts` (`matplotlib`). Parameters should be passed as dictionaries of keywords and values. For instance, you can set a light blue `wc = `WordCloud(data=text, opts={"background_color": "lightblue"})`.

You can save your word cloud to an image file or PDF using the save method:

```python
wc.save("my_wordcloud.png")
```

The file type will depend on what file extension you use.

## Bubble Charts

Bubble charts (also known as packed circle charts or bubble visualisations) arrange terms into labelled circles, which can sometimes be easier to read than traditional word clouds. They are produced in a similar manner.

```python
from lexos.visualization.cloud import WordCloud

text = "This is a sample text to demonstrate how to produce a bubble chart."
bc = BubbleChart(data=text, title="My Bubble Chart")
bc.show()
bc.save("my_bubble_chart.png")
```

Bubble charts must have the same height and width, so the figure dimensions are controlled with the `figsize` keyword with the value in inches.

```python
bc = BubbleChart(data=text, figsize=6.5, title="My Bubble Chart")
```

!!! note
    Bubble charts do not have the `opts` and `figure_opts` keywords you can access in the `WordCloud` class

## Types of Data

In the examples above, we have shown a raw text string submitted directly to the `WordCloud` and `BubbleChart` classes. In this circumstance, Lexos tokenises the text on whitespace. However, in a real-world scenario, you will probably want to perform some pre-processing on the text, such as removing punctuation and stop words. The `WordCloud` and `BubbleChart` classes will also accept a list of tokens like the following:

```python
# Original text
text = "This is a sample text to demonstrate how to produce a bubble chart."
# List of tokens with no punctuation or stop words, all lower case
tokens = ["this", "sample", "text", "demonstrate", "produce", "bubble", "chart"]
bc = BubbleChart(data=tokens)
```

You can also submit a spaCy `Doc` or list of spaCy `Doc` objects. However, you will most likely want to use spaCy to filter out unwanted tokens:

```python
tokens = [token.lower_ for token in doc if not token.is_punct and not token.is_stop]
```

This allows you to take advantage of pre-tokenised texts.

Another scenario is where you might have pre-tokenised texts is if you have already generated a document-term matrix with the Lexos `dtm` module. The `WordCloud` and `BubbleChart` classes accept a Lexos `DTM` object, as well as a pandas DataFrame produced by the `DTM.to_df()` method.

Finally, you may have a pre-generated list of term counts, such as is produced by the Python `collections.Counter` class:

```python
from collections import Counter

tokens = ["this", "is", "a", "sample", "text", "to", "demonstrate", "how", "to", "produce", "a", "bubble", "chart"]
counter = dict(Counter(tokens))
print(counter)
# {'this': 1, 'is': 1, 'a': 2, 'sample': 1, 'text': 1, 'to': 2, 'demonstrate': 1, 'how': 1, 'produce': 1, 'bubble': 1, 'chart': 1}
```

You can pass this dictionary directly to the `data` parameter in the `WordCloud` and `BubbleChart` classes.

## Limiting the Number of Terms

For both the `WordCloud` and `BubbleChart` classes, you can set the maximum number of terms to appear in the chart with the `limit` keyword. Note that this automatically displays the top *n* terms in your data. If you are submitting multiple documents, the counts will be based on all documents, unless you limit the number of documents (see below).

## Limiting the Number of Documents

If you pass a list of documents, a `DTM` object, or a pandas DataFrame, you may want to limit the chart to data from individual documents. You can do this by passing a list of document indexes (beginning with 0) to the `docs` keyword:

```python
bc = BubbleChart(data=dtm, docs[0, 2])
```

Only terms from the first and third documents in the document-term matrix will appear in the chart.

## Making Interactive Word Clouds with D3.js

Lexos provides the helper classes `D3WordCloud`, `D3BubbleChart`, and `D3MultiCloud` to generate beautiful, interactive visualisations using the Javascript library <a href="https://d3js.org/" target="_blank">D3.js</a>. Lexos D3 visualisations are standalone web pages, so they must be viewed in a web browser.

!!! note
    When you generate D3 visualisations with Lexos, a web browser should open automatically. However, this procedure may fail if your system does not have a default web browser set or your system saves temporary files in an unexpected location. In this circumstance, set the `auto_open` parameter to `False` and save the file. Then search for the file using your operating system and launch the file manually.

To generate a D3 word cloud, follow the procedure below (noting the import):

```python
from lexos.visualization.d3_wordcloud import D3WordCloud

wc = D3WordCloud(data=text)
wc.save("wordcloud.html")
```

By default, this will open a web browser with the HTML file in a temporary location. If you do not wish to open the file automatically, set `auto_open=False`.

As with the static image classes, you can set a title and chart dimensions (in pixels), and you can limit the number of terms and docs with the `limit` and `docs` keywords.

```python
wc = D3WordCloud(data=docs, docs=1, title="Custom Word Cloud", width=300, height=300, limit=30)
```

The `D3WordCloud` class provides a number of other parameters for customising the appearance of the word cloud:

- `font`: The name of the font to use.
- `spiral`: The spiral type to use for the word cloud, "archimedean" (the default) or "rectangular".
- `scale`: The scale type to use for the word cloud, "log", "sqrt", or "linear".
- `angle_count`: The number of angles to use for the word cloud.
- `angle_from`: The starting angle for the word cloud. The default is -60°.
- `angle_to`: The ending angle for the word cloud. The default is 60°.
- `background_color`: The background color of the word cloud. The default is white.
- `colorscale`: The name of a categorical d3 scale to use for the word cloud. The default is d3.scale.category20b". For other colorscales, see the <a href="https://d3js.org/d3-scale" target="_blank">d3-scale</a> documentation.

!!! note
    The available options are based on the exceptional <a href="https://www.jasondavies.com/wordcloud/" target="_blank">word cloud generator</a> produced by Jason Davies.

You can also generate a multicloud in D3:

```python
from lexos.visualization.d3_wordcloud import D3MultiCloud

# Create multi-cloud
mc = D3MultiCloud(
    data_sources=texts,
    title="Title",
    labels=None,
    cloud_width=250,
    cloud_height=250,
    columns=3
)
mc.save("multiclouds.html")
```

All the customisation parameters listed above for `D3WordCloud` are available. Notice, however, that few minor differences. You input your data using the `data_sources` keyword. Since each source document can have its own title, you can supply these titles as a list with the `labels` parameter (if you do not provide this, generic titles "Doc 1", "Doc 2", etc. will be used). Likewise, you can specify the dimensions of individual clouds (in pixels) with the `cloud_width` and `cloud_height` parameters. Finally, you can set the number of columns in the layout.

To generate a D3 bubble chart, you use the following code:

```python
from lexos.visualization.d3_bubbleviz import D3BubbleChart

bc = D3BubbleChart(data=text)
bc.save("bubble_chart.html)
```

Apart from the standard keywords, `D3BubbleChart` has two extra parameters for styling the chart.

- `margin`: A dictionary with the keys "top", "right", "bottom", and "left", used to configure the margin around the chart in pixels.
- `color`: The color scheme for the chart, either the name D3 color scheme or a list of custom colors. The default is "schemeCategory10". For other color schemes, see the <a href="https://d3js.org/d3-scale" target="_blank">d3-scale</a> documentation.

### Customising D3 Visualisations

D3 visualisations are standalone web pages, so they must be viewed in the browser. There are additional parameters for all three classes that allow you to choose whether to include the D3 Javascript in the web page (leading to a bigger file) or download it from the internet (which means it will only work if you have an internet connection). In most cases, it is safe to leave the default setting and include the D3 Javascript in the web page.

The actual logic used to produce the visualisation is not loaded from the internet, and it is not minimised. This allows you to open the HTML file and modify the Javascript, as well as the CSS styling, if you are comfortable doing so.

!!! note "Developer's Note"
    The visualisations are designed for display as web pages. However, if you are planning to incorporate them in an application, you may want to make more extensive changes. Each visualisation is produced from an HTML template, which is populated with variables passed from Python. You can design your own templates appropriate for your application's layout and specify the path to your templates with the `template` parameter.
