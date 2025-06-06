# Roadmap

This is an overview of the modules we want to work on for Sprint 2. Your individual responsibilities are on the [sign-up Google sheet](https://docs.google.com/spreadsheets/d/1zS3P8j8jeU5fP01g2e777Wz6JHGjryuYrCYClILjHUQ/edit?usp=sharing).

## Topwords

The Topwords tool in the Lexos web app helps you find terms that are more prominent in a certain document or class of documents than in other documents or classes of documents. We call these highly prominent terms "topwords" (even when the terms may not, strictly speaking, be "words".) The Topwords tool uses a Z-test to determine which terms are outliers beyond the normal range of distribution in a document or a group of documents. This is one way we can measure (at least by proxy) the terms that are significant, or "key", to a document's meaning.

The Textacy module implements several other algorithms for identifying "keyterms" (TextRank, Yake, sCake, SGRank). See [https://textacy.readthedocs.io/en/latest/api_reference/extract.html?highlight=keyterms](https://textacy.readthedocs.io/en/latest/api_reference/extract.html?highlight=keyterms). Since Lexos comes with Textacy pre-installed, it is easy to write a thin wrapper around Textacy's keyterms methods. Perhaps others could be added. Or perhaps we could create an architecture for algorithms to be added as plugins.

This new module would have to be built from scratch.


## K-Means

Most of the tools in the cluster module perform hierarchical clustering, which connect the most proximate objects into clusters (clades) and then iteratively link proximate clades until there is a single root. The number of clusters depends on what level of the tree you are at.

K-means clustering is a different form of cluster analysis in which you begin with a set number of clusters and assign objects to the pre-designated set of groups. The Lexos web app has a well-developed k-means tool, and our task is to adapt it for the Lexos API as a new submodule of the cluster module. The new submodule would follow the pattern of the other cluster submodules, performing calculations and visualisations. In general, we try to have one plotter class that produces a visualisation using matplotlib (for static images) and one that uses Plotly (for interactive images). The Lexos web app uses scatterplots and Voronoi diagrams for visualising k-means cluster analyses.

Although the other submodules in the cluster module, as well as the code for the web app, will provide something of a model to follow, this submodule needs to be built from scratch.

## Corpus/Statisics

The Corpus module encompasses the functions of both the Manage and the Statisics tools in the the web app. Its purpose is to allow users to manage documents and their metadata through a simple file storage system (in the future, it should be possible to plugin databases instead). Users should be able to load documents into a corpus, activate and deactivate them for analyse, and serialise them (and the entire corpus) to disk (more on that below). Users should also be able to generate basic statistics about their corpus (number of documents, etc.). More sophisticated statistics, such as interquartile range, should allow them to detect anomalous documents which may affect analysis.

There is already a preliminary version of the Corpus module in its own branch; however, the code has not been run, so there are bound to be bugs and other ways in which the interface could be simplified, improved, or optimised. Once it is working, test functions need to be written. The current version has the start of a plugin system whereby additional calculators could be added to generate other types of statistic, and you may wish to investigate what might be useful.

## Classification

Performing document classification is a long-running dream for Lexos. The only current implementation is the topic modelling module, but that is a unique implementation which does not attempt to assign class labels to documents. So we want to begin to develop a module from scratch to implement various types of classification algorithms. We have in mind a plugin system with some starter algorithms such as decision trees to which we could add other methods over time. A good starting point is this tutorial: [https://www.geeksforgeeks.org/text-classification-using-decision-trees-in-python/](https://www.geeksforgeeks.org/text-classification-using-decision-trees-in-python/).

## Key Words in Context (KWIC)

KWIC is a common technique for generating windows around search terms. For instance, if you have a sentence "The quick brown fox jumped over the lazy dog", and your search term is "jumped", a window of 2 terms around the search term would yield "brown fox **jumped** over the". These KWIC windows are a common format for [concordances](https://en.wikipedia.org/wiki/Concordance_(publishing)), which are often used in literary studies.

Textacy, which comes installed with Lexos, has a class for generating KWIC: [https://textacy.readthedocs.io/en/latest/api_reference/extract.html?highlight=kwic#module-textacy.extract.kwic](https://textacy.readthedocs.io/en/latest/api_reference/extract.html?highlight=kwic#module-textacy.extract.kwic). It would be easy to create a wrapper for this in the Lexos API. Whilst the new module could start as nothing more than that, it could develop into something more. The basic functionality involves search and display, so the module could be expanded to involve more search functionality. So, instead of simply outputting text, the module could mark the locations of "hits". In a web app, for instance, this would allow the app to display the particular portion of the text where the hit occurs, highlight the hit, and so on (but be careful, some of this might be better implemented on the front end).

So start with KWIC and see where it leads you.

## Documentation

We are going to finish our documentation with a lot of inconsistencies between the different modules, typos, and other problems (the tutorials will probably need stress testing). Volunteers who want to work on this will be working across the API code base. They will edit the documentation files but also perform bug fixes (or assign them to others), when bugs (inevitably) come to light.
