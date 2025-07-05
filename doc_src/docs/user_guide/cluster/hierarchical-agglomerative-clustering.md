# Hierarchical Agglomerative Clustering

## Overview

!!! important
    This page is currently under construction.

Hierarchical cluster analysis is a method grouping similar data points into a hierarchy of nested clusters. It builds a tree-like structure that shows the relationships between clusters, where closer clusters are more similar. This approach helps reveal patterns and relationships within datasets, especially complex ones, by visualizing data groupings at multiple levels of similarity.

Hierarchical clustering may be agglomerative or divisive. Agglomerative clustering starts with each data point (typically a single document in Lexos) as a separate cluster and then merges them into larger clusters called **clades**. Divisive clustering takes the opposite approach, starting with all the data points in one cluster and splitting it. Currently, Lexos only provides an agglomerative algorithm.

The the results of hierarchical cluster analysis produced by this approach are typically represented visually as a **dendrogram**, which shows the hierarchical relationshops between clusters and the distance (similarity) between them.

In order to group documents into clusters, the algorithm relies on two important settings, a distance metric and a linkage criterion.

The **distance metric** metric is a measure used to quantify the similarity or dissimilarity between data points or clusters. A simple way to understand this is that a term occurring once in document A and twice in document B will have a distance of 1 between the two documents (based on that term alone). Of course, there are more numerous different ways in which we could calculate difference, and these will be discussed in more detail below.

The **linkage criterion** determines how the distance between clusters is calculated when merging. A simple way to think of this is to imagine two circles with dots in them representing the terms. The dots closest to the outer edge of the first circle will be closest to the dots closest to the outer edge of the second circle (in the direction where the circles are closest). We could use the short distance of these circles to select both to be merged into a single cluster at the next level of the hierarchy. But, equally, we could base our decision whether the merge them on the position of the dots in the centre. We have several other option to choose from, and these will be discussed in more detail below.

With this knowledge, we can describe the clustering algorithm.

1. We start with each document as its own cluster, or "leaf".
2. We identify the two closest clusters based on our chosen distance metric and linkage criterion.
3. We merge the two closest clusters into a single cluster. We then repeat steps 2 and 3 until all documen are in one cluster (the root of the tree).
4. We plot a dendrogram to represent this hierarchical structure.

An advantage of hierarchical clustering is that we do not need to choose the number of clusters in advance, and we can explore our clusters at different levels of granularity. However, our results can be sensitive to the choice of distance metric and linkage criterion. Further, the diagram represents a single cluster at the root level and a number of clusters equal to the number of documents at the leaf level. There may be more meaningful clusters on between hierarchy between these two extremes, but there is no clear method of determining a cut-off point (known as "cutting the dendrogram"). The discussion below will provide some guidance in dealing with these issues.

## How to Perform Hierarchical Agglomerative Clustering

To perform cluster analysis and generate a dendrogram, you will need document-term matrix produced by the DTM module. Then you simply import the Dendrogram class and feed it your DTM. You will also need a list of labels for the documents in your DTM object. In the example below, we will use the default settings for the distance metric and linkage criterion.

```python
# Import the Dendrogram Class
from lexos.cluster.dendrogram import Dendrogram

# Create an instance of the Dendrogram object
dendrogram = Dendrogram()

# Generate the Dendrogram.
 dendrogram(dtm=dtm, labels=dtm.labels, show=True)
```

<img src="../../../tutorials/cluster/dendrogram.png" alt="Sample dendrogram">

### Dendrogram Settings

When we create the `Dendrogram`, we need to tell it how to measure document similarity and how to connect those similarities into a tree. Here are the key parameters you can adjust:

- `dtm`: This is our "linguistic spreadsheet" (`dtm`) that we created in the previous step. It's the essential input for the tree.
- `metric`: This sets the distance metric, which tells the dendrogram how to measure the "distance" or dissimilarity between your documents. Shorter distances mean more similar documents. Options include `euclidean` (the default), `cosine`, and `cityblock`. For other options, see **Choosing a Distance Metric** below.
- `method`: This sets the linkage criterion, which determines how individual documents (or existing clusters of documents) are joined together to form larger branches and clusters in the tree. Option `average` (the default), `single`, `complete`, and `ward`. For further information, see **Choosing a Linkage Method** below.
  - `labels`: This is simply the list of descriptive names for your documents (e.g., "Poe", "Lippard") that we defined earlier. These will appear as the leaves (endpoints) on your tree.
- `orientation`: Controls the direction of the dendrogram. The default `"top"` orients the branches so that they extend downwards from root at the top. Other options are `"bottom"`, `"left"`, and `"right"`.
- `color_threshold`: If set, branches with a distance below this threshold will be colored differently from those above it. This helps visualize clusters at a certain distance level. You can try a number like `1.0` or `1.5` to see its effect.
- `show`: Controls whether the generated tree figure is displayed automatically. If `False`, the tree will not be displayed, but you display it later by calling `dendrogram.showfig()`. There are also methods that enable you to save it to a variable or file.
- `title`: Adds a title to your dendrogram plot.
- `figsize`: A tuple `(width, height)` in inches to set the size of the overall figure. For example, `(12, 8)` for a wider and taller plot.

### Plotting Dendrograms with Plotly

The `Dendrogram` class uses Python's matplotlib library to produce static images. However, in very large dendrograms, there is a danger of leaf labels overlapping, making the plot unreadable. In this case, you can use the Plotly plotter, which provides the ability to pan and zoom around the dendrogram, making it more readable. The Plotly plotter is also ideal if you are including the dendrogram in a web app.

To use the Plotly plotter, import the `PlotlyDendrogram` class, create an instance, and use it as above.

```python
# Import the PlotlyDendrogram class
from lexos.cluster.plotly_dendrogram import PlotlyDendrogram

# Create an instance of the dendrogram
dendrogram = PlotlyDendrogram()

# Call the instance with your DTM and labels
dendrogram(dtm=dtm, labels=labels, showfig=True)
```

<img src="../../../tutorials/cluster/plotly_dendrogram.png" alt="Sample Plotly dendrogram">

Note that the image above is a static representation and does not demonstrate Plotly's interactive features.

!!! Note
    Information on saving the dendrogram needs to be added here.

## Choosing a Distance Metric

One of the most important (and least well-documented) aspects of the hierarchical clustering method is the distance metric. Since we are representing texts as document vectors, it makes sense to define document similarity by comparing the vectors. One way to do this is to measure the distance between each pair of vectors. For example, if two vectors are visualized as lines in a triangle, the hypotenuse between these lines can be used as a measure of the distance between the two documents. This method of measuring how far apart two documents are is known as **Euclidean distance**. This is the default setting used by Lexos. It is good for general comparisons but can be sensitive to the overall length of documents (longer documents might naturally have larger term counts, increasing their "distance").

Another common metric is **cosine similarity**. Imagine each document as an arrow pointing in a specific linguistic "direction." Cosine similarity measures how much these arrows point in the same direction. If the angle at which they point is almost identical, the documents are very similar, even if one document is much longer than another. This is often an excellent choice for text analysis as it focuses on stylistic or thematic *direction* rather than raw word counts.

**Cityblock distance** also called Manhattan distance is another common metric. Imagine moving on a city grid where you can only go along streets (no diagonal shortcuts). This distance is the sum of the absolute differences for each term between two documents. This metric is useful when the individual differences in term counts are important.

Many other metrics are available (e.g., "jaccard", "chebyshev") from the Python scipy package, which Lexos runs under the hood. You can find a full list in the <code><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html" target="_blank">SciPy documentation</a></code>.

The table below provides some additional guidance on how to choose a distance metric.

|     |     |     |
| --- | --- | --- |
|     | Small Number of terms per segment | Large Number of terms per segment |
| Small Vocabulary | Bray-Curtis  <br>  <br>Hamming  <br>  <br>(e.g. character dialogue) | Euclidean  <br>  <br>Chebyshev  <br>  <br>Standardized Euclidean  <br>  <br>(e.g. chapters of books) |
| Large Vocabulary | Correlation  <br>  <br>Jaccard  <br>  <br>Squared Euclidean  <br>  <br>(e.g. non-epic poetry) | Cosine  <br>  <br>Manhattan  <br>  <br>Canberra  <br>  <br>(e.g. comparing entire corpora) |

## Choosing a Linkage Method

At each stage of the clustering process, a choice must be made about whether two clusters should be joined (and recall that a single document itself forms a cluster at the lowest level of the hierarchy). "Average" is the default, but you may choose other linkage methods by clicking the button.

- Average: This method is a compromise between single and complete linkage. It takes the average distance of the points in each cluster and uses the shortest average distance for deciding which cluster should be joined to the current one. When combining two clusters, this method considers the average distance between *all* pairs of documents in the two clusters. It tends to produce well-balanced clusters.
- Single: An intuitive means for doing this is to join the cluster containing a point (e.g. a term frequency) closest to the current cluster. This is known as single linkage, which joins clusters based on only a single point. In other words, clusters are joined based on the *closest* pair of documents between them. Single linkage does not take into account the rest of the points in the cluster, and the resulting dendrograms tend to have spread out clusters. This process is called "chaining". When this happens, where documents connect one after another, forming long, straggly branches.
- Complete: Complete linkage uses the opposite approach. It takes the two points furthest apart between the current cluster and the others. The cluster with the shortest distance to the current cluster is joined to it. Complete linkage thus takes into account all the points on the vector that come before the one with the maximum distance. It tends to produce compact, evenly distributed clusters, ensuring all documents within a cluster are relatively similar to each other.
- Weighted: The weighted average linkage performs the average linkage calculation but weights the distances based on the number of terms in the cluster. It, therefore, may be a good option when there is significant variation in the size of the documents under examination.
- Ward: This method aims to minimize the increase in "variance" (or spread) within clusters when they are merged. It tries to make clusters that are as "tight" and internally similar as possible. It often produces intuitive and well-structured clusters.

Which linkage criterion you choose depends greatly on the variability of your data and your expectations of its likely cluster structure. The fact that it is very difficult to predict this in advance may explain why the "compromise" of average linkage is the best default.

## Intepreting Dendrograms

### Choosing Where to Cut the Dendrogram

Hierarchical clustering is an exploratory technique, so it's often helpful to try different cut-off points and evaluate the resulting clusters. The height of the cut determines the number of clusters. A higher cut will result in fewer, larger clusters, while a lower cut will result in more, smaller clusters.

The best way to cut a dendrogram will always depend on the specific dataset and the goals of the analysis. There's no single "right" way to do it. To determine where to cut a dendrogram for clustering, you can use visual cues like the longest vertical distance between nodes, or consider numerical criteria like the Silhouette score or Dunn's index, or even trial and error. The choice of cut-off point depends on how many clusters you want and the desired level of similarity within each cluster.

Here is a procedure to use as a starting point:

1. Identify the longest vertical distance: Look for the longest vertical line (distance) between merging nodes on the dendrogram. Cutting at this point often reveals a natural separation between clusters.
2. Consider the overall structure: Observe how the data points are grouped at different height levels. You might choose a cut that separates well-defined, compact clusters or one that creates a few large clusters.

Lexos does not offer any numerical criteria for evaluating the quality of hierarchical clusters. However, you can count the number of leaves at your cutoff point and use that as the *k* value in a k-means clustering analysis to provide comparative evidence.

In addition, Lexos does not offer a method of drawing the dendrogram showing the cut. SciPy provides the `fcluster` method for cutting dendrograms, and we will hopefully implement it in the future. See also Information here about how to add truncate mode: https://stackoverflow.com/questions/70801281/how-can-i-plot-a-truncated-dendrogram-plot-using-plotly.

### Cluster Robustness

By cutting trying different distance metrics and linkage methods, as well as by cutting the dendrogram at different heights, you can evaluate the **robustness** of individual clusters. A "robust" cluster is one that persists, regardless of the setup criteria. If the cluster is sensitive to changes in the setup criteria, it is more likely to be a statistical artefact of those criteria, rather than a meaningful pattern. This, however, is a guideline, and its usefulness will depend on your data.

!!! note "Measuring Robustness with Bootstrap Consensus Trees"
    One way to automate the process of assessing cluster robustness is to implement Bootstrap Consensus Trees, which perform clustering with multiple settings and record the most-consistent clusters. See the section on **Bootstrap Consensus Trees** below.

## Clustermaps

A clustermap is a dendrogram attached to a heatmap, showing the relative similarity of documents using a colour scale. A clustermap combines the best of two worlds: hierarchical clustering (dendrograms) and pairwise similarity representation (heatmap).

The dendrogram on the top shows the hierarchical clustering of your documents based on their content (the rows of your DTM). The dendrogram on the left shows the same clustering, but rotated. As with standalone dendrograms, shorter branches mean documents (or clusters) are more similar. The order of documents along the heatmap axes is determined by these dendrograms, grouping similar documents together.

The heatmap grid visually represents the **pairwise distances** between your documents. Each cell at the intersection of a row and a column represents the distance between two documents. The color intensity on the heatmap will represent the distance between documents: typically, darker/different colors show greater distance (less similarity), while lighter/similar colors show shorter distance (more similarity). The diagonal of the heatmap will always be the same color, usually representing zero distance, as a document has zero distance from itself.

Clustermaps can be useful for observing the following:

- **Stylistic Groupings:** Does the heatmap show a strong block of similarity among authors from the same literary period or movement?
- **Thematic Cohesion:** If your DTM focused on specific themes, do documents discussing similar themes cluster together?
- **Influence or Divergence:** You might see how a text aligns with or diverges from others, giving insights into authorship, genre, or evolution of style.

Lexos can generate static clustermap images using the Python Seaborn library or dynamic images using Plotly.

### Using Seaborn¤

To generate a clustermap with Seaborn, use the following code:

```python
# Import the ClusterMap class
from lexos.cluster.clustermap import ClusterMap

# Create a ClusterMap object
cluster_map = ClusterMap(dtm, title="My Clustermap")

# Generate the plot and save it to a variable
fig = cluster_map(dtm=dtm_df, labels=labels)
```

<img src="../../../tutorials/cluster/clustermap.png" alt="Sample clustermap">

The distance title, distance metric, and linkage method, of the dendrogram can be set in the same way by passing title, metric, and method when instantiating the class or when the class is called.

The `ClusterMap` instance can be further customized with any  <code><a href="https://seaborn.pydata.org/generated/seaborn.clustermap.html" target="_blank">Seaborn.clustermap</a></code> parameter.

The clustermap plot is not shown by default. To display the plot, generate it with `show=True` or reference it with `ClusterMap.fig`.

!!! Note
    Information on saving the clustermap needs to be added here.

!!! warning
  Once the clustermap plot has been generated, it is inadvisable to use the modebar zoom and pan buttons because this tends to separate the heatmap from the dendrogram leaves. In the future, these buttons may be removed.

### Using Plotly¤

Plotly clustermaps are somewhat experimental and may not render plots that are as informative as Seaborn clustermaps. One advantage they have is that, instead of providing labels for each document at the bottom of the graph, they provide the document labels on the x and y axes, as well as the z (distance) score in the hovertext. This allows you to mouse over individual sections of the heatmap to see which documents are represented by that particular section, as well as the exact distance values.

Plotly clustermaps are constructed in the same manner to Seaborn clustermaps:

```python
# Import the ClusterMap class
from lexos.cluster.clustermap import PlotlyClustermap

# Create a PlotlyClustermap object
cluster_map = PlotlyClustermap(dtm, title="My Clustermap")

# Generate the plot and save it to a variable
fig = cluster_map(dtm=dtm_df, labels=labels)
```

<img src="../../../tutorials/cluster/plotly_clustermap.png" alt="Sample Plotly clustermap">

Note that the image above is a static representation and does not demonstrate Plotly's hover effects.

All the options for Plotly dendrograms are available with the following differences:

- Figure size is determined by configuring the `width` and `height` parameters.
- `colorscale` is the name of a built-in Plotly colorscale. This is applied to the heatmap and converted internally to a list of colorus to apply to the dendrograms.

Two additional parameters, `hide_upper` and `hide_side` allow you to hide the individual dendrograms.

!!! warning
    Once the clustermap plot has been generated, it is inadvisable to use the modebar zoom and pan buttons because this tends to separate the heatmap from the dendrogram leaves. It may even be advisable to remove these buttons from the modebar by default.

## Bootstrap Consensus Trees

A **Bootstrap Consensus Tree** is particularly robust because it doesn't just build one tree. Instead, it builds many, many trees by randomly sampling portions of your DTM. It then finds the "consensus": the most consistently appearing relationships across all those individual trees.

Generating bootrap consensus dendrograms involves submitting the same distance metric and linkage method parameters as regular dendrogram. However, there are a few additional parameters to set:

- `cutoff`: This is a confidence threshold. As mentioned, the BCT is built from many individual "bootstrap" trees. A `cutoff` of `0.5` (which means 50%) means that a specific grouping of documents (a branch on the tree) must appear in at least 50% of all the trees generated during the `iterations` to be considered reliable enough to show up in the final consensus tree. Higher `cutoff` values (e.g., 0.7 or 0.8) will result in a "sparser" tree, showing only the most robust and consistent relationships. Lower `cutoff` values (e.g., 0.3) will show more relationships, but some of these might be less statistically reliable.
- `iterations`: This is the number of "bootstrap resampling" rounds. In each round, Lexos takes a random 80% sample of the terms (columns) from your DTM and builds a tree from that sample. More iterations (e.g., 100, 1000) makes the consensus tree more statistically reliable and representative of the underlying relationships in your texts, as it averages out more variations. However, it will take longer to compute. Fewer iterations (e.g., 10, 20) are good for quick testing or initial explorations. For final research results, `100` (the default in the `BCT` class) or higher is often recommended if computation time allows.
- `replace`: This relates to how the terms are sampled during each iteration. Setting the value to "with" means a term column can be selected multiple times within a single 80% sample (allows for more randomness). The value "without" means each term column can only be selected once per 80% sample (more stable). This setting is generally suitable for DTMs as it ensures each unique term contributes uniquely within a sample.
- `doc_labels`: This is simply the list of descriptive names for your documents (e.g., "Poe", "Lippard") that we defined earlier. These will appear as the leaves (endpoints) on your tree.
- `text_color`: Sets the color for all text on the plot (axis labels, branch lengths, and document labels). You can use "rgb(R, G, B)" format. For example: `"rgb(0, 0, 0)"` (black) or `"rgb(255, 0, 0)"` (red).
- `layout`: Sets the layout of the dendrogram, either "rectangular" (the default) or "fan".

!!! warning
    Although bootstrap consensus trees take the same parameters as normal dendrograms, the keyword arguments are not consistent between the two classes. This needs to be fixed.

### Plotting Bootstrap Consensus Trees

To create a bootstrap consensus tree with rectangular layout, use the following code, setting the parameters describe above as required:

```python
# Import the BCT class
from lexos.cluster.bootstrap_consensus2 import BCT

# Create an instance of the BCT object
bct = BCT()

# Generate the Bootstrap Consensus Tree
fig = bct(
    doc_term_matrix=dtm,
    doc_labels=labels,
    cutoff=0.5,
    iterations=10,
    replace="without",
    text_color="rgb(0, 0, 0)",
    layout="rectangular",
    title="Bootstrap Consensus Tree (Rectangular Layout)"
    showfig=True
)
```

<img src="../../../tutorials/cluster/bootstrap_consensus_rectangular.png" alt="Sample Bootstrap Consensus Tree rectangular layout">

To generate a diagram with a fan layout, set `layout="fan"` (and adjust the `title` set above).

<img src="../../../tutorials/cluster/bootstrap_consensus_fan.png" alt="Sample Bootstrap Consensus Tree fan layout">
