# API Documentation: `clustermap.py`

::: lexos.cluster.clustermap
    handler: python
    selection:
      members:
        - Clustermap
        - PlotlyClustermap
        - PlotlyClusterGrid
        - _get_matrix
        - _create_dendrogram_traces
        - Clustermap.__init__
        - Clustermap.plot
        - Clustermap.save
        - Clustermap.show
        - Clustermap._format_data
        - Clustermap._z_score
        - Clustermap._standard_scale
        - Clustermap._process_mask
        - Clustermap._calculate_linkage
        - PlotlyClustermap.__init__
        - PlotlyClustermap.save
        - PlotlyClustermap.show
        - PlotlyClustermap.to_html
        - PlotlyClustermap.to_image
        - PlotlyClustermap.write_html
        - PlotlyClustermap.write_image
        - PlotlyClustermap._adjust_layout_for_hidden_upper
        - PlotlyClustermap._set_labels
        - PlotlyClusterGrid.__init__
        - PlotlyClusterGrid._format_data
        - PlotlyClusterGrid._z_score
        - PlotlyClusterGrid._standard_scale
        - PlotlyClusterGrid._process_mask
        - PlotlyClusterGrid._calculate_linkage

The [sync_script](sync_script.md) script synchronizes the heatmap and dendrogram axes in a Plotly clustermap.
It is added to the HTML output of the clustermap to ensure that when the user zooms or pans on one axis, the corresponding axes are updated accordingly.
