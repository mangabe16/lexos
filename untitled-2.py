import plotly.graph_objects as go
fig = go.Figure(data=go.Bar(y=[1, 2, 3]))
fig.to_image(format="png")  # Should return bytes, not hang
print("Kaleido is working.")