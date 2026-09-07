import networkx as nx
import plotly.graph_objects as go

class GraphVisualizer:
    """
    Renders interactive Plotly 2D/3D network graph visualizations for CGNIVEX.
    """

    COLOR_MAP = {
        "User": "#1f77b4",     # Blue
        "Post": "#ff7f0e",     # Orange
        "Topic": "#2ca02c",    # Green
        "Emotion": "#d62728",  # Red
        "Keyword": "#9467bd"   # Purple
    }

    @classmethod
    def create_plotly_figure(cls, graph: nx.Graph) -> go.Figure:
        """
        Converts a NetworkX graph into an interactive Plotly Figure object.
        """
        if graph.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title="Empty Graph")
            return fig

        pos = nx.spring_layout(graph, seed=42)

        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_data = graph.nodes[node]
            n_type = node_data.get("type", "Post")
            label = node_data.get("label", str(node))
            deg = graph.degree(node)

            node_text.append(f"Node: {label}<br>Type: {n_type}<br>Connections: {deg}")
            node_color.append(cls.COLOR_MAP.get(n_type, "#7f7f7f"))
            node_size.append(12 + deg * 2)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[graph.nodes[n].get("label", str(n))[:10] for n in graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=2
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="CGNIVEX Contextual Heterogeneous Graph Topology",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white"
            )
        )
        return fig
