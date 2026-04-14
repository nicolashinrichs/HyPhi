"""
Visualize the curvature of an input graph.

Author: Nahid Torbati
Years: 2024
"""

# %% Import
import itertools
import os


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _get_rdbu_cmap():
    """Return RdBu colormap, preferring seaborn when available."""
    if sns is not None:
        return sns.color_palette("RdBu", as_cmap=True)
    return plt.get_cmap("RdBu")


def _require_plotly():
    if go is None:
        raise ModuleNotFoundError(
            "plotly is required for this visualization function. Install it with `pip install plotly`."
        )


def visualize_graph_with_curvature(
    graph: nx.Graph, name: str, save_dir: str = None, dpi: int = 300, figsize: tuple = (8, 8)
):
    """
    Visualize a graph with curvature values on edges using Seaborn colormap.

    Parameters
    ----------
    graph : nx.Graph
        The input graph with edge attribute "ricciCurvature".
    name : str
        Name of the graph (used in title and filename if saved).
    save_dir : str, optional
        Directory to save the visualization. If None, the plot is only displayed.
    dpi : int, default=300
        Resolution for saving the figure.
    figsize : tuple, default=(8, 8)
        Size of the matplotlib figure.
    """
    # Compute layout (spring for consistency)
    pos = nx.spring_layout(graph, seed=42)

    # Setup figure
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color="gray", node_size=5, alpha=0.3)

    # Suppress labels
    nx.draw_networkx_labels(graph, pos, labels={}, font_size=8)

    # Get curvature values
    curvature_values = nx.get_edge_attributes(graph, "ricciCurvature")
    edge_colors = [curvature_values[edge] for edge in graph.edges()]

    # Colormap
    palette = _get_rdbu_cmap()

    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_colors, edge_cmap=palette, edge_vmin=-1, edge_vmax=1)

    # Colorbar
    sm = ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="horizontal", shrink=0.6, pad=0.05)
    cbar.set_label("Ricci Curvature")

    # Title
    plt.title(f"Graph Visualization for {name}", fontsize=12)
    plt.axis("off")

    # Save or show
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"graph_curvature_{name}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Graph visualization saved to {save_path}")
    else:
        plt.show()


def visualize_graph_on_dataset_plot(G, pos):
    _require_plotly()
    # Extract nodes positions
    Xn, Yn, Zn = zip(*pos.values())

    # Create the node scatter plot
    node_trace = go.Scatter3d(x=Xn, y=Yn, z=Zn, mode="markers", marker=dict(size=3, color="gray"), hoverinfo="none")

    # Get curvature values without normalization
    curvature_values = nx.get_edge_attributes(G, "ricciCurvature")
    palette = _get_rdbu_cmap()

    # Create the edges traces
    edge_traces = []
    for i, (u, v) in enumerate(G.edges()):
        x_values = [pos[u][0], pos[v][0], None]  # None for line breaks
        y_values = [pos[u][1], pos[v][1], None]
        z_values = [pos[u][2], pos[v][2], None]

        # Map curvature value to color based on the RdBu colormap, directly using -1 to 1 range
        color_value = (curvature_values[(u, v)] + 1) / 2  # to fit RdBu
        color_rgb = palette(color_value)
        color = f"rgba({int(color_rgb[0] * 255)}, {int(color_rgb[1] * 255)}, {int(color_rgb[2] * 255)}, 0.8)"

        # Create a scatter plot for each edge
        edge_trace = go.Scatter3d(
            x=x_values, y=y_values, z=z_values, mode="lines", line=dict(color=color, width=5), hoverinfo="none"
        )
        edge_traces.append(edge_trace)

    # Set up the 3D figure with all traces
    fig = go.Figure(data=[node_trace] + edge_traces)

    # Update layout for better appearance
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", showbackground=False),
            yaxis=dict(title="Y", showbackground=False),
            zaxis=dict(title="Z", showbackground=False),
            bgcolor="white",
        ),
        title="3D Graph Visualization with Curvature-based Edge Colors",
        showlegend=False,
    )

    fig.show()


def visualize_graph_on_dataset_3d(G, pos, colors=None, node_size=20, edge_size=1.5):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot nodes
    X, Y, Z = zip(*pos)  # .values()
    if colors is not None and len(colors) > 0:
        ax.scatter(X, Y, Z, c=colors, cmap="Spectral", s=node_size)
    else:
        ax.scatter(X, Y, Z, c="gray", s=node_size)

    # Get curvature values for edges and normalize for color mapping
    curvature_values = nx.get_edge_attributes(G, "ricciCurvature")
    edge_colors = [(curvature_values[edge] + 1) / 2 for edge in G.edges()]  # Normalize to [0, 1]
    palette = _get_rdbu_cmap()

    # Plot edges with curvature-based colors
    for i, j in G.edges():
        x_values = [pos[i][0], pos[j][0]]
        y_values = [pos[i][1], pos[j][1]]
        z_values = [pos[i][2], pos[j][2]]
        color = palette(edge_colors.pop(0))
        ax.plot(x_values, y_values, z_values, color=color, lw=edge_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def visualize_graph_on_dataset_2d(G, dataset, colors=None, node_size=20, edge_size=1.5):
    """
    Visualize a graph on a 2D dataset layout.

    Parameters:
    - G: networkx.Graph
      The graph to visualize.
    - dataset: np.ndarray
      A 2D NumPy array where each row represents a point (x, y).
    """
    # Extract 2D positions for nodes from the dataset
    pos = {i: (point[0], point[1]) for i, point in enumerate(dataset)}

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot nodes
    X, Y = zip(*pos.values())
    if colors is not None and len(colors) > 0:
        ax.scatter(X, Y, c=colors, cmap="Spectral", s=node_size, label="Nodes")
    else:
        ax.scatter(X, Y, c="gray", s=node_size, label="Nodes")

    # Get curvature values for edges and normalize for color mapping
    curvature_values = nx.get_edge_attributes(G, "ricciCurvature")
    edge_colors = [(curvature_values[edge] + 1) / 2 for edge in G.edges()]  # Normalize to [0, 1]
    palette = _get_rdbu_cmap()

    # Plot edges with curvature-based colors
    for (i, j), edge_color in zip(G.edges(), edge_colors):
        x_values = [pos[i][0], pos[j][0]]
        y_values = [pos[i][1], pos[j][1]]
        color = palette(edge_color)
        ax.plot(x_values, y_values, color=color, lw=edge_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.title("Graph Visualization on 2D Dataset")
    # plt.grid(True)
    plt.show()


def curvature_distribution(data_list, name_list, plot_name, save_path=None, save=True):
    """
    Visualizes histograms of curvature distributions for multiple datasets.

    Parameters:
    - data_list: List of datasets (each dataset is a list or array of values).
    - name_list: List of names corresponding to each dataset.
    - plot_name: Name of the plot file (without extension).
    - save_path: Directory to save the plot (optional).
    - save: Boolean, if True saves the plot as a PNG file.
    """
    num_plots = len(data_list)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 4), sharex=True, sharey=True)

    if num_plots == 1:
        axes = [axes]

    for i, (data, name) in enumerate(zip(data_list, name_list)):
        if sns is not None:
            sns.histplot(data, bins=40, ax=axes[i])
        else:
            axes[i].hist(data, bins=40)
        axes[i].set_title(name)
        axes[i].set_xlim(-1, 1)
        axes[i].set_xlabel("Ricci curvature")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"{plot_name}.png")
        else:
            save_file = f"{plot_name}.png"
        plt.savefig(save_file)
    plt.show()


def heatmap_layers(data_list, name_list, plot_name, save_path=None, save=True):
    """
    Visualizes histograms of curvature distributions for multiple datasets.

    Parameters:
    - data_list: List of datasets (each dataset is a list or array of values).
    - name_list: List of names corresponding to each dataset.
    - plot_name: Name of the plot file (without extension).
    - save_path: Directory to save the plot (optional).
    - save: Boolean, if True saves the plot as a PNG file.
    """
    num_plots = len(data_list)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 4), sharex=True, sharey=True)

    if num_plots == 1:
        axes = [axes]

    for i, (data, name) in enumerate(zip(data_list, name_list)):
        if sns is not None:
            sns.heatmap(data, ax=axes[i])
        else:
            mat = np.atleast_2d(data)
            im = axes[i].imshow(mat, cmap="viridis", aspect="auto")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(name)
        axes[i].set_xlabel("Distance")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"{plot_name}.png")
        else:
            save_file = f"{plot_name}.png"
        plt.savefig(save_file)
    plt.show()


def node_to_partition(communities):
    result = {}
    for partition_id, community in enumerate(communities):
        for node in community:
            result[node] = partition_id
    return result


def visualize_graph_partitions_colors(graph: nx.Graph, partitions, name: str, save=True):
    """
    Visualize a graph with nodes colored by community partitions and edges color-coded by curvature.
    Includes a discrete legend for node partitions and a horizontal color bar for edge curvature.
    """

    node_to_community = node_to_partition(partitions)

    unique_communities = sorted(set(node_to_community.values()))
    num_communities = len(unique_communities)
    cmap = plt.cm.get_cmap("tab20", num_communities)
    community_to_color = {community: cmap(i) for i, community in enumerate(unique_communities)}

    pos = nx.spring_layout(graph)

    plt.figure(figsize=(10, 8))

    for community in unique_communities:
        nodes_in_community = [node for node in graph.nodes() if node_to_community[node] == community]
        color = community_to_color[community]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes_in_community,
            node_color=[color] * len(nodes_in_community),
            node_size=20,
            alpha=0.5,
            label=f"Community {community}",
        )

    curvature_values = nx.get_edge_attributes(graph, "ricciCurvature")
    edge_colors = [curvature_values[edge] for edge in graph.edges()]

    edge_palette = _get_rdbu_cmap()

    nx.draw_networkx_edges(
        graph, pos, width=2, edge_color=edge_colors, edge_cmap=edge_palette, edge_vmin=-1, edge_vmax=1
    )

    # nodes_to_annotate = [10, 28, 31]  # Example nodes to annotate
    # labels = {node: str(node) for node in nodes_to_annotate}
    # Uncomment below if node annotations are needed
    # nx.draw_networkx_labels(graph, pos, labels, font_size=12, font_color='red')

    sm_edges = ScalarMappable(cmap=edge_palette, norm=plt.Normalize(vmin=-1, vmax=1))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, orientation="horizontal", shrink=0.3, pad=0.03)
    cbar_edges.set_label("Curvature Value")

    plt.legend(loc="upper right", title="Communities", frameon=True)

    plt.title(f"Graph Visualization for {name}")
    plt.axis("off")
    if save:
        plt.savefig(f"{name}.svg")
    plt.show()


def visualize_graph_partitions_markers(
    graph: nx.Graph,
    partitions,
    name: str,
    save: bool = True,
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
    dpi: int = 250,
):
    """
    Visualize graph partitions with both color and marker encoding.

    Designed to remain readable even with many clusters by cycling through a
    stable marker set and using a discrete colormap.
    """
    node_to_community = node_to_partition(partitions)
    unique_communities = sorted(set(node_to_community.values()))
    n_clusters = len(unique_communities)

    # Stable, visibly distinct marker set for repeated cycling.
    marker_bank = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "8", "*"]
    marker_cycle = itertools.cycle(marker_bank)
    community_to_marker = {community: next(marker_cycle) for community in unique_communities}

    # Discrete color mapping; fallback to hsv if clusters exceed tab20.
    if n_clusters <= 20:
        cmap = plt.get_cmap("tab20", n_clusters)
        community_to_color = {community: cmap(i) for i, community in enumerate(unique_communities)}
    else:
        cmap = plt.get_cmap("hsv", n_clusters)
        community_to_color = {community: cmap(i) for i, community in enumerate(unique_communities)}

    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw edges (Ricci curvature if available; otherwise edge weights).
    curvature_values = nx.get_edge_attributes(graph, "ricciCurvature")
    if curvature_values:
        edge_colors = [curvature_values[e] for e in graph.edges()]
        edge_palette = _get_rdbu_cmap()
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=0.8,
            alpha=0.8,
            edge_color=edge_colors,
            edge_cmap=edge_palette,
            edge_vmin=-1,
            edge_vmax=1,
        )
        sm_edges = ScalarMappable(cmap=edge_palette, norm=plt.Normalize(vmin=-1, vmax=1))
        sm_edges.set_array([])
        cbar_edges = fig.colorbar(sm_edges, ax=ax, orientation="horizontal", shrink=0.5, pad=0.03)
        cbar_edges.set_label("Ricci Curvature")
    else:
        nx.draw_networkx_edges(graph, pos, ax=ax, width=0.6, alpha=0.4, edge_color="lightgray")

    # Draw nodes community by community.
    legend_handles = []
    for community in unique_communities:
        nodes_in_community = [n for n in graph.nodes() if node_to_community[n] == community]
        color = community_to_color[community]
        marker = community_to_marker[community]
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            nodelist=nodes_in_community,
            node_shape=marker,
            node_color=[color] * len(nodes_in_community),
            node_size=40,
            alpha=0.95,
            linewidths=0.2,
            edgecolors="black",
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                label=f"Cluster {community}",
            )
        )

    ax.set_title(title or f"Graph Visualization for {name} | clusters={n_clusters}")
    ax.axis("off")
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Clusters",
        ncol=1 if n_clusters < 20 else 2,
    )
    fig.tight_layout()

    if save:
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"{name}.png")
        else:
            save_file = f"{name}.png"
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_graph_on_dataset_with_colors(G, pos, partitions):
    """
    Visualize the graph with a given layout. Nodes are colored based on partitions,
    and edges are color-coded based on curvature values.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    node_to_community = node_to_partition(partitions)
    unique_communities = sorted(set(node_to_community.values()))
    num_communities = len(unique_communities)

    cmap = plt.cm.get_cmap("tab20", num_communities)
    community_to_color = {community: cmap(i) for i, community in enumerate(unique_communities)}

    X, Y, Z = zip(*pos.values())
    node_colors = [community_to_color[node_to_community[node]] for node in G.nodes()]

    ax.scatter(X, Y, Z, c=node_colors, s=20, alpha=0.7)

    curvature_values = nx.get_edge_attributes(G, "ricciCurvature")
    edge_colors = [(curvature_values[edge] + 1) / 2 for edge in G.edges()]  # Normalize to [0, 1]
    palette = _get_rdbu_cmap()

    for edge, normalized_value in zip(G.edges(), edge_colors):
        i, j = edge
        x_values = [pos[i][0], pos[j][0]]
        y_values = [pos[i][1], pos[j][1]]
        z_values = [pos[i][2], pos[j][2]]
        color = palette(normalized_value)
        ax.plot(x_values, y_values, z_values, color=color, lw=1.0, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Community {community}",
            markerfacecolor=community_to_color[community],
            markersize=10,
        )
        for community in unique_communities
    ]
    # ax.legend(handles=legend_handles, loc='upper right', title="Communities")

    plt.show()


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
