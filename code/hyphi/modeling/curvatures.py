"""
Function to estimate curvature in graph embeddings.

Author: Nahid Torbati, Simon M. Hofmann
Years: 2024
"""

# %% Import
import warnings
from functools import wraps

import networkx as nx
import numpy as np

try:
    # plotly is an optional dependency; install it to enable graph visualisation functions.
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy import linalg
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

# %% Module-level constants

# Adjacency-method selectors used in adj_matrix()
_ADJ_METHOD_COSINE = 2
_ADJ_METHOD_SPEARMAN = 3
_ADJ_METHOD_PDIST = 4

# Minimum adjacency degree to annotate a node label in graph_vis_direct()
_ANNOTATION_DEGREE_THRESHOLD = 15

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def requires_go(func):
    """Decorate to guard functions that require plotly.graph_objects as `go`."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap function."""
        if go is None:
            raise ModuleNotFoundError(
                f"plotly is required for {func.__name__}(). Install it or avoid calling {func.__name__}()."
            )
        return func(*args, **kwargs)

    return wrapper


def adj_matrix(data, method: int = 1):
    """
    Compute the adjacency matrix with different methods.

    Methods
    -------
    1: Euclidean distances
    2: Cosine similarity
    3: Spearman Correlation
    4: Euclidean Distance with pdist function.

    """
    if method == 1:
        return np.round(euclidean_distances(data, data), decimals=3)
    if method == _ADJ_METHOD_COSINE:
        cos_sim = cosine_similarity(data)
        return np.round(np.ones_like(cos_sim) - cos_sim, decimals=3)
    if method == _ADJ_METHOD_SPEARMAN:
        n = data.shape[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                rho, _ = spearmanr(data[i], data[j])
                distance_matrix[i, j] = distance_matrix[j, i] = 1 - abs(rho)
        return np.round(distance_matrix, decimals=3)
    if method == _ADJ_METHOD_PDIST:
        raise NotImplementedError("Method 4 ('Euclidean Distance with pdist function') is not implemented yet.")
    return None


def compute_ricci_curvature(adj_mat: np.ndarray, threshold: float = 0.5, alpha: float = 0.5):
    """
    Compute Ollivier-Ricci curvature (ORC) from a thresholded adjacency matrix.

    This sets every edge weight to 1, so ORC here runs on the unweighted graph. That is an
    incidental implementation choice (Ollivier curvature is defined for weighted graphs too),
    NOT the reason ORC and Forman-Ricci magnitudes differ. The two are not numerically
    comparable in absolute terms because they are different kinds of quantity: ORC is a
    normalized optimal-transport ratio ``1 - W1(m_u, m_v) / d(u, v)`` bounded above by 1; its
    lower bound is ``-2(1 - alpha)`` for random-walk laziness ``alpha``, so at this function's
    default ``alpha=0.5`` it lies in [-1, 1] (intensive). By contrast ``compute_forman_curvature``
    returns an unbounded algebraic sum (degree- and weight-driven, extensive). To compare the two, do not
    compare raw magnitudes: min-max normalize each to compare relative-intensity profiles, or
    rank the edges and use a Spearman rank-order correlation.
    """
    adjacency_matrix = np.copy(adj_mat)
    adjacency_matrix[adj_mat > threshold] = 0
    # in this setting we don't have weight attribute for edges, we need to manually attribute the values to edges
    g_generated = nx.from_numpy_array(adjacency_matrix)
    for i, j in g_generated.edges():
        g_generated[i][j]["weight"] = 1  # this line assigns weight =1 to all edges
    orc = OllivierRicci(g_generated, weight="weight", alpha=alpha, verbose="INFO", chunksize=1)
    orc.compute_ricci_curvature()
    g_generated = orc.G.copy()
    ricci_curvtures = np.array(list(nx.get_edge_attributes(g_generated, "ricciCurvature").values()))
    return ricci_curvtures, g_generated


def compute_forman_curvature(adj_mat: np.ndarray, threshold: float = 0.5):
    """
    Compute Forman-Ricci curvature (FRC) from a thresholded adjacency matrix.

    This keeps the raw (post-threshold) adjacency values as edge weights, so FRC here runs on
    the weighted graph. Like the weight choice in ``compute_ricci_curvature`` this is incidental
    (Forman curvature is defined for unweighted graphs too). FRC is an unbounded algebraic sum,
    so its magnitudes are not directly comparable to the normalized Ollivier-Ricci ratio; see
    the note in ``compute_ricci_curvature`` for how to compare them (min-max normalize or rank).
    """
    adjacency_matrix = np.copy(adj_mat)
    adjacency_matrix[adj_mat > threshold] = 0
    G_generated: nx.Graph = nx.from_numpy_array(adjacency_matrix)  # noqa: N806
    frc = FormanRicci(G=G_generated, verbose="TRACE")
    frc.compute_ricci_curvature()
    G_frc: nx.Graph = frc.G.copy()  # save an intermediate result # noqa: N806
    forman_curvtures = np.array(list(nx.get_edge_attributes(G_frc, "formanCurvature").values()))
    return forman_curvtures, G_frc


def compute_curvature_graph(graph: nx.Graph, alpha: float = 0.5):
    """Compute curvature directly on the computed Graph."""
    orc = OllivierRicci(graph, weight="weight", alpha=alpha, method="OTDSinkhornMix", chunksize=1)
    orc.compute_ricci_curvature()
    _graph = orc.G.copy()
    ricci_curvtures = np.array(list(nx.get_edge_attributes(_graph, "ricciCurvature").values()))
    return ricci_curvtures, _graph


@requires_go
def graph_vis(dist_mat: np.ndarray, threshold: float) -> None:
    """Visualize the Graph with Plotly for distance-based graph embeddings."""
    dist = np.copy(dist_mat)
    dist[dist > threshold] = 0
    graph = nx.from_numpy_array(dist)

    pos = nx.spring_layout(graph)  # Positions for all nodes
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(showscale=True, colorscale="YlGnBu", size=10, color=[]),
    )
    for node in graph.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)

    # Color node points by the degree of each node
    for _, adjacencies in enumerate(graph.adjacency()):  # _ == node
        node_trace["marker"]["color"] += (len(adjacencies[1]),)

    # Create the Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Network graph made with Plotly",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Show the Plotly figure
    fig.show()
    print(nx.is_connected(graph))


@requires_go
def graph_vis_direct(graph: nx.Graph) -> None:
    """Visualize the graph with Plotly directly on a graph G."""
    pos = nx.spring_layout(graph)  # Positions for all nodes
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(showscale=True, colorscale="YlGnBu", size=10, color=[]),
    )
    for node in graph.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)

    # Color node points by the degree of each node
    for _, adjacencies in enumerate(graph.adjacency()):  # _ == node
        node_trace["marker"]["color"] += (len(adjacencies[1]),)

    annotations = []
    for node, adjacencies in graph.adjacency():
        if len(adjacencies) > _ANNOTATION_DEGREE_THRESHOLD:
            x, y = pos[node]
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text=str(node),  # Change this to the label of the node
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-30,
                )
            )
    # Create the Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Network graph made with Plotly",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Show the Plotly figure
    fig.show()
    print(nx.is_connected(graph))


@requires_go
def visualize_dist_distribution(dist_matrix: np.ndarray) -> None:
    """
    Visualize the distance distribution.

    It shows the histogram of the distance distribution
    and within illustrates the mean, mode, and standard deviation of the distribution.
    """
    upper_triangle_indices = np.triu_indices(dist_matrix.shape[0])
    distances = dist_matrix[upper_triangle_indices]

    fig = go.Figure(data=[go.Histogram(x=distances)])

    # Calculate mean, mode, and standard deviation of distances
    mean_distance = np.mean(distances)
    mode_distance = float(np.histogram(distances, bins=50)[1][np.argmax(np.histogram(distances, bins=50)[0])])
    std_distance = np.std(distances)

    # Add vertical dash lines for mean, mode, and standard deviation
    shapes = [
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=mean_distance,
            y0=0,
            x1=mean_distance,
            y1=1,
            line=dict(color="green", width=1, dash="dash"),
        ),
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=mode_distance,
            y0=0,
            x1=mode_distance,
            y1=1,
            line=dict(color="blue", width=1, dash="dash"),
        ),
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=mean_distance - std_distance,
            y0=0,
            x1=mean_distance - std_distance,
            y1=1,
            line=dict(color="red", width=1, dash="dash"),
        ),
    ]

    # Add annotations for statistical information at the top of the figure
    annotations = [
        dict(
            x=mean_distance,
            y=1.05,
            xref="x",
            yref="paper",
            text=f"Mean: {mean_distance:.2f}",
            showarrow=False,
            font=dict(size=12, color="black", family="Arial"),
            bgcolor="lightgrey",
            bordercolor="darkgrey",
            borderwidth=1,
            textangle=-45,
        ),
        dict(
            x=mode_distance,
            y=1.25,
            xref="x",
            yref="paper",
            text=f"Mode: {mode_distance:.2f}",
            showarrow=False,
            font=dict(size=12, color="black", family="Arial"),
            bgcolor="lightgrey",
            bordercolor="darkgrey",
            borderwidth=1,
            textangle=-45,
        ),
        dict(
            x=mean_distance - std_distance,
            y=1.3,
            xref="x",
            yref="paper",
            text=f"Std: {std_distance:.2f}",
            showarrow=False,
            font=dict(size=12, color="black", family="Arial"),
            bgcolor="lightgrey",
            bordercolor="darkgrey",
            borderwidth=1,
            textangle=-45,
        ),
    ]

    # Update layout with annotations and shapes
    fig.update_layout(
        title="Distribution of distances",
        xaxis_title="Distance",
        yaxis_title="Frequency",
        width=800,  # Adjust the width of the figure
        height=500,  # Adjust the height of the figure
        annotations=annotations,
        shapes=shapes,
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels for better readability
    )

    # Show plot
    fig.show()


@requires_go
def visualise_box_plot(data1, data2, data3) -> None:
    """
    Visualize the datasets with box plots.

    This is used for comparing the curvature values.
    Plot 3 box plot in the same figure.
    """
    fig = go.Figure()

    # Add subplots for each dataset
    fig.add_trace(go.Box(y=data1, name="Data 1", marker_color="skyblue", boxmean="sd"))
    fig.add_trace(go.Box(y=data2, name="Data 2", marker_color="salmon", boxmean="sd"))
    fig.add_trace(go.Box(y=data3, name="Data 3", marker_color="lightgreen", boxmean="sd"))

    # Add histograms for each dataset below the box plots
    fig.add_trace(
        go.Histogram(
            x=data1,
            name="Data 1 Hist",
            marker_color="skyblue",
            histnorm="probability density",
            visible="legendonly",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=data2,
            name="Data 2 Hist",
            marker_color="salmon",
            histnorm="probability density",
            visible="legendonly",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=data3,
            name="Data 3 Hist",
            marker_color="lightgreen",
            histnorm="probability density",
            visible="legendonly",
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        title="Box Plot and Histogram for Three Datasets",
        xaxis_title="Values",
        yaxis_title="Frequency",
        width=800,  # Adjust the width of the figure
        height=600,  # Adjust the height of the figure
        boxmode="group",  # Grouped box plots
        yaxis=dict(domain=[0.3, 1]),  # Adjust the y-axis domain for box plots
        yaxis2=dict(domain=[0, 0.2]),  # Adjust the y-axis domain for histograms
        xaxis2=dict(anchor="y2"),  # Anchor the x-axis of histograms to the y-axis of box plots
        showlegend=False,
    )

    # Show plot
    fig.show()


def nearest_neighbor_graph(data: np.ndarray, k: int = 5, weight: bool = False):
    """
    Create graph embedding with k nearest neighbors from given data (n_samples, m_features).

    :param data: data array of shape (n_samples, m_features)
    :param k: number of neighbors [default: k=5].
    :return: generated graph for nearest neighbors, distance matrix
    """
    # Lazy import: pynndescent pulls in numba/llvmlite (~3 s to import); only load
    # it here, where the approximate-kNN path actually runs, so importing this
    # module (and hence `import hyphi`) stays fast.
    from pynndescent import NNDescent  # noqa: PLC0415

    index = NNDescent(data)  # metric="euclidean" [default], n_neighbors=30 [default]
    n_neighbor, dist_neighbor = index.query(query_data=data, k=k)
    graph = nx.Graph()
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))
    # Add edges to the graph based on the n_neighbor array
    for i, neighbors in enumerate(n_neighbor):
        for neighbor, distance in zip(neighbors, dist_neighbor[i], strict=True):
            if i != neighbor:  # Avoid self-loops
                if weight:
                    graph.add_edge(i, neighbor, weight=distance)
                    distance_matrix[i, neighbor] = distance
                else:
                    graph.add_edge(i, neighbor, weight=1)
                    distance_matrix[i, neighbor] = distance
    distance_matrix = np.round(distance_matrix, decimals=3)

    return graph, distance_matrix


def sim_graph(data: np.ndarray, k: int = 5, weight: bool = True):
    """
    Create a graph embedding with k nearest neighbors from dissimilarity or distance matrix.

    :param data: data (dissimilarity, distance measures)
    :param k: number of nearest neighbors [default: k=5].
    :param weight: selective option for weighted or unweighted graph
    :return: generated graph for nearest neighbors, distance matrix

    """
    # Sort each row separately and store the sorted tuples of index and values
    sorted_data = [sorted(enumerate(row), key=lambda x: x[1]) for row in data]

    nn_indices = np.array([[index for index, _ in row[:k]] for row in sorted_data])
    nn_distances = np.array([[value for _, value in row[:k]] for row in sorted_data])

    graph = nx.Graph()
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))
    for i, neighbors in enumerate(nn_indices):
        for neighbor, distance in zip(neighbors, nn_distances[i], strict=True):
            if i == neighbor:
                continue  # skip self-loops (a node listing itself among its neighbours)
            # Add each edge once: weighted by the distance, or unweighted (weight 1) when
            # weight is False. The distance is always recorded in distance_matrix.
            graph.add_edge(i, neighbor, weight=distance if weight else 1)
            distance_matrix[i, neighbor] = distance
    distance_matrix = np.round(distance_matrix, decimals=3)
    return graph, distance_matrix


def adaptive_neighborhood_graph(X, k_min=5, k_max=50, density_method="knn"):
    """
    Compute an adaptive-k neighborhood graph and distances.

    The per-node neighbor count is scaled by local density (the inverse of the distance to the
    k_max-th neighbor). All edge weights are set to 1 regardless of the actual Euclidean
    distance; the distance is stored as a separate ``"distance"`` edge attribute.

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n_samples, n_features).
    k_min : int, optional
        Minimum number of neighbors per node. Default is 5.
    k_max : int, optional
        Maximum number of neighbors per node (also used to estimate density). Default is 50.
    density_method : str, optional
        Method for computing local density. Only ``"knn"`` is supported.

    Returns
    -------
    G : nx.Graph
        The adaptive-k neighborhood graph (unit edge weights, distance stored separately).
    distances : np.ndarray
        The k_max-nearest-neighbor distances per node.
    adaptive_k : np.ndarray
        The per-node neighbor count chosen by local density.

    """
    n_samples = X.shape[0]

    if density_method == "knn":
        nbrs = NearestNeighbors(n_neighbors=k_max, metric="euclidean").fit(X)
        distances, _ = nbrs.kneighbors(X)
        local_density = 1 / distances[:, -1]
    else:
        raise NotImplementedError("Only 'knn' density method is implemented")

    density_min, density_max = np.min(local_density), np.max(local_density)
    normalized_density = (local_density - density_min) / (density_max - density_min)

    adaptive_k = np.round(k_min + (k_max - k_min) * normalized_density).astype(int)

    G = nx.Graph()
    G.add_nodes_from(range(n_samples))

    for i in range(n_samples):
        k = adaptive_k[i]
        _, indices = nbrs.kneighbors(X[i].reshape(1, -1), n_neighbors=k + 1)
        for j in indices[0][1:]:
            if not G.has_edge(i, j):
                distance = np.linalg.norm(X[i] - X[j])
                G.add_edge(i, j, weight=1, distance=distance)  # weight= distance

    for i, (density, k) in enumerate(zip(local_density, adaptive_k)):
        G.nodes[i]["density"] = density
        G.nodes[i]["adaptive_k"] = k

    return G, distances, adaptive_k


def compute_laplacian_matrix(graph, g_weight="ricciCurvature"):
    """
    Compute the Laplacian matrix of a graph, weighted by an explicit edge attribute.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    g_weight : str
        Edge attribute used as the weight. The special value ``"non"`` computes the
        unweighted Laplacian. Note that ``"ricciCurvature"`` and ``"formanCurvature"`` are
        curvature OUTPUTS, not connectivity weights, so weighting by one of them is only
        meaningful on a curvature-annotated graph. If any edge lacks the requested attribute,
        NetworkX silently treats that edge's weight as 1; this function warns when none OR only
        some edges carry it, so the silent fallback to (partly) unit weights is visible.

    Returns
    -------
    np.ndarray
        The Laplacian matrix.

    """
    if g_weight == "non":
        return nx.laplacian_matrix(graph).toarray()
    n_edges = graph.number_of_edges()
    n_missing = sum(1 for _, _, data in graph.edges(data=True) if g_weight not in data)
    if n_edges > 0 and n_missing > 0:
        scope = "no edge carries" if n_missing == n_edges else f"{n_missing} of {n_edges} edges lack"
        warnings.warn(
            f"compute_laplacian_matrix: {scope} the attribute {g_weight!r}; "
            "NetworkX treats those edges as weight 1, so the Laplacian is (partly) unweighted.",
            stacklevel=2,
        )
    return nx.laplacian_matrix(graph, weight=g_weight).toarray()


def heat_kernel_distance(L1, L2):
    """
    Compute the diffusion distance between two Laplacian matrices using the heat kernel.

    Parameters
    ----------
    L1, L2: Laplacian matrices of the two graphs.
    t: Time parameter for the heat kernel (controls diffusion scale).

    Returns
    -------
    The diffusion distance between the two graphs.

    """
    evals1, evecs1 = linalg.eigh(L1)
    evals2, evecs2 = linalg.eigh(L2)

    t = np.logspace(-2, 2, 100)

    K1 = np.array([evecs1 @ np.diag(np.exp(-evals1 * ti)) @ evecs1.T for ti in t])
    K2 = np.array([evecs2 @ np.diag(np.exp(-evals2 * ti)) @ evecs2.T for ti in t])

    diff = np.array([np.linalg.norm(k1 - k2, "fro") for k1, k2 in zip(K1, K2)])

    return np.max(diff)


def update_weight(graph):
    """Update weight."""
    _g = graph.copy()
    for _, _, weight in _g.edges(data=True):  # _, _ == u, v
        weight["ricciCurvature"] += 1  # changed from 1 to two just for the experiment
    return _g


def update_weights(graph):
    """
    Return a copy of the graph where each edge's 'ricciCurvature' is updated.

    Each edge is update as:
        new_ricciCurvature = 1 - old_ricciCurvature
    """
    _g = graph.copy()
    for u, v, data in _g.edges(data=True):
        if "ricciCurvature" in data:
            data["ricciCurvature"] = 1 - data["ricciCurvature"]
        else:
            raise KeyError(f"Edge ({u}, {v}) missing 'ricciCurvature' attribute.")
    return _g


def kl_divergence(p, q):
    """
    Compute Kullback-Leibler Divergence between two probability distributions.

    Parameters
    ----------
    p (array-like): First probability distribution
    q (array-like): Second probability distribution

    Returns
    -------
    float: KL divergence between p and q

    """
    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize distributions to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Pad the shorter array with zeros to match lengths
    max_length = max(len(p), len(q))
    p_padded = np.pad(p, (0, max_length - len(p)), mode="constant")
    q_padded = np.pad(q, (0, max_length - len(q)), mode="constant")

    # Compute KL divergence
    # Use small epsilon to avoid log(0)
    epsilon = 1e-10

    # Mask to ignore zero probabilities in q
    mask = (p_padded > 0) & (q_padded > 0)

    kl_div = np.sum(p_padded[mask] * np.log(p_padded[mask] / (q_padded[mask] + epsilon)))

    return kl_div


def ricci_flow(
    graph: nx.Graph,
    iteration: int = 30,
    alpha: float = 0.5,
) -> nx.Graph:
    """Compute Ricci flow for a given graph."""
    orf = OllivierRicci(
        G=graph, weight="weight", alpha=alpha, method="OTD", chunksize=1, base=1, exp_power=0, verbose="INFO"
    )
    orf.compute_ricci_flow(iterations=iteration)
    graph: nx.Graph = orf.G.copy()
    return graph


def ricci_flow_metric(graph: nx.Graph) -> np.ndarray:
    """
    Compute the pairwise shortest path distance matrix for a weighted graph.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph with weighted edges.

    Returns
    -------
    np.ndarray
        A symmetric matrix (n x n) where the element (i, j) represents
        the shortest path distance between nodes i and j.

    """
    num_nodes = graph.number_of_nodes()
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            try:
                distance = nx.shortest_path_length(graph, source=i, target=j, weight="weight")
                dist_matrix[i, j] = dist_matrix[j, i] = distance
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = dist_matrix[j, i] = np.inf

    return dist_matrix


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
