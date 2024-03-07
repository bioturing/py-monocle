from typing import Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import KDTree
from scipy.sparse import csgraph
from scipy.stats import norm
import networkx as nx


def find_nearest_principal_nodes(
    matrix: np.ndarray,
    centroids: np.ndarray,
):
  """Find the nearest nodes on the principal graph of matrix.

  Parameters
  ----------
  matrix : ``ndarray``
    The matrix to find the nearest edge, recommended UMAP embeddings.
  centroids : ``ndarray``
    The centroids of the principal graph.

  Returns
  -------
  nearest_nodes: ``list``
    The indices of the nearest nodes.
  """
  kdtree = KDTree(centroids)
  nearest_nodes = kdtree.query(matrix, k=1)[1]
  return nearest_nodes.reshape(-1)


def find_nearest_principal_edges(
    matrix: np.ndarray,
    centroids: np.ndarray,
    mst: sparse.csr_matrix,
):
  """Find the nearest edges on the principal graph of matrix.

  Parameters
  ----------
  matrix : ``ndarray``
    The matrix to find the nearest edge, recommended UMAP embeddings.
  centroids : ``ndarray``
    The centroids of the principal graph.
  mst : ``csr_matrix``
    The minimum spanning tree of the principal graph.

  Returns
  -------
  nearest_edges: ``list``
    The indices of the two vertices of the nearest edges.
  """
  if not isinstance(mst, sparse.csr_matrix):
    mst = sparse.csr_matrix(mst)
  mst = mst + mst.transpose()
  vertices = (np.diff(mst.indptr) > 0).nonzero()[0]

  nearest_nodes = find_nearest_principal_nodes(matrix, centroids[vertices])
  nearest_nodes = vertices[nearest_nodes]
  second_nodes = np.empty_like(nearest_nodes)

  for node in np.unique(nearest_nodes):
    edge_vertices = mst[node].indices
    indexes = (nearest_nodes == node).reshape(-1)

    distances = np.array([
      np.sum(np.square(matrix[indexes] - centroids[vertice]), axis=1)
      for vertice in edge_vertices
    ])
    second_nodes[indexes] = edge_vertices[np.argmin(distances, axis=0)]
  return [nearest_nodes, second_nodes]


def pairwise_distance(
    matrix_1: np.ndarray,
    matrix_2: np.ndarray,
):
  """Calculate pairwise distances of two matrix

  Parameters
  ----------
  matrix_1 : ``ndarray``
    First matrix.
  matrix_2 : ``ndarray``
    Second matrix.

  Returns
  -------
  pairwise_distances: ``ndarray``
    Pairwise distances from matrix_1 to matrix_2.
  """
  norm_sqr_1 = np.sum(matrix_1 ** 2, axis=1).reshape((-1, 1))
  norm_sqr_2 = np.sum(matrix_2 ** 2, axis=1).reshape((1, -1))
  return norm_sqr_1 + norm_sqr_2 - 2 * matrix_1 @ matrix_2.T


def minimum_spanning_tree(centroids: np.ndarray, k_nn: int = 25):
  """Find minimum spanning tree of centroids.

  Parameters
  ----------
  centroids : ``ndarray``
    Nodes of minimum spanning tree.
  k_nn: ``int``, default: 25
    Number of nearest neighbors.

  Returns
  -------
  mst : ``csr_matrix``
    The minimum spanning tree of centroids.
  """
  graph_shape = (centroids.shape[0], centroids.shape[0])

  k_nn = min(k_nn, centroids.shape[0])
  kdtree = KDTree(centroids)
  knn_distances, knn_indices = kdtree.query(centroids, k_nn)
  graph = sparse.csr_matrix(
    (knn_distances.flatten(),
     knn_indices.flatten(),
     np.arange(0, knn_distances.size + 1, k_nn)),
    shape=graph_shape)

  n_components, labels = csgraph.connected_components(graph, directed=False)

  graph: nx.Graph = nx.from_scipy_sparse_array(graph)
  if n_components > 1:
    connected_edges = []
    for i in range(1, n_components):
      cpns_i = (labels == i).nonzero()[0]
      kdtree = KDTree(centroids[cpns_i])
      for j in range(i):
        cpns_j = (labels == j).nonzero()[0]
        knn_distances, knn_indices = kdtree.query(centroids[cpns_j], 1)
        nearest_node = np.argmin(knn_distances.flatten())
        connected_edges.append([cpns_j[nearest_node],
                                cpns_i[knn_indices.flatten()[nearest_node]],
                                knn_distances.flatten()[nearest_node],
                              ])
    graph.add_weighted_edges_from(connected_edges)

  mst = nx.to_scipy_sparse_array(nx.minimum_spanning_tree(graph))
  mst.indices = mst.indices.astype("int32")
  mst.indptr = mst.indptr.astype("int32")
  return mst


def compute_cluster_connectivities(
    clusters: np.ndarray,
    matrix: Optional[np.ndarray] = None,
    graph: Optional[sparse.csr_matrix] = None,
    n_neighbors: int = 25,
):
  """Find all tip cells and check if they fit criteria to add connections.

  Parameters
  ----------
  clusters: ``ndarray``
    Clusters of cells.
  matrix: ``Optional[np.ndarray]``, default: None
    Coordinates of the cells, recommended UMAP embeddings.
  graph: ``Optional[sparse.csr_matrix]``, default: None
    Connectivities graph of matrix.
  n_neighbors: ``int``, default: 25
    Number of neighbors to compute connectivities graph.

  Returns
  -------
  cluster_mat: ``ndarray``
    Complement of cumulative distribution of cluster connectivities.
  """
  if graph is None:
    assert matrix is not None, "Request one of matrix or neighbor graph."
    kdtree = KDTree(matrix)
    n_neighbors = min(n_neighbors, matrix.shape[0])
    knn_distances, knn_indices = kdtree.query(matrix, n_neighbors)

    from umap.umap_ import fuzzy_simplicial_set

    graph = fuzzy_simplicial_set(
      matrix, n_neighbors, 0, None,
      knn_dists=np.sqrt(knn_distances), knn_indices=knn_indices
    )[0]
    graph = graph.tocsr()
  graph.data[:] = 1

  components, clusters = np.unique(clusters, return_inverse=True)
  clusters = sparse.csr_matrix((
    np.ones(len(clusters)), clusters, np.arange(len(clusters) + 1)
  ), shape=(len(clusters), len(components)))

  n_links = clusters.transpose() @ graph @ clusters
  n_links = n_links.toarray()
  np.fill_diagonal(n_links, 0)
  edges_per_cls = np.sum(n_links, axis=1)
  total_edges = np.sum(edges_per_cls)

  probs = edges_per_cls /total_edges
  theta = probs.reshape((-1, 1)) @ probs.reshape((1, -1))
  n_links_var = theta * (1 - theta) / total_edges
  n_links = n_links / total_edges - theta
  cluster_mat = 1 - norm.cdf(n_links, 0, np.sqrt(n_links_var))
  adjust_p_value = np.minimum(cluster_mat * cluster_mat.size, 1)

  return adjust_p_value


def create_projected_graph(
    matrix: np.ndarray,
    projected_points: np.ndarray,
    centroids: np.ndarray,
    mst: sparse.csr_matrix,
) -> nx.Graph:
  """Find minimum spanning tree of projected points based on the core graph.

  Parameters
  ----------
  matrix : ``ndarray``
    The expression matrix, recommended UMAP embeddings.
  projected_points : ``ndarray``
    Projected points of matrix onto the principal graph.
  centroids : ``ndarray``
    The centroids of the principal graph.
  mst : ``csr_matrix``
    The symmetrical minimum spanning tree of the principal graph.

  Returns
  -------
  projectd_graph : ``nx.Graph``
    The minimum spanning tree networkx.Graph of projected points.
  """
  nearest_edges = find_nearest_principal_edges(
    matrix, centroids, mst
  )
  first_nodes = np.minimum(*nearest_edges)
  second_nodes = np.maximum(*nearest_edges)
  edge_group = first_nodes * len(centroids) + second_nodes
  distances = np.sqrt(np.sum(
    np.square(centroids[first_nodes] - projected_points), axis=1))
  df = pd.DataFrame({
    "cell_id": np.arange(len(matrix)),
    "root": first_nodes + len(matrix),
    "group": edge_group,
    "distances": distances,
  })
  sort_df = df.sort_values(["group", "distances"]).groupby("group")
  heads = sort_df["cell_id"].head(-1)
  tails = sort_df["cell_id"].tail(-1)
  df.iloc[tails, df.columns.get_loc("root")] = heads
  df.iloc[tails, df.columns.get_loc("distances")] -= \
    df.iloc[heads, df.columns.get_loc("distances")].to_numpy()
  del df["group"]

  # Add connection of last nodes in group to second node on their nearest edges
  end_traces = sort_df["cell_id"].tail(1).to_numpy()
  end_nodes = second_nodes[end_traces]
  distances = np.sqrt(np.sum(
    np.square(centroids[end_nodes] - projected_points[end_traces]), axis=1))
  end_nodes += len(matrix)
  df = pd.concat((df,
                  pd.DataFrame({
                    "cell_id": end_traces,
                    "root": end_nodes,
                    "distances": distances,
                  })
  ))

  # Add connection of minimum spanning tree of the principal graph.
  principal_edges = mst.nonzero()
  df = pd.concat((df,
                  pd.DataFrame({
                    "cell_id": principal_edges[0] + len(matrix),
                    "root": principal_edges[1] + len(matrix),
                    "distances": mst.data,
                  })
  ))

  graph = nx.Graph()
  graph.add_weighted_edges_from(df.to_numpy())

  return nx.minimum_spanning_tree(graph)


def compute_cell_states(
    matrix: np.ndarray,
    centroids: np.ndarray,
    mst: sparse.csr_matrix,
):
  """Find states of cell.
  A cellular state is a progression that proceeds in a specific direction.

  Parameters
  ----------
  matrix : ``ndarray``
    The expression matrix, recommended UMAP embeddings.
  centroids : ``ndarray``
    The centroids of the principal graph.
  mst : ``csr_matrix``
    The symmetrical minimum spanning tree of the principal graph.

  Returns
  -------
  cell_states : ``ndarray``
    The states of cell.
  branching_nodes: ``ndarray``:
    Root nodes of cellular states.
  """
  nearest_edges = find_nearest_principal_edges(
    matrix, centroids, mst
  )
  first_nodes = np.minimum(*nearest_edges)
  second_nodes = np.maximum(*nearest_edges)
  edge_hash = first_nodes * len(centroids) + second_nodes

  states, indices = np.unique(edge_hash, return_inverse=True)

  graph = nx.from_scipy_sparse_array(mst)
  branching_nodes = [i for i in range(mst.shape[0]) if graph.degree[i] > 2]
  dfs = dict(nx.dfs_predecessors(graph, branching_nodes[0]))

  children = np.array(list(dfs.keys()))
  parents = np.array(list(dfs.values()))
  indptr = np.where(np.isin(parents, branching_nodes))[0]
  indptr = np.concatenate((indptr, [len(parents)]))

  group_states = []
  for i in range(len(indptr) - 1):
    end =  indptr[i + 1]
    end += (i + 2) < len(indptr) and \
      parents[indptr[i]] != parents[indptr[i + 1]]
    group_states.append(parents[indptr[i]:end])

  group_states = []
  visited_parents = [parents[indptr[0]]]
  for i in range(len(indptr) - 1):
    curr_states = parents[indptr[i]:indptr[i + 1]]
    if parents[indptr[i]] not in visited_parents:
      curr_states = np.concatenate(
        (curr_states, [parents[indptr[i + 1]]]))
    else:
      curr_states = np.concatenate(
        (curr_states, [children[indptr[i + 1] - 1]]))
    group_states.append(curr_states)
    if (i + 2) < len(indptr):
      visited_parents.append(parents[indptr[i + 1]])

  states_mapping = {}
  for i, group in enumerate(group_states):
    first_nodes = np.minimum(group[:-1], group[1:])
    second_nodes = np.maximum(group[:-1], group[1:])
    edge_hash = first_nodes * len(centroids) + second_nodes
    states_mapping.update({eg_hash: i for eg_hash in edge_hash})
  states = np.array([states_mapping[state] for state in states])

  return np.array(states[indices]), np.array(branching_nodes)
