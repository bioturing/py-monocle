from typing import Optional
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans
import networkx as nx

from ._utils import (
  minimum_spanning_tree,
  find_nearest_principal_edges,
  pairwise_distance,
  compute_cluster_connectivities,
)


def prune_tree(
    mst: sparse.csr_matrix, threshold: int = 10,
) -> np.ndarray:
  """
  DFS the tree and prune the branches that have diameter smaller than threshold.

  Parameters
  ----------
  mst: ``csr_matrix``    
    The symmetrize minimum spanning tree represented as a sparse matrix.
  threshold: ``int``     
    The minimal length of the diameter path for a branch to be preserved
    during graph pruning procedure.

  Returns
  -------
  centroid_indices: ``ndarray``
    list of remaining indices of centroids
  """
  branch_nodes = np.where(np.diff(mst.indptr) == 2)[0]
  if len(branch_nodes) == 0:
    return np.arange(mst.shape[0], dtype=int)
  root_node = branch_nodes[0]

  graph: nx.Graph = nx.from_scipy_sparse_array(mst)
  dfs = np.array(list(nx.dfs_preorder_nodes(graph, source=root_node)))
  visited = np.zeros(mst.shape[0], dtype=bool)

  for node in dfs:
    if node not in graph.nodes:
      continue
    if graph.degree[node] < 3:
      continue
    neighbors = [n for n in graph.neighbors(node) if not visited[n]]
    indices = np.where(np.isin(dfs, neighbors))[0]
    neighbors = dfs[indices[1:3]]
    for nb in neighbors:
      graph.remove_edge(node, nb)
      component = list(nx.dfs_preorder_nodes(graph, nb, depth_limit=threshold))
      if nx.diameter(graph.subgraph(component)) + 1 < threshold:
        graph.remove_nodes_from(component)
      else:
        graph.add_edge(node, nb)

  return np.array(graph.nodes, dtype=int)


def connect_tip_cells(
  matrix: np.ndarray,
  mst: sparse.csr_matrix,
  centroids: np.ndarray,
  clusters: np.ndarray,
  qval_thresh: float = 0.05,
  euclidean_distance_ratio: float = 1,
  geodesic_distance_ratio: float = 1/3,
):
  """Find all tip cells and check if they fit criteria to add connections.

  Parameters
  ----------
  matrix: ``ndarray``
    Coordinates of the cells, recommended UMAP embeddings.
  mst: ``csr_matrix``
    The symmetrize minimum spanning tree.
  centroids: ``ndarray``
    The centroids of the k-means clustering.
  clusters: ``ndarray``
    k-means clustering.
  qval_thresh: ``float``, default: 0.05
    Threshold of variance of clusters.
  euclidean_distance_ratio: ``float``, default: 1
    The maximal ratio between the euclidean distance of two tip nodes in the
    spanning tree and the maximum distance between any connecting points on the
    spanning tree allowed to be connected during the loop closure procedure.
  geodesic_distance_ratio: ``float``, default: 1/3
    The minimal ratio between the geodesic distance of two tip nodes in the
    spanning tree and the length of the diameter path on the spanning tree
    allowed to be connected during the loop closure procedure.

  Returns
  -------
  mst: ``csr_matrix``
    Joined circle minimum spanning tree.
  joined_nodes: ``ndarray``
    Joined tip cells.
  """
  longest_edge = np.max(mst.data)
  mst.data[:] = 1
  distance = dijkstra(mst, directed=False)
  diameter = np.max(distance)
  degrees = np.diff(mst.indptr)
  leaves = np.where(degrees == 1)[0]

  cluster_mat = compute_cluster_connectivities(matrix=matrix, clusters=clusters)
  cluster_mat_leaf = cluster_mat[leaves][:, leaves]
  np.fill_diagonal(cluster_mat_leaf, qval_thresh + 1) # ignore connections of leaves to itself
  valid_connection = np.where(cluster_mat_leaf < qval_thresh)
  joined_indices = []

  for i, j in zip(*valid_connection):
    if i == j:
      continue
    leaf_1 = leaves[i]
    leaf_2 = leaves[j]
    if euclidean_distance_ratio * longest_edge > centroids[leaf_1] @ centroids[leaf_2].T and \
      geodesic_distance_ratio * diameter <= distance[leaf_1][leaf_2]:
      joined_indices.append([leaf_1, leaf_2])
  joined_indices = np.array(joined_indices, dtype=int)
  if len(joined_indices) > 0:
    mst[joined_indices[:, 0], joined_indices[:, 1]] = 1

  return mst, joined_indices


def calculate_projection(
  matrix: np.ndarray,
  roots: np.ndarray,
  leaves: np.ndarray,
) -> np.ndarray:
  """
  Calculate the projection of a cells onto the nearest edges of principal graph.

  Parameters
  ----------
  matrix: ``ndarray``
    Coordinates of the cells, recommended UMAP embeddings.
  roots: ``ndarray``
    Coordinates nearest centroids on principal graph.
  leaves: ``ndarray``
    Coordinates of another node of nearest edges.
  Returns
  -------
  projected_points: ``ndarray`` 
    Coordinate of the projection of the cells onto the nearest edges.
  """
  edge_length = np.linalg.norm(leaves - roots, axis=1) ** 2
  proj_factor = np.sum((matrix - roots) * (leaves - roots), axis=1) / edge_length
  matrix = roots + proj_factor.reshape((-1, 1)) * (leaves - roots)
  matrix[proj_factor < 0] = roots[proj_factor < 0]
  matrix[proj_factor > 1] = leaves[proj_factor > 1]
  return matrix


def project_cells(
    matrix: np.ndarray,
    centroids: np.ndarray,
    mst: sparse.csr_matrix,
):
  """Project cells onto principal components.

  Parameters:
  -----------
  matrix : ``ndarray``
    The matrix to find the nearest edge, recommended UMAP embeddings.
  centroids : ``ndarray``
    The centroids of the principal graph.
  mst : ``csr_matrix``
    The minimum spanning tree of the principal graph.

  Returns:
  --------
  projection_points: ``numpy array``
    Coordinates of projected cells onto principal graph.
  """
  nearest_nodes, second_vertices = find_nearest_principal_edges(
    matrix, centroids, mst
  )
  return calculate_projection(
    matrix, centroids[nearest_nodes], centroids[second_vertices])


def __calc_objective_centroids(
    matrix: np.ndarray,
    centroids: np.ndarray,
    sigma: float = 0.01,
):
  pdist = pairwise_distance(matrix, centroids)
  min_distances = np.min(pdist, axis=1)
  pdist -= min_distances.reshape((-1, 1))

  phi = np.exp(- pdist / sigma)
  sum_phi = np.sum(phi, axis=1)
  probs = phi / sum_phi.reshape((-1, 1))

  obj = - sigma * np.sum(np.log(sum_phi) - min_distances / sigma)

  return probs, obj


def __learn_graph(
    matrix: np.ndarray,
    *,
    n_centroids: Optional[int] = None,
    clusters: Optional[np.ndarray] = None,
    centroids_per_log: int = 15,
    prune: bool = True,
    p_threshold: int = 10,
    form_circle: bool = False,
    kmeans_seed: int = 0,
    max_iter: int = 10,
    k_nn: int = 25,
    sigma: float = 0.01,
    gamma: float = 0.5,
    eps: float = 1e-5,
    use_clusters_as_kmeans: bool = False,
):
  if use_clusters_as_kmeans:
    assert clusters is not None, \
      "Request clusters for option ``use_clusters_as_kmeans``"
    components, labels = np.unique(clusters, return_inverse=True)
    n_centroids = len(components)
  else:
    if n_centroids is None:
      n_cluster = len(np.unique(clusters)) if clusters is not None else 1
      n_centroids = int(n_cluster * centroids_per_log * np.log10(matrix.shape[0]))
    n_centroids = min(n_centroids, matrix.shape[0])

    # Init centroidsnit
    np.random.seed(kmeans_seed)
    init = matrix[np.linspace(0, matrix.shape[0] - 1, num=n_centroids, dtype=int)]
    init += 1e-10 * np.random.standard_normal(size=(n_centroids, 1))
    solver = KMeans(
      n_clusters=n_centroids,
      init=init, max_iter=100,
      random_state=kmeans_seed,
      n_init=1,
    )
    labels = solver.fit_predict(matrix)

  kdtree = KDTree(matrix)
  knn_dists, _ = kdtree.query(matrix, k_nn)
  rho = np.exp(- np.mean(knn_dists, axis=1)) + 1e-6
  indices = sparse.csr_matrix(
    (rho, labels, np.arange(len(rho)))
  ).argmax(axis=0)
  centroids = matrix[np.array(indices).flatten()]

  # Optimize centroids of principal graph
  objective_score = 0.001
  for _ in range(max_iter):
    mst = minimum_spanning_tree(centroids)
    sym_mst = mst.toarray()
    weight = np.sum(sym_mst)
    connected = sym_mst != 0

    probs, obj_centroid = __calc_objective_centroids(
      matrix, centroids, sigma=sigma)
    objective = weight + gamma * obj_centroid
    if abs(objective - objective_score) / objective_score < eps:
      break
    objective_score = objective

    Q = 2 * (np.diag(np.sum(connected, axis=0)) - connected) + \
      gamma * np.diag(np.sum(probs, axis=0))
    B = gamma * matrix.T @ probs
    centroids = (B @ np.linalg.inv(Q)).T

  # Create principal graph
  mst = minimum_spanning_tree(centroids)

  indices = np.arange(n_centroids)
  if prune:
    indices = prune_tree(mst, p_threshold)
  if form_circle:
    mst, joined_indices = connect_tip_cells(matrix, mst, centroids, labels)
    indices = np.unique(np.concatenate((indices, joined_indices)))

  mst = mst[indices][:, indices]
  centroids = centroids[indices]
  return mst, centroids


def learn_graph(
    matrix: np.ndarray,
    n_centroids: Optional[int] = None,
    clusters: Optional[np.ndarray] = None,
    partitions: Optional[np.ndarray] = None,
    centroids_per_log: int = 15,
    prune: bool = True,
    p_threshold: int = 10,
    form_circle: bool = False,
    kmeans_seed: int = 0,
    max_iter: int = 10,
    k_nn: int = 25,
    sigma: float = 0.01,
    gamma: float = 0.5,
    eps: float = 1e-5,
    use_clusters_as_kmeans: bool = False,
):
  """
  Learn the principal graph for the given expression matrix.

  Parameters
  ----------
  matrix : ``ndarray``
    The expression matrix, recommended UMAP embeddings.
  n_centroids: ``int``, optional, default: ``None``
    The number of centroids to use for the bisecting k-means.
    If ``None``, then the number of centroids is automatically calculated.
  centroids_per_log: ``int``, default: 15
    The number of centroids per each log10 of the number of cells.
  clusters: ``ndarray``, optional, default: ``None``
    The clusters of the cells. Used to calculate n_centroids.
  partitions: ``ndarray``
    The partition labels of the cells.
    If ``partitions`` is not None, learn principal graph on each partition.
  prune: ``bool``, default: ``True``
    If ``True``, prune the principal graph.
  p_threshold: ``int``, default: ``5``
    The threshold to use for pruning the MST.
  form_circle: ``bool``, default: ``False``
    If ``True``, join circle to principal graph.
  kmeans_seed: ``int``, default: 0
    Random state for kmeans clustering.
  max_iter: ``int``, default: 10
    Max iterations to learn graph.
  k_nn: ``int``, default: 25
    Number of nearest neighbors, use to init centroids of graph.
  sigma: ``float``, default: 0.01
    Bandwidth parameter.
  gamma: ``float``, default: 0.5
    Regularization parameter for k-means.
  eps: ``float``, default: 1e-5
    Relative objective difference.
  use_clusters_as_kmeans: ``bool``, default: ``False``
    If ``True``, use clusters as kmeans clustering.

  Returns
  -------
  projection_points: ``ndarray``
    The projection points of the given expression matrix onto the principal graph.
  mst: ``csr_matrix``
    the principal graph of the given expression matrix.
  centroids: ``ndarray``
    the centroids of the principal graph.
  """
  if partitions is not None:
    centroids = []
    mst = None
    for partition in np.unique(partitions):
      indices = [partitions == partition]
      sub_mst, sub_centroids = __learn_graph(
        matrix=matrix[indices],
        n_centroids=n_centroids,
        clusters=clusters[indices] if clusters is not None else None,
        centroids_per_log=centroids_per_log,
        prune=prune,
        p_threshold=p_threshold,
        form_circle=form_circle,
        kmeans_seed=kmeans_seed,
        max_iter=max_iter,
        k_nn=k_nn,
        sigma=sigma,
        gamma=gamma,
        eps=eps,
        use_clusters_as_kmeans=use_clusters_as_kmeans,
      )
      if partition == 0:
        mst = sub_mst
      else:
        mst = sparse.bmat([[mst, None], [None, sub_mst]])
      centroids.append(sub_centroids)
    centroids = np.concatenate(centroids, axis=0)
  else:
    mst, centroids = __learn_graph(
      matrix=matrix,
      n_centroids=n_centroids,
      clusters=clusters,
      centroids_per_log=centroids_per_log,
      prune=prune,
      p_threshold=p_threshold,
      form_circle=form_circle,
      kmeans_seed=kmeans_seed,
      max_iter=max_iter,
      k_nn=k_nn,
      sigma=sigma,
      gamma=gamma,
      eps=eps,
      use_clusters_as_kmeans=use_clusters_as_kmeans,
    )
  projected_points = project_cells(matrix, centroids, mst)
  return projected_points, mst, centroids
