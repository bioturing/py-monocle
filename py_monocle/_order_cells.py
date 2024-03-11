from typing import Union, Optional
import numpy as np
from scipy import sparse
import networkx as nx

from ._utils import find_nearest_principal_nodes, create_projected_graph


def order_cells(
    matrix: np.ndarray,
    centroids: np.ndarray,
    *,
    mst: Optional[sparse.csr_matrix] = None,
    projected_points: Optional[np.ndarray] = None,
    projected_graph: Optional[nx.Graph] = None,
    root_cells: Optional[Union[int, np.ndarray]] = None,
    root_pr_cells: Optional[Union[int, np.ndarray]] = None,
):
  """Assign pseudotime to individual projected points based on their distance
  to a root point in the MST.

  Parameters
  ----------
  matrix : ``ndarray``
    The expression matrix, recommended UMAP embeddings.
  centroids : ``ndarray``
    The centroids of the principal graph.
  mst : ``Optional[csr_matrix]``, default: None
    The symmetrical minimum spanning tree of the principal graph.
  projected_points : ``Optional[ndarray]``, default: None
    Projected points of matrix onto the principal graph.
  projected_graph : ``Optional[nx.Graph]``, default: None
    The minimum spanning tree networkx.Graph of projected points.
    Centroids of projected_graph is a vertical stack of matrix and centroids.
  root_cells : ``Optional[Union[int, np.ndarray]]``, default: None
    (List) index of the expression matrix.
  root_pr_cells : ``Optional[Union[int, np.ndarray]]``, default: None
    (List) index of the centroids of principal graph.

  Returns
  -------
  pseudotime: ``ndarray``
    Pseudotime values for each cell.
  """
  assert (root_cells is not None) ^ (root_pr_cells is not None), \
    "Request one and only one of root_cells or root_pr_cells is not None."
  if root_cells is not None:
    assert matrix is not None, "Request expression matrix for root_cells."
    root_cells = matrix[root_cells].reshape((-1, 2))
    root_pr_cells = find_nearest_principal_nodes(root_cells, centroids)
  root_cells = find_nearest_principal_nodes(
    centroids[root_pr_cells].reshape((-1, 2)), matrix
  )

  if projected_graph is None:
    projected_graph = create_projected_graph(
      matrix, projected_points, centroids, mst)

  pseudotime = nx.multi_source_dijkstra_path_length(
    projected_graph, sources=set(root_cells), weight="weight")
  return np.array([pseudotime.get(i, np.inf) for i in range(len(matrix))])
