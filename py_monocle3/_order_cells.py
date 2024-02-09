import numpy as np
from typing import Union, Optional
import numpy as np
from scipy.sparse.csgraph import dijkstra

from ._utils import minimum_spanning_tree, find_nearest_principal_nodes


def order_cells(
    matrix: np.ndarray,
    projected_points: np.ndarray,
    centroids: np.ndarray,
    root_cells: Optional[Union[int, np.ndarray]] = None,
    root_pr_cells: Optional[Union[int, np.ndarray]] = None,
):
  """Assign pseudotime to individual projected points based on their distance
  to a root point in the MST.

  Parameters
  ----------
  matrix : ``ndarray``
    The expression matrix, recommended UMAP embeddings.
  projected_points : ``ndarray``
    Projected points of matrix onto the principal graph.
  centroids : ``ndarray``
    The centroids of the principal graph.
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

  mst = minimum_spanning_tree(projected_points)
  pseudotime = dijkstra(mst, directed=False, indices=root_cells)
  return np.min(pseudotime, axis=0)
