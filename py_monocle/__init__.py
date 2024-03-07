from typing import Optional, Union
import numpy as np

from . import _utils as utils
from ._learn_graph import learn_graph
from ._order_cells import order_cells


def pseudotime(
    matrix: np.ndarray,
    root_cells: Optional[Union[int, np.ndarray]] = None,
    root_pr_cells: Optional[Union[int, np.ndarray]] = None,
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
  Compute pseudotime for the given expression matrix.

  Parameters
  ----------
  matrix : ``ndarray``
    The expression matrix, recommended UMAP embeddings.
  root_cells : ``Optional[Union[int, np.ndarray]]``, default: None
    (List) index of the expression matrix.
  root_pr_cells : ``Optional[Union[int, np.ndarray]]``, default: None
    (List) index of the centroids of principal graph.
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
  pseudotime: ``ndarray``
    Pseudotime values for each cell.
  """
  projected_points, mst, centroids = learn_graph(
    matrix=matrix,
    n_centroids=n_centroids,
    clusters=clusters,
    partitions=partitions,
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

  return order_cells(
    matrix,
    centroids,
    mst=mst,
    projected_points=projected_points,
    root_cells=root_cells,
    root_pr_cells=root_pr_cells,
  )
