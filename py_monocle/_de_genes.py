from typing import Optional, Callable, List
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
from anndata import AnnData
from statsmodels.stats.multitest import fdrcorrection


__KNN_TYPE__ = Callable[[np.ndarray, int], List[np.ndarray]]
__MORANS_I_TYPE__ = Callable[[
    sparse.csr_matrix, sparse.csr_matrix], List[np.ndarray]]


def differential_expression_genes(
    expression_matrix: sparse.spmatrix,
    projected_cells: np.ndarray,
    n_neighbors: int = 30,
    knn_func: Optional[__KNN_TYPE__] = None,
    alpha: float = 0.05,
    morans_I_func: Optional[__MORANS_I_TYPE__] = None,
    morans_I_kwargs: dict = {},
) -> List[np.ndarray]:
  """Identifying genes with complex trajectory-dependence expression.

  Parameters
  ----------
  expression_matrix: ``spmatrix``
    Expression matrix to query genes.
  projected_cells : ``Optional[ndarray]``, default: None
    Projected of cells to the principal graph.
  n_neighbors: ``int``, default: 30
    Number of neighbors for creating weight matrix W.
  knn_func: ``Optional[Callable]``, default: None
    Function to compute neighbors for creating weight matrix W.
    If None, using KDTree.
  alpha: ``float``, default: 0.05
    Family-wise error rate.
  morans_I_func: ``Optional[Callable]``, default: None
    Function to compute Moran's I scores.
    If None, using ``squidpy.gr.spatial_autocorr``
  morans_I_kwargs: ``dict``, default: Empty
    Kwargs for Moran's I scores.
  principal_graph:
    The symmetrical minimum spanning tree of the principal graph.
    If not None, using to cut off connections of neighbors graph.

  Returns
  -------
  morans_i_scores: ``ndarray``
    Moran's I scores for each gene.
  p_vals: ``ndarray``
    p-value under normality assumption.
  adjusted_pvalues: ``ndarray``
    The corrected p-values.
  """
  if knn_func is not None:
    knn_distances, knn_indices = knn_func(projected_cells, n_neighbors)
  else:
    kdtree = KDTree(projected_cells)
    knn_distances, knn_indices = kdtree.query(projected_cells, n_neighbors)

  n_cells = projected_cells.shape[0]
  indptr = np.arange(0, knn_indices.size + 1, knn_indices.shape[1])
  distances_matrix = sparse.csr_matrix(
      (knn_distances.flatten(), knn_indices.flatten(), indptr),
      shape=(n_cells, n_cells), dtype=np.float64,
  )

  if morans_I_func is not None:
    morans_i_scores, p_vals = morans_I_func(
        expression_matrix, distances_matrix, **morans_I_kwargs)
  else:
    from squidpy.gr import spatial_autocorr

    adata = AnnData(X=expression_matrix)
    adata.obsp["connectivities"] = distances_matrix
    spatial_autocorr(
        adata, connectivity_key="connectivities", **morans_I_kwargs)
    moran_I_res = adata.uns["moranI"].loc[adata.var_names]
    morans_i_scores = moran_I_res["I"].values.copy()
    p_vals = moran_I_res["pval_norm"].values.copy()
  _, adjusted_pvalues = fdrcorrection(p_vals, alpha=alpha)

  return morans_i_scores, p_vals, adjusted_pvalues
