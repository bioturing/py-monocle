from typing import Optional
import numpy as np
from scipy import sparse
from scipy import stats
from statsmodels.regression.linear_model import GLS


def __diff_test(
    matrix: sparse.csc_matrix,
    pseudotime: np.ndarray,
):
  pvalues = []
  n_cells, n_genes = matrix.shape
  exp = np.zeros(n_cells).astype(np.float32)
  for i in range(n_genes):
    exp.fill(0)
    se = slice(matrix.indptr[i], matrix.indptr[i + 1])
    exp[matrix.indices[se]] = matrix.data[se]
    if np.sum(exp) == 0:
      pvalues.append(0)
      continue
    model = GLS(
      pseudotime,
      np.nan_to_num(stats.zscore(exp))
    ).fit()
    pvalues.append(model.pvalues[0])
  pvalues = np.array(pvalues).astype(np.float32)
  return pvalues


def regression_analysis(
    expression_matrix: sparse.spmatrix,
    pseudotime: np.ndarray,
    cell_states: Optional[np.ndarray] = None,
):
  """Regression analysis between genes expression and pseudotime.

  Parameters
  ----------
  expression_matrix: ``spmatrix``
    Expression matrix to query genes.
  pseudotime: ``ndarray``
    Pseudotime values for each cell.
  cell_states: ``Optional[np.ndarray]``, default: None
    The states of cell. If not None, analysis for each states.

  Returns
  -------
  analysis_result: ``ndarray``
    p-values of regression.
  """
  if not isinstance(expression_matrix, sparse.csc_matrix):
    expression_matrix = expression_matrix.tocsc()

  if cell_states is None:
    return __diff_test(expression_matrix, pseudotime)

  pvals_matrix = []
  for i in np.unique(cell_states):
    pvals_matrix.append(__diff_test(
      expression_matrix[cell_states==i],
      pseudotime[cell_states==i]
    ))
  return np.array(pvals_matrix, dtype=np.float32)
