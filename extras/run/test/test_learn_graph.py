import os
import pytest
import h5py
import numpy as np
import py_monocle
from scipy import sparse

from . import _constants as constants


@pytest.mark.parametrize("data_id", constants.DATASET)
def test_learn_graph(data_id: str):
  data_path = os.path.join(constants.DATA_FOLDER, f"{data_id}.hdf5")
  with h5py.File(data_path, "r") as f:
    matrix = f["umap"][()]
    clusters = f["monocle3/k_means_clustering"][()]
    truth_projected_points = f["monocle3/projected_points"][()]

    data = f["data"][()]
    indices = f["indices"][()]
    indptr = f["indptr"][()]
    n_cells = len(f["barcodes"])
    n_genes = len(f["features"])
    expression_matrix = sparse.csr_matrix(
        (data, indices, indptr), shape=(n_cells, n_genes))

  projected_points, _, _ = py_monocle.learn_graph(
      matrix=matrix,
      clusters=clusters,
      use_clusters_as_kmeans=True,
  )

  _, result_p_vals, _ = py_monocle.differential_expression_genes(
      expression_matrix, projected_points
  )
  result_genes = np.where(result_p_vals < 0.01)[0]

  _, truth_p_vals, _ = py_monocle.differential_expression_genes(
      expression_matrix, truth_projected_points
  )
  truth_genes = np.where(truth_p_vals < 0.01)[0]

  score = len(np.intersect1d(result_genes, truth_genes)) / len(truth_genes)
  assert score > 0.9, f"Only intersect {score} highly variable genes"
