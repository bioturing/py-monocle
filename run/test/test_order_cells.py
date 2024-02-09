import os
import pytest
import h5py
import numpy as np
from py_monocle3 import order_cells

from . import _constants as constants


@pytest.mark.parametrize("data_id", constants.DATASET)
def test_order_cells(data_id: str):
  with h5py.File(os.path.join(constants.DATA_FOLDER, f"{data_id}.hdf5")) as f:
    centroids = f["monocle3/centroids"][()]
    projected_points = f["monocle3/projected_points"][()]
    expected_pseudotime = f["monocle3/pseudotime"][()]
    matrix = f["umap"][()]
  pseudotime = order_cells(
    matrix, projected_points, centroids, root_cells=0)

  diffs = np.max(abs(pseudotime - expected_pseudotime) / expected_pseudotime)
  max_diff = np.max(np.nan_to_num(diffs))
  assert max_diff < 0.05, \
    f"Max ratio difference is {max_diff / np.max(expected_pseudotime)}."
