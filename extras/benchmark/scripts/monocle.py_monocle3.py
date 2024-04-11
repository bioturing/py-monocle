import sys
import time
import h5py
import pandas as pd
from py_monocle import pseudotime


if __name__ == "__main__":
  data_path = sys.argv[1]
  result_path = sys.argv[2]

  with h5py.File(data_path, "r") as f:
    matrix = f["umap"][()]
    louvain = f["louvain"][()]

  start_time = time.time()
  pseudotime(
      matrix=matrix, root_cells=0, clusters=louvain, form_circle=False)
  exec_time = time.time() - start_time

  res_df = pd.DataFrame({"exec_time": [exec_time]})
  res_df.to_csv(result_path, index=False)
