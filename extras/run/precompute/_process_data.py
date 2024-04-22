import os
import h5py
import pandas as pd
import numpy as np
from scipy import sparse
import scanpy as sc
import multiprocessing as mp

from . import _constants as constants


os.environ["OMP_NUM_THREADS"] = "16"
SPAWN_CTX = mp.get_context("spawn")
STDOUT = 1
STDERR = 2


def _subprocess(data_id: str, data_path: str):
  log_out = os.path.join(constants.LOGS_FOLDER, f"{data_id}.stdout")
  log_err = os.path.join(constants.LOGS_FOLDER, f"{data_id}.stderr")
  fdout = os.open(log_out, os.O_WRONLY | os.O_CREAT)
  os.dup2(fdout, STDOUT)
  fderr = os.open(log_err, os.O_WRONLY | os.O_CREAT)
  os.dup2(fderr, STDERR)
  os.execl("/usr/local/bin/Rscript", "/usr/local/bin/Rscript",
           constants.MONOCLE_SCRIPT, data_path)


def process(data_id: str):
  data_path = os.path.join(constants.DATA_FOLDER, f"{data_id}.hdf5")
  with h5py.File(data_path, "r") as f:
    barcodes = [bc.decode() for bc in f["barcodes"]]
    features = [ft.decode() for ft in f["features"]]
    data = f["data"][()]
    indices = f["indices"][()]
    indptr = f["indptr"][()]
  adata = sc.AnnData(
      X=sparse.csr_matrix((data, indices, indptr),
                          shape=(len(barcodes), len(features))),
      obs=pd.DataFrame({}, index=barcodes),
      var=pd.DataFrame({}, index=features),
  )
  sc.pp.filter_genes(adata, min_cells=10)
  sc.pp.normalize_total(adata)
  sc.pp.log1p(adata)
  sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
  sc.pp.pca(adata, n_comps=50)
  sc.pp.neighbors(adata, n_neighbors=30)
  sc.tl.louvain(adata, flavor="igraph")
  sc.tl.umap(adata)

  with h5py.File(data_path, "r+") as f:
    f.create_dataset("umap", data=adata.obsm["X_umap"])
    f.create_dataset("louvain", data=np.array(adata.obs["louvain"]))

  proc = SPAWN_CTX.Process(
      target=_subprocess,
      args=(data_id, data_path),
  )
  proc.start()
  proc.join()
  proc.close()
