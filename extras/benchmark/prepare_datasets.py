import os
import sys
import h5py
import boto3
import botocore
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from pathlib import Path
import warnings


AWS_S3_REGION = "us-west-2"
LUIN_BENCHMARK_DATA_BUCKET = "luin-benchmark-data"
CONFIG = botocore.config.Config(signature_version=botocore.UNSIGNED)

DATASETS_PREFIX = "datasets"
MATRIX_FILENAME = "matrix.hdf5"
__DIRNAME__ = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")


def _connect_to_s3_bucket():
  resource = boto3.resource("s3", region_name=AWS_S3_REGION, config=CONFIG)
  bucket = resource.Bucket(LUIN_BENCHMARK_DATA_BUCKET)
  return resource, bucket


def _is_interactive_shell():
  return os.isatty(sys.stdout.fileno())


def _download_file(s3_path: str, dst_path: Path, progress: bool = True):
  resource, bucket = _connect_to_s3_bucket()
  total_size = resource.ObjectSummary(
      bucket_name=bucket.name, key=s3_path,
  ).size

  if _is_interactive_shell() and progress:
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    def _callback(n_bytes: int):
      progress_bar.update(n_bytes)

    bucket.download_file(Key=s3_path, Filename=dst_path, Callback=_callback)
  else:
    bucket.download_file(Key=s3_path, Filename=dst_path)

  resource.meta.client.close()
  return dst_path


def _list_available_datasets():
  resource, bucket = _connect_to_s3_bucket()
  objects = [
      {"path": o.key, "etag": o.e_tag, "size": o.size}
      for o in bucket.objects.filter(Prefix=DATASETS_PREFIX)
  ]
  resource.meta.client.close()
  objects.sort(key=lambda x: x["size"])
  data_ids = [
      obj["path"].split("/")[1]
      for obj in objects
      if obj["path"].split("/")[2] == "matrix.hdf5" and obj["size"] < 20 * 2**30
  ]
  return data_ids


def _download_dataset(data_id: str, dst_path: str):
  s3_path = f"{DATASETS_PREFIX}/{data_id}"
  matrix_path = f"{s3_path}/matrix.hdf5"

  dst_matrix = os.path.join(dst_path, MATRIX_FILENAME)
  if not os.path.exists(dst_matrix):
    _download_file(matrix_path, dst_matrix, progress=True)


def _read_dataset(dataset_path: Path):
  with h5py.File(os.path.join(dataset_path, "matrix.hdf5"), "r") as f:
    data = f["data"][()].astype("float32")
    indices = f["indices"][()]
    indptr = f["indptr"][()]
    features = [feature.decode() for feature in f["features"][()]]
    barcodes = [barcode.decode() for barcode in f["barcodes"][()]]
  csr_mtx = sparse.csr_matrix(
      (data, indices, indptr),
      shape=(len(barcodes), len(features)),
  )
  return csr_mtx, barcodes, features


def _process_dataset(
    dataset_path: Path,
    csr_matrix: sparse.csr_matrix,
    device: bool = False,
  ):
  if device:
    from bioalpha import sc
  else:
    import scanpy as sc

  adata = sc.AnnData(X=csr_matrix)
  sc.pp.filter_genes(adata, min_cells=10)
  if device:
    sc.pp.log_normalize(adata)
  else:
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
  sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
  sc.pp.pca(adata, n_comps=50)
  sc.pp.neighbors(adata, n_neighbors=30)
  sc.tl.louvain(adata, flavor="igraph")
  sc.tl.umap(adata)

  with h5py.File(os.path.join(dataset_path, "matrix.hdf5"), "r+") as f:
    f.create_dataset("umap", data=adata.obsm["X_umap"])
    f.create_dataset("louvain", data=adata.obs["louvain"].to_numpy())


def main():
  datasets_path = os.path.join(__DIRNAME__, "datasets")
  os.makedirs(datasets_path, exist_ok=True)

  data_ids = _list_available_datasets()
  for i, data_id in enumerate(data_ids):
    print(f"[{i + 1:2d}/{len(data_ids)}] Downloading `{data_id}`")
    in_dataset_path = os.path.join(datasets_path, data_id)
    os.makedirs(in_dataset_path, exist_ok=True)
    _download_dataset(data_id, in_dataset_path)

  nnz, n_genes, n_cells = [], [], []

  for data_id in tqdm(data_ids):
    in_dataset_path = os.path.join(datasets_path, data_id)
    csr_mtx, barcodes, features = _read_dataset(in_dataset_path)
    _process_dataset(in_dataset_path, csr_mtx)
    nnz.append(csr_mtx.indptr[-1])
    n_cells.append(len(barcodes))
    n_genes.append(len(features))

  datasets = pd.DataFrame(
      {"n_cells": n_cells, "n_genes": n_genes, "nnz": nnz}, index=data_ids,
  )
  datasets = datasets.sort_values("n_cells")
  datasets_info_path = os.path.join(datasets_path, "datasets.csv")
  datasets.to_csv(datasets_info_path)


if __name__ == "__main__":
  main()
