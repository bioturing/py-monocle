from typing import Union, Optional
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx

from ._utils import find_nearest_principal_nodes, find_nearest_principal_edges


def order_cells(
    matrix: np.ndarray,
    projected_points: np.ndarray,
    centroids: np.ndarray,
    mst: sparse.csr_matrix,
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
  mst : ``csr_matrix``
    The symmetrical minimum spanning tree of the principal graph.
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

  # Find connection of each nearest edge
  nearest_edges = find_nearest_principal_edges(
    matrix, centroids, mst
  )
  first_nodes = np.minimum(*nearest_edges)
  second_nodes = np.maximum(*nearest_edges)
  edge_group = first_nodes * len(centroids) + second_nodes
  distances = np.sqrt(np.sum(
    np.square(centroids[first_nodes] - projected_points), axis=1))
  df = pd.DataFrame({
    "cell_id": np.arange(len(matrix)),
    "root": first_nodes + len(matrix),
    "group": edge_group,
    "distances": distances,
  })
  sort_df = df.sort_values(["group", "distances"]).groupby("group")
  heads = sort_df["cell_id"].head(-1)
  tails = sort_df["cell_id"].tail(-1)
  df.iloc[tails, df.columns.get_loc("root")] = heads
  df.iloc[tails, df.columns.get_loc("distances")] -= \
    df.iloc[heads, df.columns.get_loc("distances")].to_numpy()
  del df["group"]

  # Add connection of last nodes in group to second node on their nearest edges
  end_traces = sort_df["cell_id"].tail(1).to_numpy()
  end_nodes = second_nodes[end_traces]
  distances = np.sqrt(np.sum(
    np.square(centroids[end_nodes] - projected_points[end_traces]), axis=1))
  end_nodes += len(matrix)
  df = pd.concat((df,
                  pd.DataFrame({
                    "cell_id": end_traces,
                    "root": end_nodes,
                    "distances": distances,
                  })
  ))

  # Add connection of minimum spanning tree of the principal graph.
  principal_edges = mst.nonzero()
  df = pd.concat((df,
                  pd.DataFrame({
                    "cell_id": principal_edges[0] + len(matrix),
                    "root": principal_edges[1] + len(matrix),
                    "distances": mst.data,
                  })
  ))

  graph = nx.Graph()
  graph.add_weighted_edges_from(df.to_numpy())

  pseudotime = nx.multi_source_dijkstra_path_length(
    graph, sources=set(root_cells), weight="weight")
  return np.array([pseudotime.get(i, np.inf) for i in range(len(matrix))])
