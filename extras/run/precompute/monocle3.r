suppressMessages(library(rhdf5))
suppressMessages(library(monocle3))


args <- commandArgs(trailingOnly=TRUE)
data.path <- args[1]

# Read matrix
barcodes <- h5read(data.path, "barcodes")
features <- h5read(data.path, "features")
data <- h5read(data.path, "data")
indices <- h5read(data.path, "indices")
indptr <- h5read(data.path, "indptr")

expr.mtx <- Matrix::sparseMatrix(
  i=indices, p=indptr, x=as.numeric(data),
  dimnames=list(as.character(features), as.character(barcodes)),
  index1=FALSE, repr="C")

# Read parameters
umap <- t(h5read(data.path, "umap"))
louvain <- h5read(data.path, "louvain")

# Create cell_data_set
cds <- new_cell_data_set(expr.mtx)
rownames(cds@colData) <- barcodes

# UMAP
rownames(umap) <- barcodes
colnames(umap) <- c("umap_1", "umap_2")
cds@int_colData@listData$reducedDims@listData[["UMAP"]] <- umap

# Use 1 partition for all cells
recreate.partitions <- c(rep(1, length(cds@colData@rownames)))
names(recreate.partitions) <- cds@colData@rownames
recreate.partitions <- as.factor(recreate.partitions)
cds@clusters@listData[["UMAP"]][["partitions"]] <- recreate.partitions

# Clusters
names(louvain) <- cds@colData@rownames
louvain <- as.factor(louvain)
cds@clusters@listData[["UMAP"]][["clusters"]] <- louvain

# k-means clustering
num_clusters_in_partition <- length(unique(louvain))
X <- t(umap)
num_cells_in_partition <- ncol(X)
curr_ncenter <- monocle3:::cal_ncenter(
  num_clusters_in_partition, num_cells_in_partition)
centers <- t(X)[seq(1, ncol(X), length.out=curr_ncenter), , drop = FALSE]
centers <- centers + matrix(stats::rnorm(length(centers), sd = 1e-10),
                            nrow = nrow(centers))
kmean_res <- tryCatch({
  stats::kmeans(t(X), centers=centers, iter.max = 100)
}, error = function(err) {
  stats::kmeans(t(X), centers = curr_ncenter, iter.max = 100)
})

# Learn graph
cds <- learn_graph(cds)

# Order cells
cds <- order_cells(cds, root_cells=barcodes[length(barcodes) %/% 2 + 1])

# Write results
principal_graph_aux <- cds@principal_graph_aux@listData$UMAP

h5createGroup(data.path,"monocle3")
h5write(pseudotime(cds), file=data.path, name="monocle3/pseudotime")
h5write(
  principal_graph_aux$pr_graph_cell_proj_dist,
  file=data.path, name="monocle3/projected_points"
)
h5write(
  principal_graph_aux$dp_mst,
  file=data.path, name="monocle3/centroids"
)
h5write(
  principal_graph_aux$root_pr_nodes,
  file=data.path, name="monocle3/root_pr_nodes"
)
h5write(
  kmean_res$cluster,
  file=data.path, name="monocle3/k_means_clustering"
)

principal_graph <- principal_graph_aux$stree
h5createGroup(data.path,"monocle3/principal_graph")
h5write(
  principal_graph@x, file=data.path,
  name="monocle3/principal_graph/data"
)
h5write(
  principal_graph@i, file=data.path,
  name="monocle3/principal_graph/indices"
)
h5write(
  principal_graph@p, file=data.path,
  name="monocle3/principal_graph/indptr"
)
h5write(
  principal_graph@Dim, file=data.path,
  name="monocle3/principal_graph/shape"
)
