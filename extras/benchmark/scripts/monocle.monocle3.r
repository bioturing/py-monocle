suppressMessages(library(Matrix))
suppressMessages(library(rhdf5))
suppressMessages(library(monocle3))


args <- commandArgs(trailingOnly=TRUE)
data.path <- args[1]
result.path <- args[2]

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
rownames(cds@colData) <- as.list(barcodes)

# UMAP
rownames(umap) <- as.list(barcodes)
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

start.time <- Sys.time()

# Learn graph
cds <- learn_graph(cds)

# Order cells
cds <- order_cells(cds, root_cells=barcodes[1])

exec.time <- difftime(Sys.time(), start.time, units = "secs")
write.csv(
  data.frame(exec_time = exec.time),
  file = result.path,
  row.names = FALSE)
