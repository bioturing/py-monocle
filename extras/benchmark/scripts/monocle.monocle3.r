suppressMessages(library(Matrix))
suppressMessages(library(rhdf5))
suppressMessages(library(monocle3))


args <- commandArgs(trailingOnly=TRUE)
data.path <- args[1]
result.path <- args[2]

# Read parameters
barcodes <- h5read(data.path, "barcodes")
umap <- t(h5read(data.path, "umap"))
louvain <- h5read(data.path, "louvain")
rownames(umap) <- as.list(barcodes)
colnames(umap) <- c("umap_1", "umap_2")

# Create cell_data_set
cds <- new_cell_data_set(t(umap))
rownames(cds@colData) <- as.list(barcodes)

# UMAP
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
cds <- learn_graph(cds, close_loop=FALSE)

# Order cells
cds <- order_cells(cds, root_cells=barcodes[1])

exec.time <- difftime(Sys.time(), start.time, units = "secs")
write.csv(
  data.frame(exec_time = exec.time),
  file = result.path,
  row.names = FALSE)
