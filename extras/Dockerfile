FROM bioconductor/bioconductor_docker:3.17 as bioconductor
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as cuda

COPY --from=bioconductor / /


RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y zsh
RUN apt-get install -y vim
RUN apt-get install -y htop
RUN apt-get install -y tmux
RUN apt-get install -y cmake
RUN apt-get install -y python3
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y build-essential
RUN apt-get install -y ccache
RUN apt-get install -y gcc-9
RUN apt-get install -y g++-9
RUN apt-get install -y default-jre 
RUN apt-get install -y libomp-13-dev

RUN pip install numpy scipy scikit-learn networkx umap-learn squidpy statsmodels
RUN pip install boto3 botocore h5py pandas scanpy igraph pytest psutil openpyxl


RUN R -e "BiocManager::install('BiocGenerics')"
RUN R -e "BiocManager::install('DelayedArray')"
RUN R -e "BiocManager::install('DelayedMatrixStats')"
RUN R -e "BiocManager::install('limma')"
RUN R -e "BiocManager::install('lme4')"
RUN R -e "BiocManager::install('S4Vectors')"
RUN R -e "BiocManager::install('SingleCellExperiment')"
RUN R -e "BiocManager::install('SummarizedExperiment')"
RUN R -e "BiocManager::install('batchelor')"
RUN R -e "BiocManager::install('HDF5Array')"
RUN R -e "BiocManager::install('terra')"
RUN R -e "BiocManager::install('ggrastr')"
RUN R -e "devtools::install_github('cole-trapnell-lab/monocle3');"
RUN R -e "library(monocle3);"


RUN yes | unminimize

ARG ROOT_PASSWORD

RUN apt-get install -y openssh-server &&\
    mkdir /var/run/sshd &&\
    echo "root:${ROOT_PASSWORD}" | chpasswd &&\
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
