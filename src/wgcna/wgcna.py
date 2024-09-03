
# Importing package modules
from . import common
from . import adjacency
from . import connectivity

# magic imputation
import magic
import numpy as np
import pandas as pd

import scanpy as sc
import anndata as ad

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import HDBSCAN
import scipy.cluster.hierarchy as sch

from dynamicTreeCut import cutreeHybrid

def run_wgcna(adata: ad.AnnData, adjacency_type: str = 'unsigned', cutoff: float = 0.1, min_cluster_size: int = 10, cluster_method: str = "hierarchical", cluster_height = 0.3):
    """_summary_

    Args:
        adata (_type_): _description_

    Returns:
        _type_: _description_
    """

    # calculate adjacency
    if adjacency_type == 'unsigned':
        corr = adjacency.unsigned_adjacency(adata)
    # calculate adjacency
    if adjacency_type == 'signed':
        corr = adjacency.signed_adjacency(adata)
    # calculate adjacency
    if adjacency_type == 'signed_hybrid':
        corr = adjacency.signed_hybrid_adjacency(adata)

    np.fill_diagonal(corr, 0)

    scale_free_power = connectivity.compute_scale_free_power(corr.copy())

    tom = connectivity.compute_tom(corr**scale_free_power)

    indexer, clustermap, labels = generate_gene_modules(tom, cutoff, min_cluster_size, cluster_method, cluster_height)

    return clustermap, indexer, labels

def generate_gene_modules(tom: np.ndarray, cutoff: float = 0.1, min_cluster_size: int = 10, cluster_method: str = "hierarchical", cluster_height = 0.3):
    """_summary_

    Args:
        tom (np.ndarray): Topological overlap matrix.
        cutoff (float, optional): Percentage of genes to include when clustering. Defaults to 0.2.
    """
    # TODO: make cutoff adaptive
    print(int(tom.shape[0]*cutoff))
    cutoff_max_connectivity = pd.Series(tom.sum(axis=0)) \
        .sort_values() \
        .iloc[-int(tom.shape[0]*cutoff)]

    if cutoff == 1:
        cutoff_max_connectivity = 0

    indexer = (tom.sum(axis=0)>cutoff_max_connectivity)

    if cluster_method == "hdbscan":
        labels = HDBSCAN(min_cluster_size=min_cluster_size) \
            .fit(tom[indexer][:, indexer]) \
            .labels_
        

    if cluster_method == "hierarchical":
        Z = sch.linkage(tom[indexer][:, indexer], method='ward')
        labels = cutreeHybrid(Z, tom)

    color_mapping = common.values_to_hex(labels, "tab20")
    row_colors = np.array([color_mapping[label] for label in labels])

    clustermap = sns.clustermap(tom[indexer][:, indexer], row_colors=row_colors, col_colors=row_colors)

    # TODO: return dict of module ids and gene names
    return indexer, clustermap, labels