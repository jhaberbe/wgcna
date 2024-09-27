# Importing package modules
from . import common
from . import adjacency
from . import connectivity

import numpy as np
import pandas as pd

import scanpy as sc
import anndata as ad

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch

from dynamicTreeCut import cutreeHybrid

def run_wgcna(adata: ad.AnnData, adjacency_type: str = 'unsigned'):
    """Runs WGCNA

    Args:
        adata (ad.AnnData): Dataset of interest.
        adjacency_type (str): adjacency type of ['signed', 'unsigned', 'signed_hybrid']

    Returns:
        sns.ClusterMap: seaborn clustermap of TOM 
        np.array: array of boolean values to index their adata.var 
        np.array: clustering labels.
    """

    gene_modules = dict()

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

    labels = generate_gene_modules(tom)

    adata.var["module"] = labels

    for module_label in np.unique(labels):
        gene_modules[module_label] = pd.Series(
            tom[labels==module_label][:, labels==module_label].sum(axis=1),
            index = adata.var_names[labels==module_label]
        ).sort_values(axis=0, ascending=False)

    return gene_modules

def generate_gene_modules(tom: np.ndarray, plot=True):
    """Generate Gene Modules using Adaptive Tree Cuts and  

    Args:
        tom (np.ndarray): Topological overlap matrix.
    
    Returns:
        sns.ClusterMap: seaborn clustermap of TOM 
        np.array: array of boolean values to index their adata.var 
        np.array: clustering labels.
    """
    # TODO: make cutoff adaptive
    cutoff_max_connectivity = pd.Series(tom.sum(axis=0)) \
        .sort_values()

    Z = sch.linkage(tom, method='ward')
    labels = cutreeHybrid(Z, tom)

    color_mapping = common.values_to_hex(labels["labels"], "tab20")
    row_colors = np.array([color_mapping[label] for label in labels["labels"]])

    if plot: 
        sns.clustermap(tom, row_colors=row_colors, col_colors=row_colors)

    # TODO: return dict of module ids and gene names
    return labels["labels"]