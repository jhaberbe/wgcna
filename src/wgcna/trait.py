"""Tools for trait correlation testing."""

import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

def standardize_columns(df):
    """
    Standardize the values in each column of a DataFrame to have a mean of 0 and a standard deviation of 1.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with standardized values
    """
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

def generate_null_distribution(adata):
    """
    Generate a null distribution of standardized eigengene scores for random gene sets. 
    
    Args:
        adata (ad.AnnData): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with standardized values
    """
    null_distribution = pd.DataFrame(
        [
            sklearn.decomposition.PCA(n_components=1) \
                .fit_transform(adata[:, adata.var.index[np.random.choice(adata.shape[1], 100)]].X)[:, 0] 
            for _ in tqdm(range(1000))
        ]
    )

    return standardize_columns(null_distribution.T)

def permutation_test_module_trait_relationship(adata, null_distribution_standardized, subset_indicies):
    """
    Perform permutation testing of module trait relationship to inform.
    
    Args:
        adata (ad.AnnData): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with standardized values
    """
    for i in adata.var.module.sort_values().unique():
        star = ""
        # Fit Eigengene Score
        eigengene_scores = sklearn.decomposition.PCA(n_components=1) \
            .fit_transform(adata[:, adata.var.query(f"module == {i}").index].X)[:, 0]
        
        # Normalize
        eigengene_scores = (eigengene_scores - eigengene_scores.mean())/eigengene_scores.std()

        # Determine trait eigengene score
        trait_null_distribution_standardized = null_distribution_standardized[subset_indicies]
        trait_score = eigengene_scores[subset_indicies]

        p_trait = (null_distribution_standardized[subset_indicies].mean() > trait_score.mean()).mean()

        if p_trait < 0.05 or (1-p_trait) < 0.05:
            star = "*"

        print(i, p_trait, star)