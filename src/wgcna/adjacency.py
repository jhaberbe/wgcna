import numpy as np
import anndata as ad
from sklearn.covariance import ShrunkCovariance

def estimate_covariance(data):
    if data.shape[1] > data.shape[0]:
        data = ShrunkCovariance().fit(data.T)
    else:    
        data = np.corrcoef(data.T)
    return data

def unsigned_adjacency(adata: ad.AnnData):
    """Unsigned Adjacency

    Args:
        adata (ad.AnnData): AnnData object

    Returns:
        np.array: signed pearson's correlation array.
    """
    try: 
        data = adata.X.todense()
    except:
        data = adata.X

    data = estimate_covariance(data)

    return np.abs(data)

def signed_adjacency(adata: ad.AnnData):
    """Signed Adjacency

    Args:
        adata (ad.AnnData): AnnData object

    Returns:
        np.array: signed pearson's correlation array.
    """
    try: 
        data = adata.X.todense()
    except:
        data = adata.X

    data = estimate_covariance(data)

    return (1 + data) / 2

def signed_hybrid_adjacency(adata: ad.AnnData):
    """Signed Hybrid Adjacency

    Args:
        adata (ad.AnnData): AnnData object

    Returns:
        np.array: signed pearson's correlation array.
    """
    try: 
        data = adata.X.todense()
    except:
        data = adata.X

    data = estimate_covariance(data)

    return np.where(data > 0, data, 0)
