import numpy as np
import anndata as ad

def unsigned_adjacency(adata: ad.AnnData):
    """_summary_

    Args:
        data (_type_): _description_
    """
    try: 
        data = adata.X.todense()
    except:
        data = adata.X

    data = np.corrcoef(data.T)

    return np.abs(data)

def signed_adjacency(adata: ad.AnnData):
    """_summary_

    Args:
        data (_type_): _description_
    """
    try: 
        data = adata.X.todense()
    except:
        data = adata.X

    data = np.corrcoef(data.T)

    return (1 + data) / 2

def signed_hybrid_adjacency(adata: ad.AnnData):
    """_summary_

    Args:
        data (_type_): _description_
    """
    try: 
        data = adata.X.todense()
    except:
        data = adata.X

    data = np.corrcoef(data.T)

    return np.where(data > 0, data, 0)
