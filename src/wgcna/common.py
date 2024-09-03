import scanpy as sc

import matplotlib
import matplotlib.pyplot as plt

def annotate_gene_groups(adata):
    """Annotate Gene Groups
    Args:
        adata (ad.AnnData): AnnData with gene information.
    
    Returns:
        ad.AnnData: AnnData with updated .var fields for mitochondrial, ribosomal, heme, linc, and antisense genes
    """
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    # long intergenic non-coding RNA
    adata.var["linc"] = adata.var_names.str.startswith("LINC") | adata.var_names.str.contains(r'\.\d+$')

    # Antisense
    adata.var["antisense"] = adata.var_names.str.endswith("-AS1") | adata.var_names.str.endswith("-AS")

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb", "linc", "antisense"], inplace=True, log1p=True
    )

    return adata


def select_genes(adata, mask_cols = ["linc", "mt", "ribo", "hb", "antisense"], percent_threshold = 0.05):
    """Select a subset of adata based on expression threshold and mask columns. 

    Args:
        adata (ad.AnnData): Annotated Dataset
        mask_cols (list, optional): list of columns to mask by (excludes). Defaults to ["linc", "mt", "ribo", "hb", "antisense"].
        percent_threshold (float, optional): Percent of cells that need to express in order to work. Defaults to 0.05.

    Returns:
        ad.AnnData: Subsetted adata with genes that meet the criteria.
    """
    adata = annotate_gene_groups(adata)

    expression_meets_threshold = ((adata.X > 0).mean(axis=0)>percent_threshold).tolist()[0]
    feature_not_masked = adata.var[mask_cols].sum(axis=1).eq(0)

    return adata[:, expression_meets_threshold & feature_not_masked].copy()


def values_to_hex(values, cmap_name='tab20'):
    """Converts unique values in a list to their corresponding hex color values using a specified colormap.

    Args:
        values (list): List of unique values to convert to hex colors.
        cmap_name (str): Name of the colormap to use (default is 'tab10').

    Returns:
        dict: A dictionary where keys are the unique values and values are their corresponding hex colors.
    """
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Ensure the values are unique
    unique_values = list(set(values))
    
    # Map each unique value to a color
    hex_colors = {value: matplotlib.colors.rgb2hex(cmap(i / len(unique_values))) for i, value in enumerate(unique_values)}

    return hex_colors