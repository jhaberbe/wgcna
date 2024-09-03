import numpy as np

def compute_tom(adjacency_matrix):
    """Compute Topological Overlap Matrix

    Args:
        adjacency_matrix (np.array): Pearson correlation (or other similarity measure). 

    Returns:
        np.array: Topological Overlap Matrix
    """
    # Ensure adjacency matrix is numpy array
    A = np.array(adjacency_matrix)
    
    # Number of nodes
    n = A.shape[0]
    
    # Compute the degree of each node
    k = np.sum(A, axis=1)
    
    # Compute the l_ij (pairwise connectivity) matrix using matrix multiplication
    L = A @ A
    
    # Add the diagonal of A to L to get l_ij + A_ij
    L_plus_A = L + A
    
    # Compute min(k_i, k_j) matrix
    min_k = np.minimum.outer(k, k)
    
    # Compute TOM matrix
    TOM = L_plus_A / (min_k + 1 - A)
    
    return TOM

def compute_scale_free_power(corr_coef):
    """Compute Scale Free Power using heuristic log(connectivity) vs log(p(k))

    Args:
        corr_coef (np.array): Pairwise Pearson's correlation

    Returns:
        int: smallest power that induces a scale free topology (R^2 of 0.8).
    """
    # Create a copy of the original correlation coefficient matrix
    corr_coef_copy = corr_coef.copy()

    # Initialize variables
    r2_max = 0
    power_max = 0

    # Iterate until r2 reaches 0.8 or power exceeds 12
    for power in range(1, 21):

        # Compute the sum across axis 0 with a small constant added
        k = np.nansum(corr_coef_copy, axis=0) + 1e-10
        
        # Compute the histogram of the logarithm of k
        p, k = np.histogram(k)
        
        logk = np.log10(k)

        # Normalize the histogram and calculate the squared correlation coefficient (r2)
        normalized_p = np.log10(p / p.sum())
        r2 = np.corrcoef(logk[1:], normalized_p)[0, 1] ** 2

        # Update r2_max and power_max if r2 is the highest observed so far
        if r2 > r2_max:
            r2_max = r2
            power_max = power

        print(f"Power: {power}, R^2: {r2}")

        # If r2 has reached the threshold, break early
        if r2 >= 0.8:
            return power_max

        # Update the correlation coefficient matrix for the next iteration
        corr_coef_copy *= corr_coef
    
    return power_max
