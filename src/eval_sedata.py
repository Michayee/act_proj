from typing import List, Dict, Tuple, Union, Callable
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# hausdorff distance and frechet distance

def hausdorff_distance_KD(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between two sets of points, A and B, using KD-trees.

    Parameters:
    - A (np.ndarray): The first set of points, a two-dimensional array where each row is a point.
    - B (np.ndarray): The second set of points, a two-dimensional array where each row is a point.

    Returns:
    - float: The Hausdorff distance between point sets A and B.
    """
    tree_A = KDTree(A)
    tree_B = KDTree(B)

    # Distance from each point in B to the closest point in A
    dist_B_to_A, _ = tree_A.query(B, k=1)
    max_min_dist_B_to_A = np.max(dist_B_to_A)

    # Distance from each point in A to the closest point in B
    dist_A_to_B, _ = tree_B.query(A, k=1)
    max_min_dist_A_to_B = np.max(dist_A_to_B)

    # The Hausdorff distance is the maximum of these two distances
    return max(max_min_dist_A_to_B, max_min_dist_B_to_A)

def frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Calculate the discrete Fréchet distance between two curves represented as sequences of points P and Q.
    
    Parameters:
    - P (np.ndarray): The first sequence of points, a two-dimensional array where each row is a point.
    - Q (np.ndarray): The second sequence of points, a two-dimensional array where each row is a point.

    Returns:
    - float: The discrete Fréchet distance between curves P and Q.
    """
    n = len(P)
    m = len(Q)
    DP = np.zeros((n, m))

    # Initialize the dynamic programming table and compute the distances
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(P[i] - Q[j])
            if i == 0 and j == 0:
                DP[i][j] = dist
            elif i == 0:
                DP[i][j] = max(DP[i][j-1], dist)
            elif j == 0:
                DP[i][j] = max(DP[i-1][j], dist)
            else:
                DP[i][j] = max(min(DP[i-1][j], DP[i][j-1], DP[i-1][j-1]), dist)

    return DP[-1][-1]

def calculate_DA(temp_gamma: List[float]) -> List[float]:
    """
    Calculate the double amplitude strain (DA) for a given list of strains.
    
    The function computes the difference between the cumulative maximum and minimum 
    strains at each point in the list, which represents the DA at that point.

    Parameters:
    - temp_gamma (List[float]): List of strain values, typically 'gamma' from a DataFrame.

    Returns:
    - List[float]: List of DA values corresponding to each strain in 'temp_gamma'.
    """
    if not temp_gamma:
        return []

    list_DA = []
    temp_min = temp_gamma[0]
    temp_max = temp_gamma[0]

    # Initialize the first element as the difference of the same element
    list_DA.append(0.0)

    # Compute the DA starting from the second element
    for gamma_i in temp_gamma[1:]:
        temp_min = min(temp_min, gamma_i)
        temp_max = max(temp_max, gamma_i)
        list_DA.append(temp_max - temp_min)

    return list_DA

def eval_sampling(df_input: pd.DataFrame, index: str = 'n', nforn: int = 24) -> pd.DataFrame:
    """
    Sample `nforn` entries per cycle from a DataFrame based on a cycling index.
    
    Parameters:
    - df_input (pd.DataFrame): The input DataFrame.
    - index (str): The column name in `df_input` that contains the cycle index.
    - nforn (int): The number of samples to retain per cycle.

    Returns:
    - pd.DataFrame: A DataFrame containing the sampled entries.
    """
    # Calculate the sample indices based on the cycle index
    n_diff = df_input[index] * nforn
    n_diff_rounded = np.floor(n_diff).astype(int)
    
    # Get unique indices to sample - the first occurrence of each rounded index
    sampl_indices = n_diff_rounded.drop_duplicates(keep = 'first').index
    
    # Return the sampled DataFrame
    df_output = df_input.loc[sampl_indices].copy()
    return df_output

def eval_drained_cyc_torsion(test_file: str, simu_file: str, N_lim: float = 20, nforn: int = 24):
    """
    Evaluate drained cyclic torsion by comparing test and simulation results using Hausdorff distance.

    Parameters:
    - test_file (str): File path for the test results.
    - simu_file (str): File path for the simulation results.
    - N_lim (int): Limit for the normalized cycle index.
    - nforn (int): Number of samples per cycle.

    Returns:
    - float: Hausdorff distance between the two datasets.
    """
    try:
        test_result = pd.read_csv(test_file)
        simu_result = pd.read_csv(simu_file)
    except FileNotFoundError:
        return 999  # Return error code if files not found
    
    if simu_result.isna().any().any():
        print(f'Nan in {simu_file}')
        return 999
    
    test_sampled = eval_sampling(test_result, 'n', nforn=nforn)
    simu_sampled = eval_sampling(simu_result, 'n', nforn=nforn)
    
    test_trimmed = test_sampled[test_sampled['n'] < N_lim].copy()
    simu_trimmed = simu_sampled[simu_sampled['n'] < N_lim].copy()
    
    gamma_max = test_trimmed['gamma'].max()
    ev_max = test_trimmed['e_v'].max()
    
    test_trimmed['gamma_1'] = test_trimmed['gamma'] / gamma_max
    simu_trimmed['gamma_1'] = simu_trimmed['gamma'] / gamma_max
    
    test_trimmed['ev_1'] = test_trimmed['e_v'] / ev_max
    simu_trimmed['ev_1'] = simu_trimmed['e_v'] / ev_max
    
    test_trimmed['n_1'] = test_trimmed['n'] / N_lim
    simu_trimmed['n_1'] = simu_trimmed['n'] / N_lim
    
    set_A = test_trimmed[['n_1','gamma_1','ev_1']].to_numpy()
    set_B = simu_trimmed[['n_1','gamma_1','ev_1']].to_numpy()
    return hausdorff_distance_KD(set_A, set_B)

def eval_undrained_cyc_torsion(test_file:str,simu_file:str,DA_lim = 0.075,nforn = 24):
    """
    Evaluate undrained cyclic torsion test data by comparing test and simulation results using Hausdorff distance.

    Parameters:
    - test_file (str): File path for the test results.
    - simu_file (str): File path for the simulation results.
    - DA_lim (float): Limit for DA, double amplitude strain
    - nforn (int): Number of samples per cycle.

    Returns:
    - float: Hausdorff distance or discrepancy ratio between the two datasets, or error code if files do not exist.
    """
    try:
        test_result = pd.read_csv(test_file)
        simu_result = pd.read_csv(simu_file)
    except FileNotFoundError:
        return 99  # Return error code if files not found
    
    if simu_result.isna().any().any():
        print(f'Nan in {simu_file}')
        return 99

    test_sampled = eval_sampling(test_result, 'n', nforn=nforn)
    simu_sampled = eval_sampling(simu_result, 'n', nforn=nforn)
    test_sampled['DA'] = calculate_DA(test_sampled['gamma'].tolist())
    simu_sampled['DA'] = calculate_DA(simu_sampled['gamma'].tolist())

    test_trimmed = test_sampled[test_sampled['DA'] < DA_lim].copy()
    simu_trimmed = simu_sampled[simu_sampled['DA'] < DA_lim].copy()
        
    gamma_max = DA_lim * 0.5
    p_test = test_trimmed['p'].max()
    p_simu = simu_trimmed['p'].max()
    n_test = test_trimmed['n'].max()
    n_simu = simu_trimmed['n'].max()

    dis_ref = abs(max(n_simu+1,n_test+1)/min(n_simu+1,n_test+1) - 1)
    if dis_ref  > 0.5:
        return dis_ref
    
    else:
        test_trimmed['gamma_1'] = test_trimmed['gamma'] / gamma_max
        test_trimmed['p_1'] = test_trimmed['p'] / p_test
        simu_trimmed['gamma_1'] = simu_trimmed['gamma'] / gamma_max
        simu_trimmed['p_1'] = simu_trimmed['p'] / p_simu

        test_trimmed['n_1'] = test_trimmed['n'] / n_test
        simu_trimmed['n_1'] = simu_trimmed['n'] / n_test
         
    set_A = test_trimmed[['n_1','gamma_1','p_1']].to_numpy()
    set_B = simu_trimmed[['n_1','gamma_1','p_1']].to_numpy()
    return hausdorff_distance_KD(set_A, set_B)

def eval_undrained_mono_torsion(test_file:str,simu_file:str,nforn = 200):
    """
    Evaluate undrained cyclic torsion test data by comparing test and simulation results using Hausdorff distance.

    Parameters:
    - test_file (str): File path for the test results.
    - simu_file (str): File path for the simulation results.
    - DA_lim (float): Limit for DA, double amplitude strain
    - nforn (int): Number of samples per cycle.

    Returns:
    - float: Hausdorff distance or discrepancy ratio between the two datasets, or error code if files do not exist.
    """
    try:
        test_result = pd.read_csv(test_file)
        simu_result = pd.read_csv(simu_file)
    except FileNotFoundError:
        return 999  # Return error code if files not found
    
    if simu_result.isna().any().any():
        print(f'Nan in {simu_file}')
        return 999

    # test_sampled = eval_sampling(test_result, 'n', nforn=nforn)
    # simu_sampled = eval_sampling(simu_result, 'n', nforn=nforn)
    # test_sampled['DA'] = calculate_DA(test_sampled['tau'].tolist())
    # simu_sampled['DA'] = calculate_DA(simu_sampled['tau'].tolist())
    test_sampled = test_result.iloc[::int(test_result.shape[0]/nforn),:].copy()
    simu_sampled = simu_result.iloc[::int(simu_result.shape[0]/nforn),:].copy()

    gamma_max = max(test_result['gamma'].max(), simu_result['gamma'].max())
    p_test = test_result['p'].loc[0]
    p_simu = simu_result['p'].loc[0]

    test_sampled['p_1'] = test_sampled['p'] / p_test
    test_sampled['q_1'] = test_sampled['q'] / p_test
    test_sampled['gamma_1'] = test_sampled['gamma'] / gamma_max

    simu_sampled['p_1'] = simu_sampled['p'] / p_simu
    simu_sampled['q_1'] = simu_sampled['q'] / p_simu
    simu_sampled['gamma_1'] = simu_sampled['gamma'] / gamma_max
         
    set_A = test_sampled[['gamma_1','p_1','q_1']].to_numpy()
    set_B = simu_sampled[['gamma_1','p_1','q_1']].to_numpy()
    return frechet_distance(set_A, set_B)