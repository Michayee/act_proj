import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm
from scipy.stats.qmc import Sobol
import os
import shutil

def clear_temp(temp_folder):
    temp_folder_spec = 'temp_simu_'
    for entry in os.listdir(temp_folder):
        full_path = os.path.join(temp_folder, entry)
        if os.path.isdir(full_path) and temp_folder_spec in entry:
            try:
                shutil.rmtree(full_path)
            finally:
                pass

def sobol(batch_size : int, num_dimensions : int, n_init : int,
          results_file : str, func_eval_batch : Callable, seed = -1, temp_folder: str = None) -> None:
    
    if seed == -1:
        if_scramble = False
        seed = 0
    else:
        if_scramble = True

    SobolEngine = Sobol(d = num_dimensions,  scramble = if_scramble, seed = seed)
    sobol_list = SobolEngine.random_base2(m = int(np.log2(n_init)) + 1)

    num_iterations = int(n_init / batch_size)
    result_df = pd.DataFrame()

    pbar = tqdm(range(num_iterations), desc="SOBOL")
    sobol_i = 0
    for _ in pbar:
        X_sobol = sobol_list[batch_size * sobol_i: batch_size * (sobol_i + 1)]
        sobol_i = sobol_i + 1
        Y_sobol = func_eval_batch(X_sobol.tolist())

        new_rows = pd.DataFrame(X_sobol, columns=[f'x_{i}' for i in range(num_dimensions)])
        new_rows.insert(0, 'score', Y_sobol)

        result_df = pd.concat([result_df, new_rows], ignore_index=True)
        result_df.to_csv(results_file, float_format='%.8e', index=False)

        max_score = result_df['score'].max()
        min_score = result_df['score'].min()

        pbar.set_postfix({'Max Score': max_score, 'Min Score': min_score})

        if temp_folder is not None:
            clear_temp(temp_folder)

