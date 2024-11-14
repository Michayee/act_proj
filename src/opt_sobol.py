from torch.quasirandom import SobolEngine
import torch
import pandas as pd
# import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def sobol(batch_size : int, num_dimensions : int, n_init : int,
          results_file : str, func_eval_batch : Callable, seed = 0):
    sobol = SobolEngine(dimension = num_dimensions, scramble = False, seed = seed)
    num_iterations = int(n_init / batch_size) + 1
    result_df = pd.DataFrame()

    pbar = tqdm(range(num_iterations), desc="SOBOL")
    for _ in pbar:
        X_sobol = sobol.draw(n = batch_size).to(dtype=dtype, device=device)
        Y_sobol = torch.tensor(
            func_eval_batch(X_sobol.tolist()), dtype=dtype, device=device
        ).unsqueeze(-1)

        new_rows = pd.DataFrame(X_sobol.cpu(), columns=[f'x_{i}' for i in range(num_dimensions)])
        new_rows.insert(0, 'score', Y_sobol.cpu())

        result_df = pd.concat([result_df, new_rows], ignore_index=True)
        result_df.to_csv(results_file, float_format='%.8e', index=False)

        max_score = result_df['score'].max()
        min_score = result_df['score'].min()

        pbar.set_postfix({'Max Score': max_score, 'Min Score': min_score})

