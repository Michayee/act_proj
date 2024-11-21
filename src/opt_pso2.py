import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from typing import Callable, List, Tuple

def update_particle(x: np.ndarray, v: np.ndarray, 
                    p_best: np.ndarray, g_best: np.ndarray, 
                    bounds: List[Tuple[float, float]], 
                    w=0.5, c1=1.0, c2=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    更新粒子位置和速度。
    """
    r1, r2 = np.random.rand(), np.random.rand()
    new_v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
    new_x = x + new_v
    new_x = np.clip(new_x, a_min=[b[0] for b in bounds], a_max=[b[1] for b in bounds])  # 应用位置限制
    return new_x, new_v

def pso(num_particles: int, num_dimensions: int, num_iterations: int,
        input_file: str, results_file: str, func_eval_batch: Callable) -> None:
    bounds = [(0, 1) for _ in range(num_dimensions)]
    swarm = np.random.uniform(0, 1, (num_particles, num_dimensions))
    velocity = np.random.uniform(-1, 1, (num_particles, num_dimensions))

    result_df = pd.DataFrame(columns=['score']+[f'x_{i}' for i in range(num_dimensions)])
    result_df[[f'x_{i}' for i in range(num_dimensions)]] = swarm

    if input_file and os.path.exists(input_file):
        new_rows = pd.read_csv(input_file)
        result_df.update(new_rows)
        print(f'{new_rows['x_0'].count()} particles are provided with initial position')
        print(f'{new_rows['score'].count()} of them are provided with initial score')
    else:
        print('No initial position provided, random initial position adopted')

    print('Initializing PSO process')
    lack_score = result_df['score'].isna()
    temp_swarm = result_df.loc[lack_score, [f'x_{i}' for i in range(num_dimensions)]].to_numpy()
    temp_scores = np.array(func_eval_batch(temp_swarm))
    result_df.loc[lack_score, 'score'] = temp_scores
    result_df.to_csv(results_file, float_format='%.5e', index=False)

    swarm = result_df.loc[:, [f'x_{i}' for i in range(num_dimensions)]].to_numpy()
    scores = result_df.loc[:,'score'].to_numpy()

    personal_best = np.copy(swarm)
    personal_best_scores = np.copy(scores)
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    pbar = tqdm(range(num_iterations), desc="PSO")
    for iteration in pbar:
        pbar.set_postfix({'Best Score': global_best_score})

        # 更新粒子位置和速度
        for i in range(num_particles):
            swarm[i], velocity[i] = update_particle(swarm[i], velocity[i], personal_best[i], global_best, bounds)
        
        # 更新每代的结果并保存
        scores = np.array(func_eval_batch(swarm))
        new_rows = pd.DataFrame(swarm, columns=[f'x_{i}' for i in range(num_dimensions)])
        new_rows['score'] = scores
        result_df = pd.concat([result_df, new_rows], ignore_index=True)
        result_df.to_csv(results_file, float_format='%.5e', index=False)

        # 更新个人和全局最优
        improved = scores < personal_best_scores
        personal_best[improved] = swarm[improved]
        personal_best_scores[improved] = scores[improved]

        # 更新全局最优
        if np.min(scores) < global_best_score:
            global_best_score = np.min(scores)
            global_best = swarm[np.argmin(scores)]