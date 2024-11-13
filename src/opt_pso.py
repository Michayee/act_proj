import random
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import os
import time
from typing import List, Dict, Tuple, Union, Callable
from tqdm import tqdm

# 更新粒子位置和速度
def update_particle(x, v, p_best, g_best, bounds, w=0.5, c1=1.0, c2=1.0):
    r1, r2 = random.random(), random.random()
    new_v = [w*vi + c1*r1*(pbi-xi) + c2*r2*(gbi-xi) for xi, vi, pbi, gbi in zip(x, v, p_best, g_best)]
    new_x = [xi + vi for xi, vi in zip(x, new_v)]
    # 应用位置限制
    new_x = [max(min(xi, bounds[i][1]), bounds[i][0]) for i, xi in enumerate(new_x)]
    return new_x, new_v

# 粒子群优化算法
def pso(num_particles, 
        num_dimensions, 
        num_iterations,
        input_file: str,
        results_file: str,
        func_eval_batch: Callable):
    
    bounds = [(0, 1) for _ in range(num_dimensions)]
    # 初始化粒子群
    swarm = [[np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)] for _ in range(num_particles)]
    velocity = [np.random.uniform(0, 1, num_dimensions).tolist() for _ in range(num_particles)]

    # 如果存在初始结果，读取初始结果
    if input_file is not None:
        if os.path.exists(input_file):
            results_df = pd.read_csv(input_file)
            # 有改进空间
            # input__part = input__df[input__df['score'].isna()]
            # X_df = input__part[[f'x_{i}' for i in range(num_dimensions)]]
            # X_list = X_df.apply(lambda row: row.tolist(), axis=1).tolist()
            # Y_list = func_eval_batch(X_list)
            X_df = results_df[[f'x_{i}' for i in range(num_dimensions)]]
            X_list = X_df.apply(lambda row: row.tolist(), axis=1).tolist()
            for i in range(len(X_list)):
                swarm[i] = X_list[i]
    
    result_df = pd.DataFrame(columns=['score']+[f'x_{i}' for i in range(num_dimensions)])

    personal_best = swarm[:]
    scores = func_eval_batch(swarm) # results for the generation
    new_rows = pd.DataFrame(swarm, columns=[f'x_{i}' for i in range(num_dimensions)])
    new_rows.insert(0, 'score', scores)
    result_df = pd.concat([result_df, new_rows], ignore_index=True)
    result_df.to_csv(results_file, float_format = '%.5e', index = False)

    personal_best_scores = scores  
    # 全局最优，求最小值
    best_i = np.argmin(personal_best_scores)
    global_best = personal_best[best_i]
    global_best_score = min(personal_best_scores)

    pbar = tqdm(range(num_iterations), desc="PSO")
    
    for iteration in pbar:
        pbar.set_postfix({'best score': global_best_score})
        # print(f"Iteration {iteration+1}/{num_iterations}, Best Score: {global_best_score}")
        # 更新个人最优和全局最优
        best_i = -1
        for i in range(num_particles):
            if scores[i] < personal_best_scores[i]:
                personal_best[i] = swarm[i]
                personal_best_scores[i] = scores[i]
                if scores[i] < global_best_score:
                    global_best = swarm[i]
                    global_best_score = scores[i]
                    best_i = i
        
        # 更新粒子位置和速度
        for i in range(num_particles):
            swarm[i], velocity[i] = update_particle(swarm[i], velocity[i], personal_best[i], global_best, bounds)

        scores = func_eval_batch(swarm)
        new_rows = pd.DataFrame(swarm, columns=[f'x_{i}' for i in range(num_dimensions)])
        new_rows.insert(0, 'score', scores)
        result_df = pd.concat([result_df, new_rows], ignore_index=True)
        result_df.to_csv(results_file, float_format = '%.5e', index = False)
        
        