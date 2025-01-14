from typing import Callable, List, Tuple
# import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from deap import base, cma, creator, tools
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

def cma_es(num_inds: int, num_dimensions: int, num_iterations: int,
           input_file: str, results_file: str, func_eval_batch: Callable, temp_folder: str = None) -> None:
    
    bounds = [(0, 1) for _ in range(num_dimensions)]
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    # 使用CMA-ES策略
    strategy = cma.Strategy(centroid=[0.5] * num_dimensions, sigma = 0.5, lambda_ = num_inds)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    # toolbox.register("evaluate", func_eval_ind) # not work

    # 限制范围
    def checkBounds(min, max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] < min[i]:
                            child[i] = min[i]
                        elif child[i] > max[i]:
                            child[i] = max[i]
                return offspring
            return wrapper
        return decorator
    toolbox.decorate("generate", checkBounds([b[0] for b in bounds], [b[1] for b in bounds]))

    np.random.seed(0)

    if input_file and os.path.exists(input_file):
        input_rows = pd.read_csv(input_file)
        print(f'{input_rows["x_0"].count()} particles are provided with initial position')
        input_rows = input_rows.loc[~input_rows["x_0"].isna()]

        initial_individuals = [creator.Individual(row[[f'x_{i}' for i in range(num_dimensions)]]) for index, row in input_rows.iterrows()]
        if len(initial_individuals) < num_inds:
            initial_individuals.extend(toolbox.generate(n=num_inds - len(initial_individuals)))
    else:
        initial_individuals = toolbox.generate()
    population = initial_individuals

    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    # fitnesses = toolbox.map(toolbox.evaluate, population)
    fitnesses = func_eval_batch(population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit, # , is necessary, seemingly related to weights=(-1.0,)
    result_df = pd.DataFrame(population, columns=[f'x_{i}' for i in range(num_dimensions)])
    result_df.insert(0, 'score', fitnesses)
    result_df.to_csv(results_file, float_format='%.8e', index=False)

    toolbox.update(population)

    pbar = tqdm(range(num_iterations), desc="CMA-ES")

    min_score = np.min([ind.fitness.values[0] for ind in population])

    # 运行 CMA-ES 算法
    for _ in pbar:
        min_score = min(min_score, np.min([ind.fitness.values[0] for ind in population]))
        pbar.set_postfix({'Best fitness': min_score})

        population = toolbox.generate()

        # fitnesses = toolbox.map(toolbox.evaluate, population)
        fitnesses = func_eval_batch(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit, # , is necessary
        new_rows = pd.DataFrame(population, columns=[f'x_{i}' for i in range(num_dimensions)])
        new_rows['score'] = fitnesses
        result_df = pd.concat([result_df, new_rows], ignore_index=True)
        result_df.to_csv(results_file, float_format='%.8e', index=False)

        if temp_folder is not None:
            clear_temp(temp_folder)

        toolbox.update(population)

    pbar.close()
    # pool.close()
    # pool.join()