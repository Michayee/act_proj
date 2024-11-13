import os
import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict, Tuple, Union, Callable

def read_model_parameters(file_path: str) -> np.array:
    """从给定文件路径读取浮点数并转换为 numpy 数组"""
    with open(file_path, 'r') as file:
        numbers = file.read().strip().split()
        parameters = np.array([float(num) for num in numbers])
    return parameters

def build_result_tree(base_path: str, func_para2x: Callable) -> tuple[KDTree, dict]:
    """遍历 base_path 下的所有子文件夹，并为 modelPara.dat 中的参数构建 KDTree"""
    parameter_list = []
    x_list = []
    folder_index = {}
    index = 0
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        file_path = os.path.join(folder_path, 'modelPara.dat')
        if os.path.isfile(file_path):
            parameters = read_model_parameters(file_path)
            # parameter_list.append(parameters)
            x_list.append(func_para2x(parameters))
            folder_index[index] = folder
            index += 1
    # if parameter_list:
    #     kdtree = KDTree(np.array(parameter_list))
    #     return kdtree, folder_index
    if x_list:
        kdtree = KDTree(np.array(x_list))
        return kdtree, folder_index
    else:
        return None, None

def find_closest_folder(kdtree: KDTree, folder_index: dict, target_vector: np.array) -> str:
    """使用 KDTree 查找与 target_vector 欧式距离最近的文件夹"""
    if kdtree is not None:
        distance, idx = kdtree.query(target_vector)
        # print(distance)
        closest_folder = folder_index[idx]
        return closest_folder
    else:
        return "No folders available"
    

if __name__ =='main':
    import pandas as pd
    from src.target_cycliq import ParaTrans
    paras_file = 'para_table.csv'
    paras_config = ParaTrans(paras_file)

    # 设置基础路径和目标向量
    base_path = 'test_eval/test_1730341687/'  # 替换为你的父文件夹路径
    
    # 构建 KDTree 和文件夹索引
    kdtree, folder_index = build_result_tree(base_path)

    # 结果处理    
    temp_df = pd.read_csv(base_path + 'PSO_1730341687.csv')
    temp_df[paras_config.keys] = temp_df.apply(lambda row: paras_config.x2para([row[f'x_{i}'] for i in range(paras_config.x_len)]), axis=1, result_type='expand')
    
    min_rows = []
    # 分组处理每12行
    for i in range(0, len(temp_df), 12):
        # 获取当前12行的数据
        subset = temp_df.iloc[i:i+12]
        # 找到这12行中 score 最小的行
        min_rows.append(subset['score'].idxmin())
    
    for row in min_rows:
        target_vector = temp_df.loc[row,[f'x_{i}' for i in range(paras_config.x_len)]].to_numpy()
        closest_folder_name = find_closest_folder(kdtree, folder_index, target_vector)
        print("The closest folder is:", closest_folder_name)