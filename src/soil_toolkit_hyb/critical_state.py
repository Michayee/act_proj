import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Tuple, Union, Callable
from scipy.optimize import curve_fit

def get_ein(s: str) -> float:
    """
    从特定格式的字符串中提取数值。
    
    参数:
    s (str): 输入字符串，例如 'ein_0_752'。
    
    返回:
    ein: 包含提取的数值的字典。
    """
    # 定义要查找的模式：'p_数字'、'ein_数字'、'cyc_数字'
    # 注意数字可能包含小数点
    pattern = r'ein_(\d+_\d+)'
    
    # 使用正则表达式搜索匹配项
    match = re.search(pattern, s)
    if match:
        # 将匹配的字符串转换为相应的数值类型
        ein = float(match.group(1).replace('_', '.'))
        # 返回一个字典，包含提取的数值
        return ein
    else:
        # 如果没有找到匹配项，返回 None 或抛出异常
        return None

def find_eta_point(data_file : str, ein : float = None) -> pd.DataFrame:
    '''
    根据输入的文件，找临界状态点
    '''
    data_df = pd.read_csv(data_file)
    if ein == None:
        ein = get_ein(data_file)
    if ein == None:
        print('Lack initial void ratio')
        return pd.DataFrame()
    
    data_df['eta'] = data_df['q']/data_df['p']
    data_df['e'] = - data_df['e_v'] * (1 + ein) + ein

    if 'HCT' in data_file:
        theta = 0.0
    elif 'Tri' in data_file:
        theta = np.pi / 2

    eta_loc = data_df[data_df['p'] > 10]['eta'].idxmax()
    eta_p = pd.DataFrame(
        {'p': [data_df['p'].loc[eta_loc]],
         'e': [data_df['e'].loc[eta_loc]],
         'p_or_d': [1.0],
         'theta': [theta],
         'eta': [data_df['eta'].loc[eta_loc]],
         'source': os.path.split(data_file)[-1]}
    )

    if 'mono' in data_file:
        if 'undrained' in data_file:
            eta_loc = data_df['p'].idxmin()
        elif 'drained' in data_file:
            eta_loc = data_df['e'].idxmin()

        eta_d = pd.DataFrame(
            {'p': [data_df['p'].loc[eta_loc]],
             'e': [data_df['e'].loc[eta_loc]],
             'p_or_d': [0.0],
             'theta': [theta],
             'eta': [data_df['eta'].loc[eta_loc]],
             'source': os.path.split(data_file)[-1]}
        )

    elif  (('csr' in data_file) or ('cyc' in data_file)) and ('undrained' in data_file):
        eta_d = pd.DataFrame()
        p_0 = data_df['p'][0]
        n_liq = data_df[data_df['p'] < 0.05 * p_0]['n'].min()
        if n_liq is not np.nan:
            data_df = data_df[data_df['n'] < n_liq]
    
        data_df['eta'] = data_df['q']/data_df['p']
        data_df['e'] = - data_df['e_v'] * (1 + ein) + ein
    
        data_df['ncyc'] = np.ceil(data_df['n'] * 2 + 0.5)
        # data_df['ncyc'] = data_df['n'] * 2 + 0.5
        # 计算每个周期的 q 的最大值的 90%
        max_q_80 = data_df.groupby('ncyc')['q'].max().multiply(0.8)
        
        # 计算每个周期的 p 的最小值
        min_p_per_cycle = data_df.groupby('ncyc')['p'].min()
        
        # 映射最大值的 90% 和最小 p 值回到原 DataFrame
        data_df['max_q_80'] = data_df['ncyc'].map(max_q_80)
        data_df['min_p'] = data_df['ncyc'].map(min_p_per_cycle)
        
        # 筛选符合 q 大于其周期最大值的 90% 且 p 为该周期的最小值的行
        filtered_df = data_df[(data_df['q'] < data_df['max_q_80']) & (data_df['p'] == data_df['min_p']) & (data_df['p'] > 10)].copy()
        filtered_df['p_or_d'] = 0.0
        filtered_df['theta'] = theta
        filtered_df['source'] = os.path.split(data_file)[-1]
        eta_d = filtered_df[['p','e','p_or_d','theta','eta','source']]

    return pd.concat([eta_p, eta_d], ignore_index=True)



# 用一个 Mfc 控制
def eta_func1(x, e0, lamdac, ksi, Mfc, nb, nd):
    p_n, e_n, p_or_d, theta,  = x.T  # 解包输入数组
    sinphi = 3.0 * Mfc / (Mfc + 6.0)
    tanphi = sinphi / np.sqrt(1.0 - sinphi**2)
    Mfo = 2 * np.sqrt(3.0) * tanphi / np.sqrt(3.0 + 4.0 * tanphi**2)
    sin3theta = np.sin(3 * theta)
    gtheta = 1 / (1 + Mfc / 6.0 * (sin3theta + sin3theta**2) + (Mfc - Mfo) / Mfo * (1 - sin3theta**2))
    psi = e_n - (e0 - lamdac * (p_n / 100)**ksi)
    eta_p = np.exp(-nb * psi) * gtheta * Mfc
    eta_d = np.exp(nd * psi) * gtheta * Mfc
    return eta_p * p_or_d + eta_d * (1 - p_or_d)

param_constrains1 = [
    (0.8, 1.5), # e0
    (0.0, 1.0), # lambdac
    (0.5, 1.5), # ksi
    (0.5, 2.4), # Mfc
    # (0.5, 1.4), # Mfo
    (0.0, 5.0), # nb
    (0.0, 5.0), # nd
]

param_ini1 = [1.0, 0.1, 1.0, 1.5, 1.0, 1.0]

# 用 Mfc 和 Mfo 一起控制
def eta_func2(x, e0, lamdac, ksi, Mfc, Mfo, nb, nd):
    p_n, e_n, p_or_d, theta,  = x.T  # 解包输入数组
    # sinphi = 3.0 * Mfc / (Mfc + 6.0)
    # tanphi = sinphi / np.sqrt(1.0 - sinphi**2)
    # Mfo = 2 * np.sqrt(3.0) * tanphi / np.sqrt(3.0 + 4.0 * tanphi**2)
    sin3theta = np.sin(3 * theta)
    gtheta = 1 / (1 + Mfc / 6.0 * (sin3theta + sin3theta**2) + (Mfc - Mfo) / Mfo * (1 - sin3theta**2))
    psi = e_n - (e0 - lamdac * (p_n / 100)**ksi)
    eta_p = np.exp(-nb * psi) * gtheta * Mfc
    eta_d = np.exp(nd * psi) * gtheta * Mfc
    return eta_p * p_or_d + eta_d * (1 - p_or_d)

param_constrains2 = [
    (0.8, 1.0), # e0
    (0.0, 1.0), # lambdac
    (0.5, 1.5), # ksi
    (0.5, 2.4), # Mfc
    (0.5, 1.4), # Mfo
    (0.0, 5.0), # nb
    (0.0, 5.0), # nd
]

param_ini2 = [1.0, 0.1, 1.0, 1.5, 1.2, 1.0, 1.0]

# 用一个 Mfc 控制，区分原状和重塑
def eta_func3(x, e0, lamdac, ksi, Mfc, nb_0, nd_0, nb_1, nd_1):
    p_n, e_n, p_or_d, theta, is_remoulded,  = x.T  # 解包输入数组
    nb = is_remoulded * nb_1 + (1.0 - is_remoulded) * nb_0
    nd = is_remoulded * nd_1 + (1.0 - is_remoulded) * nd_0
    sinphi = 3.0 * Mfc / (Mfc + 6.0)
    tanphi = sinphi / np.sqrt(1.0 - sinphi**2)
    Mfo = 2 * np.sqrt(3.0) * tanphi / np.sqrt(3.0 + 4.0 * tanphi**2)
    sin3theta = np.sin(3 * theta)
    gtheta = 1 / (1 + Mfc / 6.0 * (sin3theta + sin3theta**2) + (Mfc - Mfo) / Mfo * (1 - sin3theta**2))
    psi = e_n - (e0 - lamdac * (p_n / 100)**ksi)
    eta_p = np.exp(-nb * psi) * gtheta * Mfc
    eta_d = np.exp(nd * psi) * gtheta * Mfc
    return eta_p * p_or_d + eta_d * (1 - p_or_d)

param_constrains3 = [
    (0.8, 1.1), # e0
    (0.0, 1.0), # lambdac
    (0.7, 1.5), # ksi
    (0.5, 2.4), # Mfc
    # (0.5, 1.4), # Mfo
    (0.0, 5.0), # nb_0
    (0.0, 7.0), # nd_0
    (0.0, 5.0), # nb_1
    (0.0, 7.0), # nd_1
]

param_ini3 = [1.0, 0.1, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0]

def est_cs_para(data_df: pd.DataFrame , est_type: int = 0, max_iter: int = 100000,
                eta_func: Callable = None, param_constrains : List = None, param_ini : List = None):
    '''
    To estimate critical state parameters
    '''
    # match est_type:
    #     case 1:
    #         eta_func = eta_func1
    #         param_constrains = param_constrains1
    #         param_ini = param_ini1
    #     case 2:
    #         eta_func = eta_func2
    #         param_constrains = param_constrains2
    #         param_ini = param_ini2
    #     case 3:
    #         eta_func = eta_func3
    #         param_constrains = param_constrains3
    #         param_ini = param_ini3
    if est_type == 1:
        eta_func = eta_func1
        param_constrains = param_constrains1
        param_ini = param_ini1
        data = data_df[['p','e','p_or_d','theta']].to_numpy()
    elif est_type == 2:
        eta_func = eta_func2
        param_constrains = param_constrains2
        param_ini = param_ini2
        data = data_df[['p','e','p_or_d','theta']].to_numpy()
    elif est_type == 3:
        eta_func = eta_func3
        param_constrains = param_constrains3
        param_ini = param_ini3
        data = data_df[['p','e','p_or_d','theta', 'is_remoulded']].to_numpy()

    z = data_df[['eta']].to_numpy().squeeze()

    popt, pcov = curve_fit(eta_func, data, z, p0 = param_ini, bounds = np.array(param_constrains).T, maxfev = max_iter)
    z_new = eta_func(data, *popt)

    output_df = data_df.copy()
    output_df['eta_est'] = z_new

    return popt, output_df