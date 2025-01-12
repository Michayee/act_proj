import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from PIL import Image as im

import os

import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Union, Callable
from matplotlib.axes import Axes

# --------------- font related start ---------------
# font_path = "C:/Windows/font_others/times_simsun.ttf"
# font_path = "C:/Windows/font_others/arial_simhei.ttf"
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)
config = {
    'font.family': 'serif',
    # 'font.serif': [prop.get_name(), 'DejaVu Serif'],
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': [prop.get_name(), 'DejaVu Sans'],
    'font.size': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'backend': 'Agg',
}
rcParams.update(config)
# --------------- font related end ---------------

import re
def extract_ein(s: str):
    """
    从特定格式的字符串中提取数值。
    
    参数:
    s (str): 输入字符串，例如 'p_200_ein_0_752'。
    
    返回:
    dict: 包含提取的数值的字典。
    """
    # 定义要查找的模式：'p_数字'、'ein_数字'、'cyc_数字'
    # 注意数字可能包含小数点
    pattern = r'ein_(\d+_\d+)'
    
    # 使用正则表达式搜索匹配项
    m = re.search(pattern, s)
    if m:
        # 将匹配的字符串转换为相应的数值类型
        ein = float(m.group(1).replace('_', '.'))
        
        # 返回一个字典，包含提取的数值
        return ein
    else:
        # 如果没有找到匹配项，返回 None 或抛出异常
        return None

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

def plot_by_list(data_df: pd.DataFrame, plot_config: List[Tuple[Axes, str, str]], color: str, label: str = '', lw :float = 0.75):
    label_dict = {
        'p': r'$p \ (kPa)$',
        'q': r'$q \ (kPa)$',
        'tau': r'$\tau \ (kPa)$',
        'e_a': r'$\epsilon_{axial} \ (\%)$',
        'e_v': r'$\epsilon_{v} \ (\%)$',
        'e_q': r'$\epsilon_{q} \ (\%)$',
        'gamma': r'$\gamma \ (\%)$',
        'n': r'$N_{cycles}$',
        'e': r'$e$',
        'ru':r'$r_u$'
    }
    for ax_i, var1, var2 in plot_config:
        data_x = data_df[var2]
        data_y = data_df[var1]
        if var2 in ['e_a','e_v','gamma','e_q']:
            data_x = data_x * 100
        if var1 in ['e_a','e_v','gamma','e_q']:
            data_y = data_y * 100
        ax_i.plot(data_x, data_y, linewidth = lw, label = label,color = color)
        ax_i.set_xlabel(label_dict[var2])
        ax_i.set_ylabel(label_dict[var1])

def plot_undrained_cyclic_HCT(file_test, file_simu, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.45
    nrows = 2
    ncols = 3
    top = 0.90
    bottom = 0.15
    left = 0.10
    right = 0.98
    
    hspace = 0.5
    wspace = 0.4
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, top=top, bottom=bottom, right=right, left=left, wspace = wspace, hspace=hspace)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[0, 2])
    ax4  = fig.add_subplot(gs[1, 0])
    ax5  = fig.add_subplot(gs[1, 1])
    ax6  = fig.add_subplot(gs[1, 2])
    
    data_test = pd.read_csv(file_test)
    data_simu = pd.read_csv(file_simu)
    
    data_test['DA'] = calculate_DA(data_test['gamma'].tolist())
    data_simu['DA'] = calculate_DA(data_simu['gamma'].tolist())
    
    data_test = data_test[data_test['DA'] < 0.075]
    data_simu = data_simu[data_simu['DA'] < 0.075]
    
    gamma_lim = 4.0
    p_lim = data_test['p'].max() * 1.1
    tau_lim = data_test['tau'].max() * 1.1
    n_lim = max(data_test['n'].max(), data_simu['n'].max()) * 1.1
    
    ax1.set_xlim([-1,n_lim])
    ax4.set_xlim([-1,n_lim])
    ax1.set_ylim([-gamma_lim,gamma_lim])
    ax4.set_ylim([-gamma_lim,gamma_lim])
    
    ax2.set_xlim([-1,n_lim])
    ax5.set_xlim([-1,n_lim])
    ax2.set_ylim([0,p_lim])
    ax5.set_ylim([0,p_lim])
    
    ax3.set_xlim([-gamma_lim,gamma_lim])
    ax6.set_xlim([-gamma_lim,gamma_lim])
    ax3.set_ylim([-tau_lim,tau_lim])
    ax6.set_ylim([-tau_lim,tau_lim])
    
    plot_by_list(data_test,[[ax1,'gamma', 'n'],[ax2, 'p', 'n'],[ax3, 'tau','gamma']], color = '#920A2F')
    plot_by_list(data_simu,[[ax4,'gamma', 'n'],[ax5, 'p', 'n'],[ax6, 'tau','gamma']], color = '#00449C')
    
    fig.text(0.5, 0.98, file_test.split('\\')[-1].replace('.txt',''), fontsize=9, va='top', ha='center')
    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_drained_cyclic_HCT(file_test, file_simu, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.25
    nrows = 1
    ncols = 3
    top = 0.80
    bottom = 0.3
    left = 0.10
    right = 0.98
    hspace = 0.5
    wspace = 0.4
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, top=top, bottom=bottom, right=right, left=left, wspace = wspace, hspace=hspace)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[0, 2])
    
    data_test = pd.read_csv(file_test)
    data_simu = pd.read_csv(file_simu)

    data_test = data_test[data_test['n'] < 20]
    data_simu = data_simu[data_simu['n'] < 20]
    
    ein = extract_ein(file_test)
    data_test['e'] = - data_test['e_v'] * (1 + ein) + ein
    data_simu['e'] = - data_simu['e_v'] * (1 + ein) + ein
    
    plot_by_list(data_test,[[ax1,'gamma','n'], [ax2,'e_v','n']], color = '#920A2F')
    plot_by_list(data_simu,[[ax1,'gamma','n'], [ax2,'e_v','n']], color = '#00449C')
    
    data_test = data_test[data_test['n'] < 2]
    data_simu = data_simu[data_simu['n'] < 2]
    
    plot_by_list(data_test,[[ax3,'tau','gamma']], color = '#920A2F')
    plot_by_list(data_simu,[[ax3,'tau','gamma']], color = '#00449C')

    fig.text(0.5, 0.93, file_test.split('\\')[-1].replace('.txt',''), fontsize=9, va='top', ha='center')
    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_drained_mono_Tri(file_test, file_simu, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.25
    nrows = 1
    ncols = 3
    top = 0.80
    bottom = 0.3
    left = 0.10
    right = 0.98
    hspace = 0.5
    wspace = 0.4
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, top=top, bottom=bottom, right=right, left=left, wspace = wspace, hspace=hspace)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[0, 2])
    
    data_test = pd.read_csv(file_test)
    data_simu = pd.read_csv(file_simu)
    
    ein = extract_ein(file_test)
    data_test['e'] = - data_test['e_v'] * (1 + ein) + ein
    data_simu['e'] = - data_simu['e_v'] * (1 + ein) + ein
    
    plot_by_list(data_test,[[ax1,'q','e_a'], [ax2,'e_v','e_a'], [ax3,'e','p']], color = '#920A2F')
    plot_by_list(data_simu,[[ax1,'q','e_a'], [ax2,'e_v','e_a'], [ax3,'e','p']], color = '#00449C')
    
    fig.text(0.5, 0.93, file_test.split('\\')[-1].replace('.txt',''), fontsize=9, va='top', ha='center')
    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_undrained_mono_Tri(file_test, file_simu, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.25
    nrows = 1
    ncols = 3
    top = 0.80
    bottom = 0.3
    left = 0.10
    right = 0.98
    hspace = 0.5
    wspace = 0.4
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, top=top, bottom=bottom, right=right, left=left, wspace = wspace, hspace=hspace)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[0, 2])
    
    data_test = pd.read_csv(file_test)
    data_simu = pd.read_csv(file_simu)
    
    # ein = extract_ein(file_test)
    # data_test['e'] = - data_test['e_v'] * (1 + ein) + ein
    # data_simu['e'] = - data_simu['e_v'] * (1 + ein) + ein
    
    plot_by_list(data_test,[[ax1,'q','e_a'], [ax2,'p','e_a'], [ax3,'q','p']], color = '#920A2F')
    plot_by_list(data_simu,[[ax1,'q','e_a'], [ax2,'p','e_a'], [ax3,'q','p']], color = '#00449C')
    
    fig.text(0.5, 0.93, file_test.split('\\')[-1].replace('.txt',''), fontsize=9, va='top', ha='center')
    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_drained_mono_HCT(file_test, file_simu, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.25
    nrows = 1
    ncols = 3
    top = 0.80
    bottom = 0.3
    left = 0.10
    right = 0.98
    hspace = 0.5
    wspace = 0.4
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, top=top, bottom=bottom, right=right, left=left, wspace = wspace, hspace=hspace)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[0, 2])
    
    data_test = pd.read_csv(file_test)
    data_simu = pd.read_csv(file_simu)
    
    ein = extract_ein(file_test)
    data_test['e'] = - data_test['e_v'] * (1 + ein) + ein
    data_simu['e'] = - data_simu['e_v'] * (1 + ein) + ein
    
    plot_by_list(data_test,[[ax1,'tau','gamma'], [ax2,'e_v','gamma']], color = '#920A2F')
    plot_by_list(data_simu,[[ax1,'tau','gamma'], [ax2,'e_v','gamma']], color = '#00449C')
  
    fig.text(0.5, 0.93, file_test.split('\\')[-1].replace('.txt',''), fontsize=9, va='top', ha='center')
    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_undrained_mono_HCT(file_test, file_simu, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.25
    nrows = 1
    ncols = 3
    top = 0.80
    bottom = 0.3
    left = 0.10
    right = 0.98
    hspace = 0.5
    wspace = 0.4
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, top=top, bottom=bottom, right=right, left=left, wspace = wspace, hspace=hspace)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    # ax3  = fig.add_subplot(gs[0, 2])
    
    data_test = pd.read_csv(file_test)
    data_simu = pd.read_csv(file_simu)
    
    # ein = extract_ein(file_test)
    # data_test['e'] = - data_test['e_v'] * (1 + ein) + ein
    # data_simu['e'] = - data_simu['e_v'] * (1 + ein) + ein
    
    plot_by_list(data_test,[[ax1,'tau','gamma'], [ax2,'p','gamma']], color = '#920A2F')
    plot_by_list(data_simu,[[ax1,'tau','gamma'], [ax2,'p','gamma']], color = '#00449C')
    
    fig.text(0.5, 0.93, file_test.split('\\')[-1].replace('.txt',''), fontsize=9, va='top', ha='center')
    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_para_table(temp_ax,temp_para):
    temp_para_str = [f'{var:.4g}' for var in temp_para]
    if len(temp_para_str) == 16: 
        cell_text = [[r'$G_0$', r'$\kappa$', r'$h$', r'$M$', r'$d_{re,1}$', r'$d_{re,2}$', r'$\gamma_{d,r}$', r'$\alpha$'],
                    temp_para_str[:8],
                    [r'$d_{ir}$', r'$\lambda_c$', r'$\xi$', r'$e_0$', r'$n^p$', r'$n^d$', r'$\beta_1$', r'$\beta_2$'],
                    temp_para_str[8:]]
    else:
        cell_text = [[r'$G_0$', r'$\kappa$', r'$h$', r'$M$', r'$d_{re,1}$', r'$d_{re,2}$', r'$\gamma_{d,r}$', r'$\alpha$'],
            temp_para_str[:8],
            [r'$d_{ir}$', r'$\lambda_c$', r'$\xi$', r'$e_0$', r'$n^p$', r'$n^d$', '', ''],
            temp_para_str[8:]+['','']]
    temp_ax.axis('tight')
    temp_ax.axis('off')
    table = temp_ax.table(cellText=cell_text, colLabels=None, loc='center', cellLoc='center')
    # 调整表格布局
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    linewidth = 0.75  # 设定线宽
    for key, cell in table.get_celld().items():
        cell.set_linewidth(linewidth)

def plot_para(para_list, png_name = None):
    img_width = 15 # in centimeters
    img_ratio = 1/0.25
    
    fig = plt.figure(figsize = (img_width/2.54,img_width/img_ratio/2.54))
    ax1 = fig.add_subplot(111)
    plot_para_table(ax1,para_list)

    if png_name is not None:
        fig.savefig(png_name, dpi = 300)

    plt.close()

def plot_compare_exp_config(exp_config: Dict, test_folder, simu_folder, para_list):
    plot_para(para_list, png_name = os.path.join(simu_folder, 'para.png'))
    for exp_type, exp_specs in exp_config.items():
        if exp_type in ['undrained_cyclic_HCT',
                        'undrained_cyclic_HCT_liqcyc',
                        'undrained_cyclic_HCT_5cyc'
                        ]:
            for exp_spec in exp_specs:
                p_in, e_in, csr = exp_spec[:3]
                datafile_name = f'{exp_type.split('_')[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_csr_{csr:.3f}'.replace('.', '_')+'.txt'
                file_test = os.path.join(test_folder, datafile_name)
                file_simu = os.path.join(simu_folder, datafile_name)
                png_name = os.path.join(simu_folder ,  datafile_name.replace('.txt', '.png'))
                plot_undrained_cyclic_HCT(file_test, file_simu, png_name =  png_name)

        elif exp_type in ['drained_cyclic_HCT',
                          'drained_cyclic_HCT_5cyc',
                          ]:
            for exp_spec in exp_specs:
                p_in, e_in, csr = exp_spec[:3]
                datafile_name = f'{exp_type.split('_')[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_csr_{csr:.3f}'.replace('.', '_')+'.txt'
                file_test = os.path.join(test_folder, datafile_name)
                file_simu = os.path.join(simu_folder, datafile_name)
                png_name = os.path.join(simu_folder ,  datafile_name.replace('.txt', '.png'))
                plot_drained_cyclic_HCT(file_test, file_simu, png_name =  png_name)

        elif exp_type in ['undrained_cyclic_Tri',
                        ]:
            pass

        elif exp_type in ['drained_mono_Tri',
                          ]:
            for p_in, e_in, e_a in exp_specs:
                datafile_name = f'{exp_type.split('_')[0]}_Tri_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                file_test = os.path.join(test_folder, datafile_name)
                file_simu = os.path.join(simu_folder, datafile_name)
                png_name = os.path.join(simu_folder ,  datafile_name.replace('.txt', '.png'))
                plot_drained_mono_Tri(file_test, file_simu, png_name = png_name)

        elif exp_type in ['undrained_mono_Tri',
                          ]:
            for p_in, e_in, e_a in exp_specs:
                datafile_name = f'{exp_type.split('_')[0]}_Tri_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                file_test = os.path.join(test_folder, datafile_name)
                file_simu = os.path.join(simu_folder, datafile_name)
                png_name = os.path.join(simu_folder ,  datafile_name.replace('.txt', '.png'))
                plot_undrained_mono_Tri(file_test, file_simu, png_name = png_name)

        elif exp_type in ['drained_mono_HCT',
                          ]:
            for p_in, e_in, e_s in exp_specs:
                datafile_name = f'{exp_type.split('_')[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                file_test = os.path.join(test_folder, datafile_name)
                file_simu = os.path.join(simu_folder, datafile_name)
                png_name = os.path.join(simu_folder ,  datafile_name.replace('.txt', '.png'))
                plot_drained_mono_HCT(file_test, file_simu, png_name =png_name)

        elif exp_type in ['undrained_mono_HCT',
                          ]:
            for p_in, e_in, e_s in exp_specs:
                datafile_name = f'{exp_type.split('_')[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                file_test = os.path.join(test_folder, datafile_name)
                file_simu = os.path.join(simu_folder, datafile_name)
                png_name = os.path.join(simu_folder ,  datafile_name.replace('.txt', '.png'))
                plot_undrained_mono_HCT(file_test, file_simu, png_name = png_name)

        elif exp_type in ['undrained_cyclic_HCT_with_initial_shear',
                          ]:
            pass