'''
plot different type of soil tests

'drained_cyclic_HCT'
'undrained_cyclic_HCT'
'drained_cyclic_Tri'
'undrained_cyclic_Tri'

'drained_mono_HCT'
'undrained_mono_HCT'
'drained_mono_Tri'
'undrained_mono_Tri'
'''

from typing import List, Dict, Tuple, Union, Callable
from matplotlib.axes import Axes
import pandas as pd

def plot_by_list(data_df: pd.DataFrame, plot_config: List[List[Axes, str, str]], color: str, label: str = '', lw :float = 0.75):
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

def compare_undrained_cyc_HCT(png_name, 
                              df_test, color_test = '#920A2F', label_test = 'Test',
                              df_simu = None, color_simu = '#456990', label_simu = 'Simu',
                              lw = 0.75):
    pass

def compare_drained_cyc_HCT():
    pass

def compare_undrained_mono_HCT():
    pass

def compare_drained_mono_HCT():
    pass

def compare_undrained_cyc_Tri():
    pass

def compare_drained_cyc_Tri():
    pass

def compare_undrained_mono_Tri():
    pass

def compare_drained_mono_Tri():
    pass
        