import numpy as np
import time
import os
import shutil
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as patches
# import scipy
# from scipy import signal
# from numpy.fft import fft,ifft,fftfreq
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import subprocess

from typing import List, Dict, Tuple, Union, Callable
import multiprocessing

def calculate_n(seq: np.ndarray, tau0: float = 0) -> np.ndarray:
    """
    Calculate the normalized phase (n) for a sequence based on zero-crossings and peak values.

    Parameters:
    - seq (np.ndarray): The input sequence of tau values.
    - tau0: static initial shear stress.

    Returns:
    - seq_n (np.ndarray): The cycle sequence.
    """
    seq = seq - tau0

    if seq[2] < 0:
        seq = -seq

    # Detect zero-crossings (change of sign) in the sequence
    sign_changes = np.diff(np.sign(seq))
    zero_crossings = np.where((sign_changes != 0) & (np.sign(seq[:-1]) != 0))[0] + 1

    # Include the start and end indices of the sequence for segmenting
    segments_indices = np.concatenate(([0], zero_crossings, [len(seq)]))
    
    max_indices = []
    max_values = []

    seq_n = np.zeros(len(seq))

    start, end = 0, len(seq)

    # Process each segment
    for i in range(len(segments_indices) - 1):
        start, end = segments_indices[i], segments_indices[i + 1]
        segment = seq[start:end]
        if segment.size == 0:
            continue
        abs_max_idx = np.argmax(np.abs(segment))
        max_indices.append(start + abs_max_idx)
        max_values.append(segment[abs_max_idx])
        try:
            seq_n[start:end] = np.arcsin(seq[start:end] / abs(segment[abs_max_idx]))
        except:
            print('error in calculating seq_n')
            print(seq_n[start:end])

    # Convert lists to numpy arrays
    max_indices = np.array(max_indices)
    max_values = np.array(max_values)

    # Adjust the last segment's max value if it's less than 90% of the penultimate
    if len(max_values) > 1 and np.abs(max_values[-1]) < 0.9 * np.abs(max_values[-2]):
        max_indices = np.delete(max_indices, -1)
        seq_n[start:end] = np.arcsin(seq[start:end] / abs(max_values[-2]))

    # Generate max points count array
    max_points_counts = np.zeros(len(seq), dtype=int)
    for idx in max_indices:
        max_points_counts[idx:] += 1
    
    seq_n = (seq_n * (-1) ** max_points_counts + max_points_counts * np.pi) / (2 * np.pi)
    
    return seq_n

def run_exe(simu_folder: str, executable_name: str, timeout: float = 120):
    '''
    Run .exe file
    '''
    original_dir = os.getcwd()
    os.chdir(simu_folder)
    command = f'.\\{executable_name}'
    try:
        with subprocess.Popen(command) as process:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        os.chdir(original_dir)

def pp_exe(simu_folder: str, datafile_name: str, cyclic: bool = True):
    '''
    Post Processing for 'stress0.dat' and 'strain0.dat'
    Convert them into a .txt with specified colomns
    b alpha are not included
    '''
    temp_strain = pd.read_table(os.path.join(simu_folder, f"strain0.dat"),header=None)
    temp_stress = pd.read_table(os.path.join(simu_folder, f"stress0.dat"),header=None)

    sigma_z = - temp_stress[8]
    sigma_r = - temp_stress[4]
    sigma_t = - temp_stress[0]
    temp_sign = np.sign(temp_stress.iloc[2,2]) # type: ignore
    sigma_zt = temp_sign * temp_stress[2]
    
    sigma_1 = (sigma_z + sigma_t) / 2 + np.sqrt(((sigma_z - sigma_t) / 2)**2 + sigma_zt**2)
    sigma_2 = sigma_r
    sigma_3 = (sigma_z + sigma_t) / 2 - np.sqrt(((sigma_z - sigma_t) / 2)**2 + sigma_zt**2)
    p = (sigma_1 + sigma_2 + sigma_3) / 3
    q = np.sqrt(0.5 * ((sigma_1 - sigma_2)**2 + (sigma_2 - sigma_3)**2 + (sigma_3 - sigma_1)**2))
    tau = sigma_zt
    
    # for future, zero frac problem?
    # b = (sigma_2 - sigma_3) / (sigma_1 - sigma_3)
    # alpha = 0.5 * np.arctan(2 * sigma_zt / (sigma_z - sigma_t))
    
    epsilon_z =  - temp_strain[8]
    epsilon_r =  - temp_strain[4]
    epsilon_t =  - temp_strain[0]
    temp_sign = np.sign(temp_strain.iloc[2,2]) # type: ignore
    epsilon_zt = temp_sign * temp_strain[2]
    
    e1 = (epsilon_z + epsilon_t) / 2 + np.sqrt(((epsilon_z - epsilon_t) / 2)**2 + (epsilon_zt)**2)
    e2 = epsilon_r
    e3 = (epsilon_z + epsilon_t) / 2 - np.sqrt(((epsilon_z - epsilon_t) / 2)**2 + (epsilon_zt)**2)
    ev = epsilon_z + epsilon_r + epsilon_t
    eq = np.sqrt(2/9 * ((e1 - e2)**2 + (e2 - e3)**2 + (e1 - e3)**2))
    gamma = epsilon_zt * 2

    temp_df = pd.DataFrame({
        'p':p,
        'q':q,
        'tau':tau,
        'e_a': epsilon_z,
        'e_v': ev,
        'e_q': eq,
        'gamma': gamma
    })

    if cyclic:
        temp_tau = temp_df['tau'].to_numpy()
        # 加一个 try 和错误处理，打印名称，给一个应付的结果
        try:
            temp_n = calculate_n(temp_tau)
            temp_df['n'] = temp_n
        except:
            temp_df['n'] = temp_tau * 0.0
            print(f'error in calculating n for {os.path.join(simu_folder, datafile_name)}')
    
    temp_df.to_csv(os.path.join(simu_folder, datafile_name), float_format = '%.5e', index = False)
    
    os.remove(os.path.join(simu_folder, f"strain0.dat"))
    os.remove(os.path.join(simu_folder, f"stress0.dat"))

def clear_exe(simu_folder: str):
    '''
    clear .exe files
    '''
    for filename in os.listdir(simu_folder):
        file_path = os.path.join(simu_folder, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith('.exe'):
                os.remove(file_path)

class ExpTarget:
    def __init__(self, 
                 exp_config: Dict[str, List[List[float]]]):
        """
        Specify a series of experiment
        HCT: hollow cylinder torsion
        Tri: triaxial
    
        Attributes:
            experiment config in form of dict:
            keys:
                'drained_cyclic_HCT'
                'drained_cyclic_HCT_5cyc'
                'undrained_cyclic_HCT'
                'undrained_cyclic_HCT_liqcyc'
                'undrained_cyclic_HCT_5cyc'
            values:
                [[p_in, e_in, csr], ...]
                initial pressure, initial void ratio, cyclic stress ratio

                or

                [[p_in, e_in, csr, max_iter], ...]
                initial pressure, initial void ratio, cyclic stress ratio, max iteration steps

            keys:
                'drained_mono_Tri'
                'undrained_mono_Tri'
            values:
                [[p_in, e_in, e_a], ...]
                initial pressure, initial void ratio, axial strain
            
            keys:
                'drained_mono_HCT'
                'undrained_mono_HCT'
            values:
                [[p_in, e_in, e_s], ...]
                initial pressure, initial void ratio, shear strain

            To be implemented:

            keys:
                'undrained_cyclic_Tri'
            values:
                [[p_in, e_in, csr], ...]
                initial pressure, initial void ratio, cyclic stress ratio (dev stress / p)

            keys:
                'undrained_cyclic_HCT_with_initial_shear'
            values:
                [[p_in, e_in, csr, ssr]]
                initial pressure, initial void ratio, cyclic stress ratio, static stress ratio

        """

        self.exp_config = exp_config

    def ini_simu(self,
                 exe_folder = 'executables/',
                 simu_folder = 'simu temp/'):
        if os.path.exists(simu_folder):
            shutil.rmtree(simu_folder)
        os.makedirs(simu_folder)

        if os.path.exists(exe_folder):
            for exp_type in self.exp_config:
                shutil.copy(os.path.join(exe_folder, f'{exp_type}.exe'),os.path.join(simu_folder, f'{exp_type}.exe'))
            # shutil.copytree(exe_folder, simu_folder)
        else:
            print('Source folder not exists')
    
    def conduct_simu(self,
                     para,
                     simu_folder = 'simu temp/',
                     t_lim = 120,
                     if_clear_exe = True):

        # set cycliq parameters
        temp_line = ' '.join([str(temp_var) for temp_var in para])
        with open(os.path.join(simu_folder, 'modelPara.dat'),'w') as temp_file:
            temp_file.writelines(temp_line)

        for exp_type, exp_specs in self.exp_config.items():
            match exp_type:
                case (
                    'drained_cyclic_HCT' |
                    'drained_cyclic_HCT_5cyc' |
                    'undrained_cyclic_HCT' |
                    'undrained_cyclic_HCT_liqcyc' |
                    'undrained_cyclic_HCT_5cyc'
                ):
                    for temp_i in range(len(exp_specs)):
                        if len(exp_specs[temp_i]) == 3:
                            exp_specs[temp_i].append(80000) # max_iter is 80000 by default

                    for p_in, e_in, csr, max_iter in exp_specs:
                        # write experiment specs
                        stress_initial = [- p_in if i in [0, 4, 8] else 0.0 for i in range(9)]
                        file_paths = {
                            'initialStress.dat': ' '.join(f'{x:.3f}' for x in stress_initial),
                            'initialVoidRatio.dat': str(e_in),
                            'cyclicShearStress.dat': str(p_in * csr),
                            'maxIter.dat': str(max_iter)
                        }
                        for file_name, content in file_paths.items():
                            with open(os.path.join(simu_folder, file_name), 'w') as file:
                                file.write(content)
                        
                        # run exe
                        run_exe(simu_folder, f'{exp_type}.exe', t_lim)
    
                        # data process
                        datafile_name = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_csr_{csr:.3f}'.replace('.', '_')+'.txt'
                        pp_exe(simu_folder, datafile_name, cyclic = True)


                case (
                    'undrained_cyclic_Tri' 
                ):
                    pass

                case (
                    'drained_mono_Tri' |
                    'undrained_mono_Tri'
                ):
                    for p_in, e_in, e_a in exp_specs:
                        # write experiment specs
                        stress_initial = [0.0 for _ in range(9)]
                        stress_initial[0] = -p_in + 0.1
                        stress_initial[4] = -p_in + 0.1
                        stress_initial[8] = -p_in - 0.2
                        file_paths = {
                            'initialStress.dat': ' '.join(f'{x:.3f}' for x in stress_initial),
                            'initialVoidRatio.dat': str(e_in),
                            'axialStrain.dat': str(e_a)
                        }
                        for file_name, content in file_paths.items():
                            with open(os.path.join(simu_folder, file_name), 'w') as file:
                                file.write(content)
                        
                        # run exe
                        run_exe(simu_folder, f'{exp_type}.exe', t_lim)
    
                        # data process
                        datafile_name = f'{exp_type.split("_")[0]}_Tri_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                        pp_exe(simu_folder, datafile_name, cyclic = False)

                case (
                    'drained_mono_HCT' |
                    'undrained_mono_HCT'
                ):
                    for p_in, e_in, e_s in exp_specs:
                        # write experiment specs
                        stress_initial = [- p_in if i in [0, 4, 8] else 0.0 for i in range(9)]
                        file_paths = {
                            'initialStress.dat': ' '.join(f'{x:.3f}' for x in stress_initial),
                            'initialVoidRatio.dat': str(e_in),
                            'shearStrain.dat': str(e_s)
                        }
                        for file_name, content in file_paths.items():
                            with open(os.path.join(simu_folder, file_name), 'w') as file:
                                file.write(content)
                        
                        # run exe
                        run_exe(simu_folder, f'{exp_type}.exe', t_lim)
    
                        # data process
                        datafile_name = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                        pp_exe(simu_folder, datafile_name, cyclic = False)

                case (
                    'undrained_cyclic_HCT_with_initial_shear'
                ):
                    pass
        
        if if_clear_exe:
            clear_exe(simu_folder)

class EvalCycliq:
    def __init__(self, exp_config: ExpTarget, exp_folder: str, eval_folder: str,
                 eval_config: Dict[str, Callable], 
                 exe_folder: str = 'executables', N_PR: int = 12):
        self.exp_config = exp_config # 试验的配置
        self.exp_folder = exp_folder # 试验数据的位置
        self.eval_folder = eval_folder # optimazition 数据的目录
        self.eval_config = eval_config # 试验类型 + 函数名，传递评估函数
        self.exe_folder = exe_folder # exe 文件的位置
        self.N_PR = N_PR # 并行数

        # if not os.path.exists(eval_folder):
        #     os.makedirs(eval_folder)

        if not os.path.exists(exe_folder):
            print('vital error: executable files lacks')

        if not os.path.exists(exp_folder):
            print('vital error: experiment data lacks')
        else:
            for exp_type, exp_specs in self.exp_config.exp_config.items():
                match exp_type:
                    case (
                        'drained_cyclic_HCT' |
                        'drained_cyclic_HCT_5cyc' |
                        'undrained_cyclic_HCT' |
                        'undrained_cyclic_HCT_liqcyc' |
                        'undrained_cyclic_HCT_5cyc'
                    ):
                        for temp_i in range(len(exp_specs)):
                            if len(exp_specs[temp_i]) == 3:
                                exp_specs[temp_i].append(80000) # max_iter is 80000 by default
    
                        for p_in, e_in, csr, max_iter in exp_specs:
                            temp_filename = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_csr_{csr:.3f}'.replace('.', '_')+'.txt'
                            if not os.path.exists(os.path.join(exp_folder, temp_filename)):
                                print(f'vital error: experiment data lacks {temp_filename}')
                    case (
                        'undrained_cyclic_Tri' 
                    ):
                        pass
    
                    case (
                        'drained_mono_Tri' |
                        'undrained_mono_Tri'
                    ):
                        for p_in, e_in, e_a in exp_specs:
                            temp_filename = f'{exp_type.split("_")[0]}_Tri_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                            if not os.path.exists(os.path.join(exp_folder, temp_filename)):
                                print(f'vital error: experiment data lacks {temp_filename}')
    
                    case (
                        'drained_mono_HCT' |
                        'undrained_mono_HCT'
                    ):
                        for p_in, e_in, e_s in exp_specs:
                            temp_filename = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                            if not os.path.exists(os.path.join(exp_folder, temp_filename)):
                                print(f'vital error: experiment data lacks {temp_filename}')
    
                    case (
                        'undrained_cyclic_HCT_with_initial_shear'
                    ):
                        pass
    
    def eval_obj_1(self, para):
        '''
           to be constructed
        '''
        scores = []

        temp_pid = os.getpid()
        temp_time0 = time.time()
        temp_name = f'temp_simu_{str(int(temp_time0))}_{str(temp_pid)}/'
        temp_folder = os.path.join(self.eval_folder, temp_name)

        while os.path.exists(temp_folder):
            temp_folder = temp_folder + '_2'
        
        self.exp_config.ini_simu(exe_folder = self.exe_folder, simu_folder = temp_folder)
        self.exp_config.conduct_simu(para, simu_folder = temp_folder)
        for exp_type, exp_specs in self.exp_config.exp_config.items():
            match exp_type:
                case (
                    'drained_cyclic_HCT' |
                    'drained_cyclic_HCT_5cyc' |
                    'undrained_cyclic_HCT' |
                    'undrained_cyclic_HCT_liqcyc' |
                    'undrained_cyclic_HCT_5cyc'
                ):
                    for temp_i in range(len(exp_specs)):
                        if len(exp_specs[temp_i]) == 3:
                            exp_specs[temp_i].append(80000) # max_iter is 80000 by default

                    for p_in, e_in, csr, max_iter in exp_specs:
                        temp_filename = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_csr_{csr:.3f}'.replace('.', '_')+'.txt'
                        scores.append(self.eval_config[exp_type](os.path.join(self.exp_folder, temp_filename), os.path.join(temp_folder, temp_filename)))
                        
                case (
                    'undrained_cyclic_Tri' 
                ):
                    pass

                case (
                    'drained_mono_Tri' |
                    'undrained_mono_Tri'
                ):
                    for p_in, e_in, e_a in exp_specs:
                        temp_filename = f'{exp_type.split("_")[0]}_Tri_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                        scores.append(self.eval_config[exp_type](os.path.join(self.exp_folder, temp_filename), os.path.join(temp_folder, temp_filename)))

                case (
                    'drained_mono_HCT' |
                    'undrained_mono_HCT'
                ):
                    for p_in, e_in, e_s in exp_specs:
                        temp_filename = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                        scores.append(self.eval_config[exp_type](os.path.join(self.exp_folder, temp_filename), os.path.join(temp_folder, temp_filename)))

                case (
                    'undrained_cyclic_HCT_with_initial_shear'
                ):
                    pass

        final_score = np.sqrt(np.mean(np.square(scores)))
        
        # return final_score, temp_folder
        return final_score

    def eval_batch(self, para_batch):
        pool = multiprocessing.Pool(processes = self.N_PR)
        obj_batch = pool.map(self.eval_obj_1, para_batch)
        pool.close()
        pool.join()
        return obj_batch
    
    def eval_existed_folder(self, temp_folder):
        scores = []
        for exp_type, exp_specs in self.exp_config.exp_config.items():
            match exp_type:
                case (
                    'drained_cyclic_HCT' |
                    'drained_cyclic_HCT_5cyc' |
                    'undrained_cyclic_HCT' |
                    'undrained_cyclic_HCT_liqcyc' |
                    'undrained_cyclic_HCT_5cyc'
                ):
                    for temp_i in range(len(exp_specs)):
                        if len(exp_specs[temp_i]) == 3:
                            exp_specs[temp_i].append(80000) # max_iter is 80000 by default

                    for p_in, e_in, csr, max_iter in exp_specs:
                        temp_filename = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_csr_{csr:.3f}'.replace('.', '_')+'.txt'
                        scores.append(self.eval_config[exp_type](os.path.join(self.exp_folder, temp_filename), os.path.join(temp_folder, temp_filename)))
                        
                case (
                    'undrained_cyclic_Tri' 
                ):
                    pass

                case (
                    'drained_mono_Tri' |
                    'undrained_mono_Tri'
                ):
                    for p_in, e_in, e_a in exp_specs:
                        temp_filename = f'{exp_type.split("_")[0]}_Tri_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                        scores.append(self.eval_config[exp_type](os.path.join(self.exp_folder, temp_filename), os.path.join(temp_folder, temp_filename)))

                case (
                    'drained_mono_HCT' |
                    'undrained_mono_HCT'
                ):
                    for p_in, e_in, e_s in exp_specs:
                        temp_filename = f'{exp_type.split("_")[0]}_HCT_p_{p_in:.0f}_ein_{e_in:.3f}_mono'.replace('.', '_')+'.txt'
                        scores.append(self.eval_config[exp_type](os.path.join(self.exp_folder, temp_filename), os.path.join(temp_folder, temp_filename)))

                case (
                    'undrained_cyclic_HCT_with_initial_shear'
                ):
                    pass
        final_score = np.sqrt(np.mean(np.square(scores)))
        # return final_score, temp_folder
        return final_score
    
    def eval_existed_folder_batch(self, folder_batch):
        pool = multiprocessing.Pool(processes = self.N_PR)
        obj_batch = pool.map(self.eval_existed_folder, folder_batch)
        pool.close()
        pool.join()
        return obj_batch
    
    def show_result(self, result_folder, png_name):
        pass

class ParaTrans:
    def __init__(self, csv_path):
        self.keys = ['G0', 'Kappa', 'h', 'M', 'dre1', 'dre2', 'rdr', 'eta', 'dir', 'lambdac', 'ksi', 'e0', 'nb', 'nd', 'beta1', 'beta2']
        self.params = pd.read_csv(csv_path)
        
        if not set(self.keys).issubset(set(self.params['var'])):
            raise ValueError("CSV lacks some necessary parameters.")

        self.params['range'] = self.params['max'] - self.params['min']
        self.para_temp = dict(zip(self.params['var'], self.params['min']))
        self.mappable_params = self.params[self.params['range'] != 0].reset_index(drop=True)
        self.x_len = len(self.mappable_params)

    def x2para(self, x):
        if len(x) != self.x_len:
            raise ValueError('Incorrect length of input vector x.')
        
        tempdict_para = self.para_temp.copy()
        for i, row in self.mappable_params.iterrows():
            tempdict_para[row['var']] = x[i] * row['range'] + row['min']
        
        return [tempdict_para[key] for key in self.keys]

    def para2x(self, para):
        para_dict = dict(zip(self.keys, para))
        return [(para_dict[row['var']] - row['min']) / row['range'] for _, row in self.mappable_params.iterrows()]

    def x2para_nl(self, x):
        # non-linear mapping
        if len(x) != self.x_len:
            raise ValueError('Incorrect length of input vector x.')
        
        tempdict_para = self.para_temp.copy()
        for i, row in self.mappable_params.iterrows():
            if row['min'] > 0:
                tempdict_para[row['var']] = row['min'] * np.exp(x[i] * (np.log(row['max']) - np.log(row['min'])))
            else:
                raise ValueError('non positive min value in input params')
        
        return [tempdict_para[key] for key in self.keys]

    def para2x_nl(self, para):
         # non-linear mapping
        para_dict = dict(zip(self.keys, para))
        return [(np.log(para_dict[row['var']]) - np.log(row['min'])) /  (np.log(row['max']) - np.log(row['min'])) for _, row in self.mappable_params.iterrows()]