from src.target_cycliq import ParaTrans, ExpTarget, EvalCycliq
from src.eval_sedata import eval_undrained_mono_torsion
from src.opt_pso2 import pso
import time, os, shutil
import pandas as pd

if __name__ == "__main__":

    N_PR = 12
    
    # 目标试验的定义
    exp_folder='target_exp/coral/'
    # exp_dict = {
    #     'undrained_cyclic_HCT':[
    #         [100, 0.647, 0.127],
    #         [100, 0.647, 0.147],
    #         [100, 0.619, 0.174],
    #         [100, 0.612, 0.199],
    #         [100, 0.593, 0.200],
    #         [100, 0.587, 0.248],
    #         [100, 0.591, 0.272],
    #         [100, 0.558, 0.296],
    #         [100, 0.553, 0.391]
    #     ],
    #     'drained_cyclic_HCT':[
    #         [200, 0.588, 0.170],
    #         [200, 0.589, 0.250]
    #     ]
    # }
    
    exp_dict = {
        'undrained_mono_HCT':[
            [60, 0.761, 0.150],
            [100, 0.772, 0.170],
        ],
    }
    
    # 参数搜索空间
    paras_file = 'para_table.csv'
    
    # 目标试验的评估方法
    eval_dict = {
        'undrained_mono_HCT': eval_undrained_mono_torsion
    }
    
    paras_config = ParaTrans(paras_file)
    paras_len = paras_config.x_len
    exp_config = ExpTarget(exp_dict)

    # 过程文件写入的位置
    temp_time = str(int(time.time()))
    eval_folder = f'test_eval/test_{temp_time}/'

    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    # 最终定义好的目标函数
    eval_cycliq = EvalCycliq(exp_config = exp_config,
                             exp_folder = exp_folder,
                             eval_folder = eval_folder,
                             eval_config = eval_dict, 
                             exe_folder = 'executables/',
                             N_PR = N_PR)
    

    # 搜索 + 后处理的程序
    input_file = None
    results_file = eval_folder + f'coral_PSO_{int(time.time())}.csv'
    
    nofparticle = 12
    nofgen = 5

    def func_eval_batch_PSO(swarm):
        paras = [ paras_config.x2para(x) for x in swarm ]
        return eval_cycliq.eval_batch(paras)

    pso(num_particles = nofparticle,
        num_dimensions = paras_len,
        num_iterations = nofgen,
        input_file = input_file,
        results_file = results_file,
        func_eval_batch = func_eval_batch_PSO)
    
    # 输入文件拷贝一份
    if input_file is not None and os.path.exists(input_file):
        shutil.copy(input_file, eval_folder)
    shutil.copy(__file__, eval_folder)
    
    # 后处理补全结果
    temp_df = pd.read_csv(results_file)
    temp_df[paras_config.keys] = temp_df.apply(lambda row: paras_config.x2para([row[f'x_{i}'] for i in range(paras_config.x_len)]), axis=1, result_type='expand')
    temp_df.to_csv(results_file, float_format='%.5e', index=False)

    # 找每一代效果最好的结果，绘制出来