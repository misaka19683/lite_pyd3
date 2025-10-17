from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import polars as pl
import submitit

from core_module.core import (
    Parameters,
    Equation22Hamiltonian, # z方向磁场 - phase II
    Equation23Hamiltonian, # y方向交替磁场 - phase I
    run_dmrg_optimized,
)

task_paras1=[0,0,'Sz']
task_paras2=[0,0,'Sy']

task_paras3=[0.1,0,'Sz']
task_paras4=[0.1,0,'Sy']

task_paras5=[0.2,0,'Sz']
task_paras6=[0.2,0,'Sy']

task_paras7=[0.3,0,'Sz']
task_paras8=[0.3,0,'Sy']

task_paras9=[0.4,0,'Sz']
task_paras10=[0.4,0,'Sy']

task_paras11=[0.5,0,'Sz']
task_paras12=[0.5,0,'Sy']

l,h,f=task_paras4
single_task = {
    'L':8, #debug 检测小系统是否厄米对称
    # 'L': 96,
    'theta': 0.55 * np.pi,
    'phi': 0.15 * np.pi,
    # 'boundary': 'open',# debug 使用，检查厄米不对称是否是由于边界引起的
    'boundary': 'periodic',
    'conserve': None,
    # 'precision': 'fast',
    'lambda_val': l,
    'h_val': h,
    'field_type': f,
}

def calculate_single_task(task: dict) -> float:
    try:
        params = Parameters(
            L=task['L'],
            J_prime=task['lambda_val'],
            theta=task['theta'],
            phi=task['phi'],
            h=task['h_val'],
            boundary=task['boundary'],
            conserve=task['conserve'],
        )
        if task['field_type'] == 'Sz':
            model = Equation22Hamiltonian(params)
        elif task['field_type'] == 'Sy':
            model = Equation23Hamiltonian(params)
        else:
            raise ValueError(f"Unknown field_type: {task['field_type']}")



        # assert model.H_MPO.is_equal(model.H_MPO.dagger())
        if model.H_MPO.is_equal(model.H_MPO.dagger()):
            print("H_MPO is symmetric.")
        else:
            raise ValueError("H_MPO is not symmetric.")

        # print(model.H_MPO.get_W())


        # 精细调节dmrg_config参数

        ################################################
        dmrg_config={
            "trunc_params": {
                "chi_max": 100, # 最大保留的bond dimension
                "svd_min": 1e-13, # 最小奇异值截断
                "trunc_cut": 1e-8, # 截断阈值
            },
            "mixer": True, #
            "max_sweeps": 20, # 扫描次数
            "chi_list": {
                0: 20,
                5: 50,
                10: 100
            },
            ############### 控制精度 #################
            # 参考itensors，目前只关注能量的收敛
            "max_E_err": 1e-10,  # 能量截断  # 控制物理量的精度
            # "max_S_err": 1e-8,   # 熵截断  # 控制纠缠信息的精度
        }
        dmrg_config1={
            "trunc_params": {
                "chi_max": 20, # 最大保留的bond dimension
                "svd_min": 1e-13, # 最小奇异值截断
                "trunc_cut": 1e-5, # 截断阈值
            },
            "mixer": True, #
            "max_sweeps": 15, # 扫描次数
            # "chi_list": {
            #     0: 20,
            #     # 5: 50,
            #     # 10: 100
            # },
            "max_E_err": 1e-10,  # 能量截断  # 控制物理量的精度
        }


        #####################################################
        # 计算基态能量
        # energy, _ = run_dmrg_optimized(model, precision=task['precision'])
        energy, _ = run_dmrg_optimized(model, dmrg_params=dmrg_config)
        return energy

    except Exception as e:
        print(f"Error in task {task}: {e}")
        return np.nan

def main():
    energy = calculate_single_task(single_task)
    print(f"energy: {energy}")

if __name__ == '__main__':
    main()