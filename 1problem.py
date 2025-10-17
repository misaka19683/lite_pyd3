# 目前使用Equation22和Equation23来分别表示z方向和y方向交替磁场。
# 预期结果是基态能量对z方向磁场的响应比对y方向磁场的响应要大

"""
Phase 2 Problem 1: 磁场效应分析
使用Equation22和Equation23分别实现z方向磁场和y方向交替磁场，
使用Parameters中的h属性来标记磁场大小，
使用run_dmrg_optimized来计算基态能量
条件: L=96, PBC (周期性边界条件)
"""

from pathlib import Path
from datetime import datetime
from typing import Optional,Literal

import submitit
import polars as pl
import numpy as np

from core_module.core import (
    Parameters,
    Equation22Hamiltonian,
    Equation23Hamiltonian,
    run_dmrg_optimized,
)

def create_tasks()->pl.DataFrame:
    base_params1 = {
        "L": 96,
        "J_prime": 0.0,  # λ = 0
        "theta": 0.55 * np.pi,
        "phi": 0.15 * np.pi,
        "boundary": "periodic",
        "conserve": None,
    }
    base_params2={
        "L": 96,
        "J_prime": 0.0,  # λ = 0
        "theta": 0.55 * np.pi,
        "phi": 0.30 * np.pi,
        "boundary": "periodic",
        "conserve": None,
    }
    h_values=np.linspace(0,0.002, 10+1)
    # h_values=np.array([0.0,2e-4,4e-4,6e-4,8e-4,10e-4,12e-4,14e-4,16e-4,18e-4,2e-3])
    tasks:list[dict]=[]
    for h_val in h_values:
        tasks.append({'h_val':h_val,'field_type':'z',**base_params1,'params_set':1})
        tasks.append({'h_val':h_val,'field_type':'y',**base_params1,'params_set':1})
        tasks.append({'h_val':h_val,'field_type':'z',**base_params2,'params_set':2})
        tasks.append({'h_val':h_val,'field_type':'y',**base_params2,'params_set':2})
    df_temp_tasks=pl.DataFrame(tasks)
    cols=[
        'L',
        'theta',
        'phi',
        'boundary',
        'conserve',
        'h_val',
        'field_type',
        'params_set'
    ]
    return df_temp_tasks.select(cols)

def calculate_single_task(task:dict) -> float :
    try:
        params=Parameters(
            L=task['L'],
            J_prime=0, # lambda的参数选取为0
            theta=task['theta'],
            phi=task['phi'],
            h=task['h_val'],
            boundary=task['boundary'],
            conserve=task['conserve'],
        )
        if task['field_type']=='z':
            model = Equation22Hamiltonian(params)
        elif task['field_type']=='y':
            model = Equation23Hamiltonian(params)
        else:
            raise ValueError(f"未知的 field_type: {task['field_type']}")

        # 检查哈密顿量是否厄米共轭
        if model.H_MPO.is_equal(model.H_MPO.dagger()):
            print("H_MPO is symmetric.")
        else:
            raise ValueError("H_MPO is not symmetric.")

        # 精细调节dmrg_config参数
        dmrg_config = {
            "trunc_params": {
                "chi_max": 20,  # 最大保留的bond dimension
                "svd_min": 1e-13,  # 最小奇异值截断
                "trunc_cut": 1e-5,  # 截断阈值
            },
            "mixer": True,  #
            "max_sweeps": 15,  # 扫描次数
            # "chi_list": {0: 20, 5: 50, 10: 100},
            "max_E_err": 1e-10,  # 能量截断  # 控制物理量的精度
        }
        energy, _ = run_dmrg_optimized(model=model, dmrg_params=dmrg_config)
        return energy
    except Exception as e:
        print(f"Error in task {task}: {e}")
        return np.nan

def process_dataframe(df:pl.DataFrame,mode:Optional[Literal['local','slurm','debug']])->pl.DataFrame:
    """
    使用 submitit 处理 DataFrame 中的任务
    :param df: 输入的 DataFrame
    :param mode: 处理模式 ('local', 'slurm' 或 'debug')
    :return: 处理后的 DataFrame
    """
    start_time=datetime.now()
    if mode is None:
        mode='debug'

    if mode not in ['local','slurm','debug']:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'local', 'slurm', or 'debug'.")


    executor = submitit.AutoExecutor(folder=Path(__file__).parent/f"1_problem_{mode}_logs",cluster=mode)
    if mode == 'debug':
        pass
    elif mode == 'local':
        executor.update_parameters(
            timeout_min=60,
            cpus_per_task=8,
            mem_gb=4,  # 每个任务使用 4 GB 内存
            tasks_per_node=1,#?  # 每个节点运行 1 个任务
        )
    elif mode == 'slurm':
        executor.update_parameters(
            timeout_min=60*24,  # 1 天
            slurm_partition="xhhetdnormal",  # 使用 雄衡 分区
            # slurm_gres="gpu:1",  # 每个任务请求 1 个 GPU
            cpus_per_task=32,  # 每个任务使用 32 个 CPU 核心
            mem_gb=200,  # 每个任务使用 200 GB 内存
            # nodes=1,  # 每个任务使用 1 个节点
            tasks_per_node=2,  # 每个节点运行 1 个任务
            # max_num_timeout=60*24,  # 最大超时重试次数
            slurm_array_parallelism=20,  # 限制同时运行的数组任务数量
        )

    # for row in df.iter_rows(named=True):
        # job=executor.submit(calculate_single_task, row)
    futures=[]
    with executor.batch():
        for row in df.iter_rows(named=True):
            futures.append(executor.submit(calculate_single_task, row))

    results=[f.result() for f in futures]

    df = df.with_columns(pl.Series("energy", results))

    print(f"Finished in {datetime.now()-start_time}")
    return df
if __name__ == '__main__':
    df_tasks=create_tasks()
    df_results=process_dataframe(df_tasks,mode='debug')

    ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir=Path(__file__).parent/"1_phase2_problem1_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file=out_dir/f"phase2_problem1_results_{ts}.csv"

    df_results.write_csv(out_file)
    print(f"Results saved to {out_file}")