from datetime import datetime
from pathlib import Path
from typing import Optional,Literal

import numpy as np
import polars as pl
import submitit

from core_module.core import (
    Parameters,
    Equation22Hamiltonian, # z方向磁场 - phase II
    Equation23Hamiltonian, # y方向交替磁场 - phase I
    run_dmrg_optimized,
)
def create_tasks()-> pl.DataFrame:
    """
    生成最小任务集：[ (lambda, h, field_type, base_params) ]
    """
    base_params = {
        'L': 96,
        'precision': 'fast',
        # 'precision': 'supercomputer',
        'theta': 0.55 * np.pi,
        'phi': 0.15 * np.pi,
        'boundary': 'periodic',
        'conserve': None,
    }
    lambda_values = np.linspace(0, 1, 10 + 1)
    h_values = np.linspace(0, 0.001, 5 + 1)
    tasks:list[dict]=[]

    for lam in lambda_values:
        for h_val in h_values:
            tasks.append({'lambda_val': lam, 'h_val': h_val, 'field_type': 'Sz', **base_params})
            tasks.append({'lambda_val': lam, 'h_val': h_val, 'field_type': 'Sy', **base_params}) # Sy_alternating
    df_temp_tasks=pl.DataFrame(tasks)
    cols=[
        'L',
        'theta',
        'phi',
        'boundary',
        'conserve',
        'precision',
        'lambda_val',
        'h_val',
        'field_type',
    ]
    return df_temp_tasks.select(cols)

def calculate_single_task(task: dict) -> float:
    try:
        params = Parameters(
            L=task['L'],
            J_prime=task.get('lambda_val', 0),
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
        # 精细调节dmrg_config参数
        ################################################
        dmrg_config1={
            "trunc_params": {
                "chi_max": 100, # 最大保留的bond dimension
                "svd_min": 1e-13, # 最小奇异值截断
                "trunc_cut": 1e-8, # 截断阈值
            },
            "mixer": True, #
            "max_sweeps": 15, # 扫描次数
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
        dmrg_config={
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



    executor = submitit.AutoExecutor(folder=Path(__file__).parent/f"{mode}_logs",cluster=mode)
    if mode == 'debug':
        pass
    elif mode == 'local':
        executor.update_parameters(
            timeout_min=60*24*10,
            cpus_per_task=8,
            mem_gb=4,  # 每个任务使用 4 GB 内存
            tasks_per_node=1,#?  # 每个节点运行 1 个任务
            # max_workers=5,
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
    df_tasks = create_tasks()

    # df_results = process_dataframe(df_tasks, mode='debug')
    df_results = process_dataframe(df_tasks, mode='local')
    # df_results = process_dataframe(df_tasks, mode='slurm')

    ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir=Path(__file__).parent/"phase2_problem2_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file=out_dir/f"phase2_problem2_results_{ts}.csv"

    df_results.write_csv(out_file)
    print(f"Results saved to {out_file}")