# 导入绘图和数据处理相关库
import matplotlib.pyplot as plt  # 用于绘制静态图像
# import hvplot.polars  # 可选的高级可视化库（未使用）
import polars as pl  # 可选的高性能DataFrame库
from pathlib import Path  # 用于处理文件路径

# 构造结果csv文件的路径
path0 = Path(__file__).parent / "phase2_problem2_results" / "phase2_problem2_results_2025-09-28_17-12-33.csv"
path1 = Path(__file__).parent / "phase2_problem2_results" / "phase2_problem2_results_2025-09-29_03-36-52.csv"
path2 = Path(__file__).parent / "phase2_problem2_results" / "phase2_problem2_results_2025-09-30_10-32-00.csv"
path3 = Path(__file__).parent / "phase2_problem2_results" / "phase2_problem2_results_2025-10-19_16-10-04.csv"

# 读取csv数据为Polars DataFrame
df = pl.read_csv(path3)
# 获取所有唯一的lambda参数，并排序
lambdas = df['lambda_val'].unique().sort().to_list()
n = len(lambdas)  # lambda参数的数量

# 创建n个子图，每个lambda一个
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6, n*4), sharex=True)

# 如果只有一个lambda，axes不是列表，需要转为列表
if n == 1:
    axes = [axes]

# 遍历每个lambda值，分别绘制其对应的能量曲线
for ax, lam in zip(axes, lambdas):
    sub_df = df.filter(pl.col('lambda_val') == lam)  # 选取当前lambda的数据

    # 对每种field_type分别绘制能量随h变化的曲线
    for mtype in ['Sz', 'Sy_alternating']:
        mtype_df = sub_df.filter(pl.col('field_type') == mtype)
        ax.plot(mtype_df['h_val'].to_list(), mtype_df['energy'].to_list(), marker='o', label=mtype)
    ax.set_title(f"lambda={lam:.2f}")  # 设置子图标题
    ax.set_ylabel("Ground State Energy")  # y轴标签
    ax.grid(True)  # 显示网格
    ax.legend()  # 显示图例

# 最后一个子图设置x轴标签
axes[-1].set_xlabel("h value")
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示所有子图

# 下面是备用的绘图代码（未使用）
# first_plot=df.plot()
# first_plot.imshow()