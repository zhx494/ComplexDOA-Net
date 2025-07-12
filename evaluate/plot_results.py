import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

def plot_spectrum(y_true, y_pred, idx=0, save_path=None, degs=None, angle_grid=None, title_prefix=None):
    plt.figure(figsize=(8,4))
    plt.plot(y_true[idx], label='True')
    plt.plot(y_pred[idx], label='Predicted')
    # 标注真实信号源位置
    if degs is not None and angle_grid is not None:
        for d in degs[idx]:
            plt.axvline(x=np.argmin(np.abs(angle_grid - d)), color='r', linestyle='--', alpha=0.5, label='True Source' if d==degs[idx][0] else None)
    if title_prefix:
        plt.title(f'{title_prefix} Sample {idx} Spectrum')
    else:
        plt.title(f'Sample {idx} Spectrum')
    plt.xlabel('Angle Index')
    plt.ylabel('Intensity')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_superres_imaging_figures(y_true, y_pred, degs, angle_grid, mode, save_dir, num_samples=5):
    """
    批量绘制超分辨率成像效果图，区分分类与回归模型
    y_true, y_pred: shape=(N, n_angles)
    degs: shape=(N, n_targets) or object array
    angle_grid: 角度网格
    mode: 'classification' or 'regression'
    save_dir: 保存目录
    num_samples: 绘制样本数
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(y_true))):
        save_path = os.path.join(save_dir, f'superres_{mode}_sample{i+1}.png')
        plot_spectrum(y_true, y_pred, idx=i, save_path=save_path, degs=degs, angle_grid=angle_grid, title_prefix='超分辨率成像')

def plot_metric_curve(x, y, xlabel, ylabel, title, save_path=None):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_dataflow_diagram(save_path='results/dataflow.png'):
    """
    保存实验数据处理流程图到指定路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    # 流程节点
    boxes = ["数据生成", "数据预处理", "模型训练", "模型评估", "结果可视化", "保存到results目录"]
    y = 0.5
    for i, text in enumerate(boxes):
        ax.add_patch(mpatches.FancyBboxPatch((i*1.5, y), 1.3, 0.5, boxstyle="round,pad=0.1", fc="#e0e0e0"))
        ax.text(i*1.5+0.65, y+0.25, text, ha='center', va='center', fontsize=12)
        if i < len(boxes)-1:
            ax.annotate('', xy=(i*1.5+1.3, y+0.25), xytext=((i+1)*1.5, y+0.25),
                        arrowprops=dict(arrowstyle="->", lw=2))
    plt.xlim(-0.5, len(boxes)*1.5-0.5)
    plt.ylim(0, 1.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_performance_comparison_fig(csv_path, mode, save_path):
    """
    读取实验结果csv，绘制并保存算法性能对比图
    mode: 'classification' or 'regression'，仅影响标题
    save_path: 保存路径
    """
    if not os.path.exists(csv_path):
        print(f"未找到实验结果文件: {csv_path}")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取实验结果文件失败: {csv_path}, 错误: {e}")
        return
    if not isinstance(df, pd.DataFrame) or 'NumTargets' not in df.columns:
        print(f"实验结果文件内容无效: {csv_path}")
        return
    plt.figure(figsize=(8,6))
    num_targets_list = sorted(pd.Series(df['NumTargets']).dropna().astype(int).unique())
    for num_targets in num_targets_list:
        sub = df[df['NumTargets'] == num_targets]
        plt.plot(sub['SNR'], sub['DL'], marker='o', label=f'DL, Targets={num_targets}')
        plt.plot(sub['SNR'], sub['FFT'], marker='s', label=f'FFT, Targets={num_targets}')
        plt.plot(sub['SNR'], sub['MUSIC'], marker='^', label=f'MUSIC, Targets={num_targets}')
        plt.plot(sub['SNR'], sub['SBL'], marker='x', label=f'SBL, Targets={num_targets}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Hausdorff Distance')
    plt.title(f'算法性能对比 ({mode})')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 下方为实验性能曲线的旧代码，暂时注释避免linter报错
# df = pd.read_csv('fig/experiment_results.csv')
# for num_targets in df['NumTargets'].unique():
#     sub = df[df['NumTargets'] == num_targets]
#     plt.plot(sub['SNR'], sub['DL'], label='Deep Learning')
#     plt.plot(sub['SNR'], sub['FFT'], label='FFT')
#     plt.plot(sub['SNR'], sub['MUSIC'], label='MUSIC')
#     plt.plot(sub['SNR'], sub['SBL'], label='SBL')
#     plt.xlabel('SNR (dB)')
#     plt.ylabel('Hausdorff Distance')
#     plt.title(f'Num Targets: {num_targets}')
#     plt.legend()
#     plt.savefig(f'fig/curve_targets_{num_targets}.png')
#     plt.close()