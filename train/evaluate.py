import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import matplotlib
from scipy.signal import find_peaks

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def hausdorff_distance(y_true, y_pred):
    """
    计算两个空间谱的Hausdorff距离
    """
    # 找到非零点
    true_indices = np.where(y_true > 0.1)[0]
    pred_indices = np.where(y_pred > 0.1)[0]
    
    if len(true_indices) == 0 or len(pred_indices) == 0:
        return np.inf
    
    # 计算双向Hausdorff距离
    forward, _, _ = directed_hausdorff(true_indices.reshape(-1, 1), pred_indices.reshape(-1, 1))
    backward, _, _ = directed_hausdorff(pred_indices.reshape(-1, 1), true_indices.reshape(-1, 1))
    
    return max(forward, backward)

def recovery_rate(y_true, y_pred, threshold=0.5, distance_threshold=2):
    """
    计算恢复成功率
    threshold: 检测阈值
    distance_threshold: 允许的最大距离误差
    """
    # 对于分类任务，找到预测值大于阈值的索引
    true_peaks = np.where(y_true > threshold)[0]
    pred_peaks = np.where(y_pred > threshold)[0]
    
    if len(true_peaks) == 0:
        return 0.0
    
    # 计算每个真实峰是否被成功恢复
    recovered = 0
    for true_peak in true_peaks:
        if np.any(np.abs(pred_peaks - true_peak) <= distance_threshold):
            recovered += 1
    
    return recovered / len(true_peaks)

def plot_accuracy_curve(acc_list, save_path):
    plt.figure()
    plt.plot(range(1, len(acc_list)+1), acc_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练过程中的正确率变化')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, X_test, y_test, degs_test=None, task='classification', save_dir='results', num_samples_dict={1:2, 2:2, 3:3}):
    """
    支持按目标数分组采样可视化
    degs_test: 真实DOA角度（shape: N, n_targets）
    num_samples_dict: {目标数: 可视化样本数}
    """
    os.makedirs(save_dir, exist_ok=True)
    model = load_model(model_path)
    y_pred = model.predict(X_test)

    # 计算评估指标
    hausdorff_distances = []
    recovery_rates = []
    
    for i in range(len(X_test)):
        h_dist = hausdorff_distance(y_test[i], y_pred[i])
        if h_dist != np.inf:
            hausdorff_distances.append(h_dist)
        recovery_rates.append(recovery_rate(y_test[i], y_pred[i]))
    
    avg_hausdorff = np.mean(hausdorff_distances)
    avg_recovery = np.mean(recovery_rates)
    
    print(f"平均Hausdorff距离: {avg_hausdorff:.4f}")
    print(f"平均恢复成功率: {avg_recovery:.4f}")
    
    # 可视化：按目标数分组采样
    if degs_test is not None:
        for n_targets, n_vis in num_samples_dict.items():
            idx = [i for i, d in enumerate(degs_test) if len(d)==n_targets]
            if len(idx)==0: continue
            np.random.shuffle(idx)
            for j, i in enumerate(idx[:n_vis]):
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(y_test[i])
                plt.title(f"{n_targets}目标-样本{i+1}-真实标签")
                plt.xlabel("角度索引")
                plt.ylabel("强度")
                plt.subplot(1, 2, 2)
                plt.plot(y_pred[i])
                plt.title(f"{n_targets}目标-样本{i+1}-预测结果")
                plt.xlabel("角度索引")
                plt.ylabel("强度")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{task}_{n_targets}targets_{j+1}.png'))
                plt.close()
    else:
        # 兼容无degs_test时的旧逻辑
        for i in range(min(5, len(X_test))):
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(y_test[i])
            plt.title(f"样本 {i+1} - 真实标签")
            plt.xlabel("角度索引")
            plt.ylabel("强度")
            plt.subplot(1, 2, 2)
            plt.plot(y_pred[i])
            plt.title(f"样本 {i+1} - 预测结果")
            plt.xlabel("角度索引")
            plt.ylabel("强度")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{task}_{i+1}.png'))
            plt.close()

    return avg_hausdorff, avg_recovery

def save_doa_batch_results(X_test, y_test, degs_test, model, num_targets, save_dir, angle_grid=None, threshold=0.5):
    """
    批量DOA与目标数可视化，按目标数分组保存
    X_test: 测试输入
    y_test: 测试标签
    degs_test: 真实DOA角度（shape: N, n_targets）
    model: 已加载模型
    num_targets: 当前处理的目标数（1/2/3）
    save_dir: 保存目录
    angle_grid: 角度网格（如有）
    threshold: 峰值检测阈值
    """
    if model is None:
        print("模型未加载，跳过该轮...")
        return
    os.makedirs(save_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    idx = [i for i, d in enumerate(degs_test) if len(d) == num_targets]
    if len(idx) == 0:
        print(f"无{num_targets}个目标的样本，跳过...")
        return
    true_doa = [degs_test[i] for i in idx]
    pred_doa = []
    true_num = [num_targets for _ in idx]
    pred_num = []
    for i in idx:
        # 峰值检测
        peaks, _ = find_peaks(y_pred[i], height=threshold)
        # 若有angle_grid则转为角度，否则用索引
        if angle_grid is not None:
            pred_doa.append(angle_grid[peaks])
        else:
            pred_doa.append(peaks)
        pred_num.append(len(peaks))
    # 左图：DOA
    plt.figure()
    for j in range(num_targets):
        plt.scatter(range(len(idx)), [d[j] if len(d)>j else np.nan for d in true_doa], label=f"θ{j+1}真值", marker='o')
        plt.scatter(range(len(idx)), [d[j] if len(d)>j else np.nan for d in pred_doa], label=f"θ{j+1}预测", marker='x')
    plt.xlabel("Sample index")
    plt.ylabel("DOA(°)")
    plt.legend()
    plt.title(f"{num_targets}个目标DOA结果")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'doa_{num_targets}_targets.png'))
    plt.close()
    # 右图：目标数
    plt.figure()
    plt.scatter(range(len(idx)), true_num, label="真实目标数", marker='o')
    plt.scatter(range(len(idx)), pred_num, label="预测目标数", marker='x')
    plt.xlabel("Sample index")
    plt.ylabel("目标数")
    plt.legend()
    plt.title(f"{num_targets}个目标数恢复结果")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'num_{num_targets}_targets.png'))
    plt.close()

if __name__ == "__main__":
    # 加载数据
    X_test = np.load('data/test/Xcov.npy')
    y_test = np.load('data/test/label.npy')
    degs_test = np.load('data/test/degs.npy', allow_pickle=True)
    # 假设角度网格
    angle_grid = np.linspace(-40, 40, 81)
    # 加载模型（以回归模型为例，可切换为分类模型）
    model = load_model('checkpoint/regression_best.h5')
    # 轮次与目录映射
    round_settings = [
        (1, 'results/one_target'),
        (2, 'results/two_target'),
        (3, 'results/three_target'),
    ]
    for num_targets, save_dir in round_settings:
        print(f"\n==== 处理{num_targets}个目标样本，结果保存到{save_dir} ====")
        save_doa_batch_results(X_test, y_test, degs_test, model, num_targets, save_dir, angle_grid=angle_grid, threshold=0.5)