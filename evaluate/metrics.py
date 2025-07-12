import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ========================
# 传统算法评估接口
# ========================
def steering_vector(N, deg):
    d = 0.5
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    n = np.arange(N).reshape(-1, 1)
    theta = np.deg2rad(deg)
    phases = k * d * n * np.sin(theta)
    return np.exp(1j * phases)

def FFT(signal, num_antennas=10, ang_min=-30, ang_max=31, ang_step=1):
    ang_list = np.arange(ang_min, ang_max, ang_step)
    a_theta = steering_vector(num_antennas, ang_list).conj()
    AH = a_theta.T
    spec = np.abs(AH @ signal / num_antennas).squeeze()
    return ang_list, spec

def IAA(y, Niter=15):
    N = y.shape[0]
    ang_list = np.arange(-90, 91, 1)
    A = steering_vector(N, ang_list)
    AH = A.conj().T
    N, K = A.shape
    y = y.squeeze()
    A = A
    AH = AH
    Pk = (AH @ y / N) ** 2
    P = np.diag(Pk)
    R = A @ P @ AH
    for _ in range(Niter):
        R += 0e-3 * np.eye(N)
        ak_R = AH @ np.linalg.pinv(R)
        T = ak_R @ y
        B = ak_R @ A
        b = np.diag(B)
        sk = T / np.abs(b)
        Pk = np.abs(sk) ** 2
        P = np.diag(Pk)
        R = A @ P @ A.conj().T
    spec = Pk[60:121]  # -30:1:30
    ang_list = np.arange(-30, 31, 1)
    return ang_list, spec

def estimate_doa(ang_list, spec, scale=0.7):
    max_height = np.max(spec)
    min_peak_height = (scale * max_height)
    peaks, properties = find_peaks(spec, height=min_peak_height)
    sorted_indices = np.argsort(properties['peak_heights'])[::-1]
    sorted_peaks = peaks[sorted_indices]
    doa = ang_list[sorted_peaks]
    return doa

# ========================
# 评估指标
# ========================
def hausdorff_distance(y_true, y_pred):
    """
    计算批量样本的Hausdorff距离（适合空间谱/多峰回归）
    y_true, y_pred: shape=(batch, n)
    返回: shape=(batch,)
    """
    dists = np.abs(y_true[..., None] - y_pred[..., None, :])
    forward = np.max(np.min(dists, axis=-1), axis=-1)
    backward = np.max(np.min(dists, axis=-2), axis=-1)
    return forward + backward

def recovery_success_rate(y_true, y_pred, threshold=0.1):
    """
    计算恢复成功率：预测峰值与真实峰值距离小于阈值视为成功
    y_true, y_pred: shape=(batch, n)
    threshold: 成功判定阈值
    返回: 成功率（0~1）
    """
    true_idx = np.argmax(y_true, axis=-1)
    pred_idx = np.argmax(y_pred, axis=-1)
    success = np.abs(true_idx - pred_idx) <= threshold
    return np.mean(success)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ========================
# 可视化工具
# ========================
def plot_spectrum(y_true, y_pred, idx=0, save_path=None):
    plt.figure(figsize=(8,4))
    plt.plot(y_true[idx], label='True')
    plt.plot(y_pred[idx], label='Predicted')
    plt.legend()
    plt.title(f'Sample {idx} Spectrum')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_metric_curve(x, y, xlabel, ylabel, title, save_path=None):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()