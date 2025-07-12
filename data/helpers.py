import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal

def generate_ula_steering_vector(theta_deg, N, d=0.5):
    """
    生成均匀线阵的导向矢量
    theta_deg: 入射角度(度)
    N: 阵元数
    d: 阵元间距(波长)
    """
    theta_rad = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * d * n * np.sin(theta_rad))

def generate_signal_samples(N, K, snapshots, SNR_dB, theta_deg):
    """
    生成K个信号源的阵列信号快照
    N: 阵元数
    K: 信号源数
    snapshots: 快拍数
    SNR_dB: 信噪比(dB)
    theta_deg: 信号源角度列表
    """
    # 生成导向矢量
    A = np.array([generate_ula_steering_vector(theta, N) for theta in theta_deg]).T  # N x K
    
    # 生成信号
    S = (np.random.randn(K, snapshots) + 1j * np.random.randn(K, snapshots)) / np.sqrt(2)  # K x snapshots
    
    # 生成噪声
    SNR = 10 ** (SNR_dB / 10)
    sigma_n = np.sqrt(1 / SNR)
    noise = (sigma_n / np.sqrt(2)) * (np.random.randn(N, snapshots) + 1j * np.random.randn(N, snapshots))
    
    # 生成接收信号
    X = A @ S + noise
    
    # 计算协方差矩阵
    R = X @ X.conj().T / snapshots
    
    return X, R

def generate_gaussian_label(angles, angle_grid, sigma=1.0):
    """
    生成高斯型空间谱标签
    angles: 真实角度列表
    angle_grid: 角度网格
    sigma: 高斯宽度
    """
    label = np.zeros(len(angle_grid))
    for angle in angles:
        # 找到最近的网格点
        idx = np.argmin(np.abs(angle_grid - angle))
        # 生成高斯分布
        gaussian = np.exp(-0.5 * ((angle_grid - angle) / sigma) ** 2)
        # 归一化
        gaussian = gaussian / np.max(gaussian)
        # 叠加到标签上
        label = np.maximum(label, gaussian)
    return label

def generate_onehot_label(angles, angle_grid, multi_label=True):
    """
    生成one-hot或multi-label标签
    angles: 真实角度列表
    angle_grid: 角度网格
    multi_label: 是否为多标签
    """
    label = np.zeros(len(angle_grid))
    for angle in angles:
        # 找到最近的网格点
        idx = np.argmin(np.abs(angle_grid - angle))
        label[idx] = 1
    return label

def generate_dataset(N=10, snapshots=256, num_samples=1000, max_targets=3, snr_db=30, 
                    angle_grid=np.linspace(-40, 40, 81), mode='classification', sigma=1.0, save_dir='./data'):
    """
    生成数据集
    N: 阵元数
    snapshots: 快拍数
    num_samples: 样本数
    max_targets: 最大目标数
    snr_db: 信噪比范围
    angle_grid: 角度网格
    mode: 'classification'或'regression'
    sigma: 高斯宽度(回归模式)
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化数据和标签数组
    X_cov_real = np.zeros((num_samples, N, N))
    X_cov_imag = np.zeros((num_samples, N, N))
    if mode == 'classification':
        labels = np.zeros((num_samples, len(angle_grid)))
    else:
        labels = np.zeros((num_samples, len(angle_grid)))
    
    for i in range(num_samples):
        # 随机生成目标数量
        num_targets = np.random.randint(1, max_targets + 1)
        
        # 随机生成角度
        angles = np.random.choice(angle_grid, size=num_targets, replace=False)
        
        # 生成信号
        _, R = generate_signal_samples(N, num_targets, snapshots, snr_db, angles)
        
        # 保存协方差矩阵的实部和虚部
        X_cov_real[i] = np.real(R)
        X_cov_imag[i] = np.imag(R)
        
        # 生成标签
        if mode == 'classification':
            labels[i] = generate_onehot_label(angles, angle_grid, multi_label=True)
        else:
            labels[i] = generate_gaussian_label(angles, angle_grid, sigma=sigma)
    
    # 重塑为(样本数, N, N, 2)形式，方便CNN处理
    X_cov = np.stack((X_cov_real, X_cov_imag), axis=-1)
    
    # 保存数据
    np.save(os.path.join(save_dir, 'Xcov.npy'), X_cov)
    np.save(os.path.join(save_dir, 'label.npy'), labels)
    
    print(f"生成了{num_samples}个样本，保存到{save_dir}")
    return X_cov, labels 