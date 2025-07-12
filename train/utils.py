import numpy as np

def hausdorff_distance(y_true, y_pred):
    """
    计算批量样本的Hausdorff距离（适合空间谱/多峰回归）
    y_true, y_pred: shape=(batch, n)
    返回: shape=(batch,)
    """
    # 计算每个预测点到最近真实点的距离
    dists = np.abs(y_true[..., None] - y_pred[..., None, :])  # (batch, n, n)
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
    # 假设y_true/y_pred为空间谱，取最大值索引
    true_idx = np.argmax(y_true, axis=-1)
    pred_idx = np.argmax(y_pred, axis=-1)
    success = np.abs(true_idx - pred_idx) <= threshold
    return np.mean(success)
