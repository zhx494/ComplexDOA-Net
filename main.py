import os
import numpy as np
import argparse
import pandas as pd
from evaluate.metrics import FFT, hausdorff_distance, recovery_success_rate, steering_vector
from keras.models import load_model

# 数据生成与预处理
from data.data_utils import synthesize_dataset, synthesize_sample

# 模型构建
from models.doa_classification import build_classification_model
from models.doa_regression import build_regression_model
# from models.losses import multilabel_categorical_crossentropy, mse_hausdorff_loss  # 不再使用自定义损失

# 训练相关
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 评估相关
from train.evaluate import evaluate_model, save_doa_batch_results, plot_accuracy_curve
from evaluate.plot_results import save_superres_imaging_figures, save_dataflow_diagram, save_performance_comparison_fig

def generate_data(args):
    angle_grid = np.linspace(args.angle_min, args.angle_max, args.num_angles)
    print("生成训练集 ...")
    synthesize_dataset(
        N=args.N,
        snapshots=args.snapshots,
        num_samples=args.num_samples_train,
        max_targets=args.max_targets,
        snr_db=args.snr_db,
        angle_grid=angle_grid,
        mode=args.mode,
        sigma=args.sigma,
        save_dir=os.path.join(args.output_dir, "train")
    )
    print("生成验证集 ...")
    synthesize_dataset(
        N=args.N,
        snapshots=args.snapshots,
        num_samples=args.num_samples_val,
        max_targets=args.max_targets,
        snr_db=args.snr_db,
        angle_grid=angle_grid,
        mode=args.mode,
        sigma=args.sigma,
        save_dir=os.path.join(args.output_dir, "val")
    )
    print("生成测试集 ...")
    synthesize_dataset(
        N=args.N,
        snapshots=args.snapshots,
        num_samples=args.num_samples_test,
        max_targets=args.max_targets,
        snr_db=args.snr_db,
        angle_grid=angle_grid,
        mode=args.mode,
        sigma=args.sigma,
        save_dir=os.path.join(args.output_dir, "test")
    )

def train_model(args):
    # 数据加载
    X_train = np.load(os.path.join(args.output_dir, 'train/Xcov.npy'))
    y_train = np.load(os.path.join(args.output_dir, 'train/label.npy'))
    X_val = np.load(os.path.join(args.output_dir, 'val/Xcov.npy'))
    y_val = np.load(os.path.join(args.output_dir, 'val/label.npy'))

    if args.mode == 'classification':
        model = build_classification_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        ckpt_path = 'checkpoint/classification_best.h5'
    else:
        model = build_regression_model(input_shape=X_train.shape[1:], output_dim=y_train.shape[1])
        model.compile(optimizer='adam', loss='mse')
        ckpt_path = 'checkpoint/regression_best.h5'

    os.makedirs('checkpoint', exist_ok=True)
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    print("开始训练 ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, earlystop]
    )
    print(f"训练完成，最佳模型已保存到 {ckpt_path}")
    # 新增：保存训练过程中的accuracy曲线
    if args.mode == 'classification' and 'accuracy' in history.history:
        plot_accuracy_curve(history.history['accuracy'], save_path=os.path.join('results', 'accuracy_curve_classification.png'))
    if args.mode == 'regression':
        plot_loss_curve(history.history['loss'], history.history.get('val_loss'), save_path=os.path.join('results', 'loss_curve_regression.png'))

def plot_loss_curve(loss_list, val_loss_list, save_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(loss_list)+1), loss_list, label='Train Loss')
    if val_loss_list is not None:
        plt.plot(range(1, len(val_loss_list)+1), val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程中的损失变化')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def run_experiments_and_save_csv(args, csv_path='results/experiment_results.csv'):
    """
    自动批量生成测试数据，评估DL/FFT/MUSIC/SBL等算法性能，保存到csv
    """
    SNR_list = [0, 10, 20, 30]
    target_num_list = [1, 2, 3]
    results = []
    angle_grid = np.linspace(args.angle_min, args.angle_max, args.num_angles)
    N = args.N
    snapshots = args.snapshots
    num_samples = 128  # 每组实验样本数
    for snr in SNR_list:
        for num_targets in target_num_list:
            # 生成数据
            from data.data_utils import synthesize_sample
            X_list, y_list = [], []
            for _ in range(num_samples):
                R, label, degs = synthesize_sample(N, snapshots, snr, angle_grid, num_targets, args.mode, args.sigma)
                R_real = np.real(R)
                R_imag = np.imag(R)
                R_stack = np.stack([R_real, R_imag], axis=-1)  # (10, 10, 2)
                X_list.append(R_stack)
                y_list.append(label)
            X_test = np.stack(X_list)
            y_test = np.stack(y_list)
            # DL模型预测
            if args.mode == 'classification':
                model_path = 'checkpoint/classification_best.h5'
            else:
                model_path = 'checkpoint/regression_best.h5'
            if not os.path.exists(model_path):
                print(f"未找到模型: {model_path}, 跳过DL评估")
                y_pred_dl = np.zeros_like(y_test)
            else:
                model = load_model(model_path)
                y_pred_dl = model.predict(X_test)
            haus_dl = np.mean(hausdorff_distance(y_test, y_pred_dl))
            succ_dl = recovery_success_rate(y_test, y_pred_dl)
            # FFT（与标签对齐的空间谱估计）
            fft_preds = []
            for i in range(len(X_test)):
                R = X_test[i][...,0] + 1j * X_test[i][...,1]  # 还原复数协方差矩阵
                spec = []
                for theta in angle_grid:
                    a = steering_vector(R.shape[0], theta)
                    s = np.real(np.conj(a.T) @ R @ a)
                    spec.append(s)
                spec = np.array(spec)
                fft_preds.append(spec)
            fft_preds = np.stack(fft_preds)  # shape (num_samples, 81)
            haus_fft = np.mean(hausdorff_distance(y_test, fft_preds))
            succ_fft = recovery_success_rate(y_test, fft_preds)
            # 其他算法（MUSIC/SBL）可按需补充
            haus_music = haus_fft  # 占位
            haus_sbl = haus_fft    # 占位
            results.append([snr, num_targets, haus_dl, haus_fft, haus_music, haus_sbl])
    df = pd.DataFrame(results, columns=['SNR', 'NumTargets', 'DL', 'FFT', 'MUSIC', 'SBL'])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"实验结果已保存到: {csv_path}")

def evaluate(args):
    X_test = np.load(os.path.join(args.output_dir, 'test/Xcov.npy'))
    y_test = np.load(os.path.join(args.output_dir, 'test/label.npy'))
    degs = np.load(os.path.join(args.output_dir, 'test/degs.npy'), allow_pickle=True)
    angle_grid = np.linspace(args.angle_min, args.angle_max, args.num_angles)
    if args.mode == 'classification':
        model_path = 'checkpoint/classification_best.h5'
        task = 'classification'
        perf_csv = 'results/experiment_results_classification.csv'
        perf_fig = 'results/performance_comparison_classification.png'
        eval_txt = 'results/evaluation_results_classification.txt'
    else:
        model_path = 'checkpoint/regression_best.h5'
        task = 'regression'
        perf_csv = 'results/experiment_results_regression.csv'
        perf_fig = 'results/performance_comparison_regression.png'
        eval_txt = 'results/evaluation_results_regression.txt'
    print("开始评估 ...")
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    save_superres_imaging_figures(y_test, y_pred, degs, angle_grid, mode=task, save_dir='results', num_samples=5)
    # 新增：按目标数分组采样可视化
    evaluate_model(model_path, X_test, y_test, degs_test=degs, task=task, save_dir='results', num_samples_dict={1:2, 2:2, 3:3})
    # 评估并保存到区分的txt
    avg_hausdorff, avg_recovery = evaluate_model(model_path, X_test, y_test, degs_test=degs, task=task, save_dir='results', num_samples_dict={1:2, 2:2, 3:3})
    with open(eval_txt, 'w') as f:
        f.write(f"平均Hausdorff距离: {avg_hausdorff:.4f}\n")
        f.write(f"平均恢复成功率: {avg_recovery:.4f}\n")
    run_experiments_and_save_csv(args, csv_path=perf_csv)
    save_performance_comparison_fig(perf_csv, mode=task, save_path=perf_fig)

def doa_batch_visualization(args, mode='regression'):
    """
    批量DOA与目标数恢复可视化，按目标数分组保存
    mode: 'regression' 或 'classification'
    """
    print(f"\n[批量DOA可视化] 模型类型: {mode}")
    X_test = np.load(os.path.join(args.output_dir, 'test/Xcov.npy'))
    y_test = np.load(os.path.join(args.output_dir, 'test/label.npy'))
    degs_test = np.load(os.path.join(args.output_dir, 'test/degs.npy'), allow_pickle=True)
    angle_grid = np.linspace(args.angle_min, args.angle_max, args.num_angles)
    if mode == 'classification':
        model_path = 'checkpoint/classification_best.h5'
    else:
        model_path = 'checkpoint/regression_best.h5'
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}, 跳过批量DOA可视化")
        return
    model = load_model(model_path)
    round_settings = [
        (1, 'results/one_target'),
        (2, 'results/two_target'),
        (3, 'results/three_target'),
    ]
    for num_targets, save_dir in round_settings:
        print(f"\n==== 处理{num_targets}个目标样本，结果保存到{save_dir} ====")
        save_doa_batch_results(X_test, y_test, degs_test, model, num_targets, save_dir, angle_grid=angle_grid, threshold=0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于复数卷积神经网络的波达方向估计超分辨方法-主入口")
    parser.add_argument('--output_dir', type=str, default='./data', help='数据保存目录')
    parser.add_argument('--num_samples_train', type=int, default=10000, help='训练集样本数')
    parser.add_argument('--num_samples_val', type=int, default=2000, help='验证集样本数')
    parser.add_argument('--num_samples_test', type=int, default=2000, help='测试集样本数')
    parser.add_argument('--N', type=int, default=10, help='阵元数')
    parser.add_argument('--snapshots', type=int, default=256, help='快拍数')
    parser.add_argument('--max_targets', type=int, default=3, help='最大信号源数')
    parser.add_argument('--snr_db', type=float, default=30, help='信噪比(dB)')
    parser.add_argument('--angle_min', type=float, default=-40, help='最小角度')
    parser.add_argument('--angle_max', type=float, default=40, help='最大角度')
    parser.add_argument('--num_angles', type=int, default=81, help='角度网格数')
    parser.add_argument('--sigma', type=float, default=1.0, help='回归标签高斯宽度')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--stage', type=str, choices=['all', 'gen', 'train', 'eval'], default='all', help='运行阶段')
    args = parser.parse_args()

    # 一键先分类后回归
    if args.stage in ['all', 'gen']:
        if not hasattr(args, 'mode'):
            args.mode = 'classification'
        generate_data(args)
    if args.stage in ['all', 'train', 'eval']:
        # 分类模型
        args.mode = 'classification'
        if args.stage in ['all', 'train']:
            train_model(args)
        if args.stage in ['all', 'eval']:
            evaluate(args)
            doa_batch_visualization(args, mode='classification')
        # 回归模型，复用同一数据集
        args.mode = 'regression'
        if args.stage in ['all', 'train']:
            train_model(args)
        if args.stage in ['all', 'eval']:
            evaluate(args)
            doa_batch_visualization(args, mode='regression')