import os
import numpy as np
import argparse

def array_steering_vec(N, deg):
    d = 0.5  # 阵元间距（单位：波长）
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    n = np.arange(N).reshape(-1, 1)
    theta = np.deg2rad(deg)
    phases = k * d * n * np.sin(theta)
    return np.exp(1j * phases)

def synthesize_complex_signal(N=10, snapshots=256, snr_db=10, angles=None):
    if angles is None:
        angles = [0]
    K = len(angles)
    S = (np.random.randn(K, snapshots) + 1j * np.random.randn(K, snapshots)) / np.sqrt(2)
    A = array_steering_vec(N, np.array(angles))
    X = A @ S
    signal_power = np.mean(np.abs(X)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = (np.random.randn(N, snapshots) + 1j * np.random.randn(N, snapshots)) / np.sqrt(2) * np.sqrt(noise_power)
    return X + noise

def calc_cov_matrix(X):
    return (X @ X.conj().T) / X.shape[1]

def make_class_label(angles, angle_grid):
    lbl = np.zeros(len(angle_grid), dtype=np.float32)
    for d in angles:
        idx = np.argmin(np.abs(angle_grid - d))
        lbl[idx] = 1.0
    return lbl

def make_reg_label(angles, angle_grid, sigma=1.0):
    lbl = np.zeros(len(angle_grid), dtype=np.float32)
    for d in angles:
        lbl += np.exp(-0.5 * ((angle_grid - d) / sigma) ** 2)
    lbl = lbl / lbl.max()  # 归一化到[0,1]
    return lbl

def synthesize_sample(N, snapshots, snr_db, angle_grid, max_targets, mode='classification', sigma=1.0):
    n_targets = np.random.randint(1, max_targets+1)
    angles = np.random.choice(angle_grid, size=n_targets, replace=False)
    X = synthesize_complex_signal(N, snapshots, snr_db, angles)
    cov_mat = calc_cov_matrix(X)
    if mode == 'classification':
        lbl = make_class_label(angles, angle_grid)
    else:
        lbl = make_reg_label(angles, angle_grid, sigma)
    return cov_mat, lbl, angles

def save_numpy(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)

def synthesize_dataset(
    N, snapshots, num_samples, max_targets, snr_db, angle_grid, mode, sigma, save_dir
):
    real_covs = []
    imag_covs = []
    lbls = []
    angle_list = []
    for _ in range(num_samples):
        cov_mat, lbl, angles = synthesize_sample(
            N, snapshots, snr_db, angle_grid, max_targets, mode, sigma
        )
        real_covs.append(np.real(cov_mat))
        imag_covs.append(np.imag(cov_mat))
        lbls.append(lbl)
        angle_list.append(angles)
    real_covs = np.array(real_covs)
    imag_covs = np.array(imag_covs)
    X_covs = np.stack([real_covs, imag_covs], axis=-1)
    lbls = np.stack(lbls)
    angle_list = np.array(angle_list, dtype=object)
    os.makedirs(save_dir, exist_ok=True)
    save_numpy(X_covs, os.path.join(save_dir, "Xcov.npy"))
    save_numpy(lbls, os.path.join(save_dir, "label.npy"))
    save_numpy(angle_list, os.path.join(save_dir, "degs.npy"))
    print(f"Saved: {save_dir}/Xcov.npy, label.npy, degs.npy")
    return X_covs, lbls, angle_list

def main_data_gen(args):
    angle_grid = np.linspace(args.angle_min, args.angle_max, args.num_angles)
    print("Generating training data ...")
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
    print("Generating test data ...")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset for DOA super-resolution.')
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save data')
    parser.add_argument('--num_samples_train', type=int, default=100000, help='Number of training samples')
    parser.add_argument('--num_samples_test', type=int, default=1024, help='Number of test samples')
    parser.add_argument('--N', type=int, default=10, help='Number of antenna elements')
    parser.add_argument('--snapshots', type=int, default=256, help='Number of snapshots per sample')
    parser.add_argument('--max_targets', type=int, default=3, help='Maximum number of targets per sample')
    parser.add_argument('--snr_db', type=float, default=30, help='SNR in dB')
    parser.add_argument('--angle_min', type=float, default=-40, help='Minimum angle (deg)')
    parser.add_argument('--angle_max', type=float, default=40, help='Maximum angle (deg)')
    parser.add_argument('--num_angles', type=int, default=81, help='Number of angle grid points')
    parser.add_argument('--mode', type=str, choices=['classification', 'regression'], default='classification', help='Label mode')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for regression label (Gaussian width)')
    args = parser.parse_args()
    main_data_gen(args)
