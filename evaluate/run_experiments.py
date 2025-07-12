import numpy as np
from data.dataset_gen import generate_dataset
from train.train_classification import train_and_evaluate as train_dl
from data.eval_fun import fft_doa, music_doa, sbl_doa
from evaluate.metrics import hausdorff_distance, recovery_success_rate

SNR_list = [0, 10, 20, 30]
target_num_list = [1, 2, 3]
results = []

for snr in SNR_list:
    for num_targets in target_num_list:
        # 1. 生成数据
        X_test, y_test, degs_test = generate_dataset(snr=snr, num_targets=num_targets, ...)

        # 2. 深度学习模型评估
        y_pred_dl = train_dl(X_test, y_test, ...)  # 或直接加载已训练模型
        haus_dl = hausdorff_distance(y_test, y_pred_dl)
        succ_dl = recovery_success_rate(y_test, y_pred_dl)

        # 3. 传统算法评估
        y_pred_fft = fft_doa(X_test, ...)
        y_pred_music = music_doa(X_test, ...)
        y_pred_sbl = sbl_doa(X_test, ...)

        haus_fft = hausdorff_distance(y_test, y_pred_fft)
        haus_music = hausdorff_distance(y_test, y_pred_music)
        haus_sbl = hausdorff_distance(y_test, y_pred_sbl)

        # 4. 结果记录
        results.append([snr, num_targets, np.mean(haus_dl), np.mean(haus_fft), np.mean(haus_music), np.mean(haus_sbl)])

# 保存为csv
import pandas as pd
df = pd.DataFrame(results, columns=['SNR', 'NumTargets', 'DL', 'FFT', 'MUSIC', 'SBL'])
df.to_csv('fig/experiment_results.csv', index=False)