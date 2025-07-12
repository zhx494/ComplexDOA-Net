import numpy as np
from keras.models import load_model
from evaluate.metrics import hausdorff_distance, recovery_success_rate, mean_squared_error

def evaluate_on_test(model_path, X_test, y_test, task='regression'):
    model = load_model(model_path, compile=False)
    y_pred = model.predict(X_test)
    hausdorff = hausdorff_distance(y_test, y_pred)
    success = recovery_success_rate(y_test, y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(f"Hausdorff均值: {np.mean(hausdorff):.4f}")
    print(f"恢复成功率: {success:.4f}")
    print(f"MSE: {mse:.4f}")
    # 可保存为txt/csv
    return hausdorff, success, mse

if __name__ == '__main__':
    X_test = np.load('data/test/Xcov.npy')
    y_test = np.load('data/test/label.npy')
    evaluate_on_test('checkpoint/classification_best.h5', X_test, y_test, task='classification')
    evaluate_on_test('checkpoint/regression_best.h5', X_test, y_test, task='regression')