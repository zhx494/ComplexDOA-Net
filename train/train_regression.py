import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.doa_regression import build_regression_model
from models.losses import mse_hausdorff_loss

def train_regression_model(data_dir='./data', epochs=100, batch_size=32):
    """
    训练回归模型
    """
    # 加载数据
    X_train = np.load(os.path.join(data_dir, 'train/Xcov.npy'))
    y_train = np.load(os.path.join(data_dir, 'train/label.npy'))
    X_val = np.load(os.path.join(data_dir, 'val/Xcov.npy'))
    y_val = np.load(os.path.join(data_dir, 'val/label.npy'))
    
    # 构建模型
    model = build_regression_model(input_shape=X_train.shape[1:], output_dim=y_train.shape[1])
    model.compile(optimizer='adam', loss=mse_hausdorff_loss)
    
    # 设置回调
    os.makedirs('checkpoint', exist_ok=True)
    checkpoint = ModelCheckpoint('checkpoint/regression_best.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping])
    
    return model

if __name__ == "__main__":
    model = train_regression_model(data_dir='./data', epochs=100, batch_size=32)
    print("回归模型训练完成！")