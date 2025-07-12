import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.doa_classification import build_classification_model
from models.losses import multilabel_categorical_crossentropy

def train_classification_model(data_dir='./data', epochs=100, batch_size=32):
    """
    训练分类模型
    """
    # 加载数据
    X_train = np.load(os.path.join(data_dir, 'train/Xcov.npy'))
    y_train = np.load(os.path.join(data_dir, 'train/label.npy'))
    X_val = np.load(os.path.join(data_dir, 'val/Xcov.npy'))
    y_val = np.load(os.path.join(data_dir, 'val/label.npy'))
    
    # 构建模型
    model = build_classification_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    model.compile(optimizer='adam', loss=multilabel_categorical_crossentropy, metrics=['accuracy'])
    
    # 设置回调
    os.makedirs('checkpoint', exist_ok=True)
    checkpoint = ModelCheckpoint('checkpoint/classification_best.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping])
    
    return model

if __name__ == "__main__":
    model = train_classification_model(data_dir='./data', epochs=100, batch_size=32)
    print("分类模型训练完成！")