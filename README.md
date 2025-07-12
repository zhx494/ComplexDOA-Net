# 基于复数卷积神经网络的波达方向估计超分辨系统

本项目实现了一个使用复数卷积神经网络进行波达方向(DOA, Direction of Arrival)估计的深度学习系统。系统支持分类和回归两种模式，通过深度学习提高DOA估计的超分辨性能。

## 项目结构

```
├── checkpoint/        # 保存训练好的模型权重
├── complexnn/         # 复数神经网络实现
├── data/              # 数据生成和处理
├── evaluate/          # 模型评估和可视化
├── models/            # 模型定义
├── results/           # 结果和可视化图表
├── train/             # 训练相关代码
├── Deep_RSA_DOA-main/ # 参考项目(仅数据生成部分)
├── main.py            # 主入口
└── requirements.txt   # 项目依赖
```

## 复数神经网络模块说明

### 关于complexnn模块

本项目中的复数神经网络模块(`complexnn/`)参考了deep_complex_networks-master项目的实现。该模块提供了对复数卷积神经网络的全面支持，包括：

1. **复数卷积操作**(`conv.py`)：
   - 复数卷积层的实现
   - 支持多种复数卷积方式
   - 兼容Keras框架的API设计

2. **复数批标准化**(`bn.py`)：
   - 适用于复数数据的批归一化
   - 保持复数特性的同时进行标准化处理

3. **复数激活函数**：
   - 复数ReLU、复数sigmoid等激活函数
   - 保持相位信息的非线性变换

4. **复数初始化方法**(`init.py`)：
   - 专为复数网络设计的权重初始化策略
   - 考虑复数分布特性的初始化器

我们在参考原始项目的基础上，针对波达方向估计任务进行了适应性修改和优化，以提高模型在DOA估计中的性能。

## 数据生成模块说明

### 关于Deep_RSA_DOA-main引用说明

本项目中的数据生成模块(`data/`)参考了Deep_RSA_DOA-main项目的实现。我们基于该项目的数据生成逻辑，进行了以下改进和调整：

1. **数据格式转换**：从PyTorch框架转换为NumPy实现，提高了与TensorFlow/Keras的兼容性
2. **功能扩展**：
   - 添加了回归标签生成模式(`make_reg_label`)，支持高斯形式的连续标签
   - 增加了快照数(snapshots)参数，更符合实际信号处理场景
   - 添加了协方差矩阵计算和保存功能
3. **代码结构优化**：
   - 将数据生成功能模块化，提供更清晰的接口
   - 添加了更详细的函数文档和中文注释

### 主要数据生成函数

- `array_steering_vec`: 生成阵列导向矢量
- `synthesize_complex_signal`: 合成复数信号
- `calc_cov_matrix`: 计算协方差矩阵
- `make_class_label`/`make_reg_label`: 生成分类/回归标签
- `synthesize_sample`: 合成单个样本
- `synthesize_dataset`: 生成整个数据集

### 数据格式

生成的数据包含三个文件：
- `Xcov.npy`: 协方差矩阵数据，形状为(样本数, N, N, 2)，最后一维表示实部和虚部
- `label.npy`: 标签数据，形状为(样本数, 角度网格数)
- `degs.npy`: 真实角度值，用于评估和可视化

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 数据生成：
```bash
python data/data_utils.py --num_samples_train 1000 --num_samples_test 200 --mode classification
```

3. 模型训练和评估（一键完成）（或者直接运行main.py）：   
```bash
python main.py --stage all --mode classification
```

4. 分步执行：
```bash
# 仅生成数据
python main.py --stage gen
# 仅训练模型
python main.py --stage train --mode classification
# 仅评估模型
python main.py --stage eval --mode classification
```

## 引用和致谢

- 复数神经网络模块参考了deep_complex_networks-master项目的实现，感谢原作者提供的复数深度学习框架。项目地址：https://github.com/ChihebTrabelsi/deep_complex_networks/tree/master
- 数据生成部分参考了Deep_RSA_DOA项目的实现，我们对原项目作者表示感谢。项目地址：https://github.com/ruxinzh/Deep_RSA_DOA/tree/main

我们对以上参考代码进行了重构、优化和扩展，以适应波达方向估计超分辨任务的需求。 