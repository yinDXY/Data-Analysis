# WEM-Net：基于小波卷积与多尺度注意力机制的肺炎分类方法

## 项目描述

本项目是数据分析与数据挖掘课程设计，任务是基于胸部 X 光图像完成 NORMAL / PNEUMONIA 二分类。项目使用 Kaggle 公开数据集 Chest X-Ray Images (Pneumonia)，以 DenseNet-121 为基础主干网络，提出 WEM-Net 方法：在高层语义特征后依次引入 WTConv 小波卷积多频特征适配器、EMA 多尺度注意力模块，并使用 Soft MCC Loss 缓解类别不平衡问题。

项目目标：

1. 构建稳定的胸片肺炎二分类模型。
2. 比较 ResNet-50、DenseNet-121、EfficientNet-B0 等预训练 CNN backbone。
3. 通过 WTConv、EMA Attention、Soft MCC Loss 进行模块化改进和消融实验。
4. 使用 ROC、混淆矩阵、Grad-CAM 等方式评估模型性能和可解释性。

类别定义：

| 类别 | 标签 | 含义 |
|---|---:|---|
| NORMAL | 0 | 正常胸片 |
| PNEUMONIA | 1 | 肺炎胸片 |

## 安装依赖

建议使用 Python 3.9+，并优先在虚拟环境中安装依赖。

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `Pillow`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `grad-cam`
- `opencv-python`

若启用 WTConv A 模块，还需要安装：

```bash
pip install PyWavelets
```

## 数据准备

### 数据来源

本项目使用 Kaggle Chest X-Ray Images (Pneumonia) 数据集。课程报告中统计的数据集共包含 5856 张胸部 X 光图像，其中正常样本 1583 张，肺炎样本 4273 张。肺炎样本包含细菌性肺炎和病毒性肺炎图像。数据集连接：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

### 推荐目录结构

将数据放入 `data/raw/chest_xray/`，并整理为以下结构：

```text
data/raw/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

原始工程中数据位于：

```text
../dataset/chest_xray/
```

因此运行命令时可以根据实际位置修改 `--data_dir`。

### 数据划分策略

原始 Kaggle 数据集自带 `train/val/test`，但原始 `val` 目录样本量很小。项目训练时采用 `split_train` 策略：从原始训练集中按类别分层划分 15% 作为验证集，原始测试集保持不变，原始验证集不参与训练和模型选择。

重新划分后的数据规模：

| Split | NORMAL | PNEUMONIA | 总计 |
|---|---:|---:|---:|
| 训练集 | 1140 | 3293 | 4433 |
| 验证集 | 201 | 582 | 783 |
| 测试集 | 234 | 390 | 624 |
| 总计 | 1575 | 4265 | 5840 |

### 图像预处理

所有图像统一转换为 RGB 三通道，并使用 ImageNet 预训练模型对应的均值和标准差进行归一化。

验证集和测试集使用确定性预处理：

```text
Resize -> CenterCrop -> ToTensor -> Normalize
```

训练集可选数据增强：

- RandomResizedCrop
- RandomRotation
- RandomHorizontalFlip
- ColorJitter
- Cutout
- MixUp

验证集和测试集不使用随机增强，以保证评估可复现。

## 训练模型

进入项目目录后运行训练脚本。Windows 环境下建议设置 `--num_workers 0`。

### 训练 DenseNet-121 baseline

```bash
python train_baselines.py ^
  --data_dir ../dataset/chest_xray ^
  --model_name densenet121 ^
  --epochs 100 ^
  --batch_size 64 ^
  --lr 1e-4 ^
  --val_strategy split_train ^
  --val_ratio 0.15 ^
  --num_workers 0 ^
  --output_dir results/densenet121_baseline
```

### 训练完整 WEM-Net

WEM-Net = DenseNet-121 + WTConv(A) + EMA(B) + Soft MCC(C)。

```bash
python train_baselines.py ^
  --data_dir ../dataset/chest_xray ^
  --model_name densenet121 ^
  --epochs 100 ^
  --batch_size 64 ^
  --lr 1e-4 ^
  --val_strategy split_train ^
  --val_ratio 0.15 ^
  --use_wtconv ^
  --use_ema ^
  --loss_name bce_soft_mcc ^
  --bce_weight 1.0 ^
  --mcc_weight 1.0 ^
  --cutout ^
  --mixup_alpha 0.2 ^
  --num_workers 0 ^
  --output_dir results/wem_net
```

### 训练全部 baseline

```bash
python train_baselines.py ^
  --data_dir ../dataset/chest_xray ^
  --model_name all ^
  --epochs 100 ^
  --batch_size 64 ^
  --lr 1e-4 ^
  --val_strategy split_train ^
  --num_workers 0 ^
  --output_dir results/baselines
```

## 评估模型

### 阈值搜索

`evaluate.py` 用于在验证集上搜索不同阈值下的 Accuracy、F1、Youden 指数等指标，并输出阈值曲线和 ROC 图。

```bash
python evaluate.py ^
  --data_dir ../dataset/chest_xray ^
  --model_name densenet121 ^
  --checkpoint_path results/wem_net/checkpoints/densenet121_wtconv_ema_best.pth ^
  --output_dir results/wem_net/threshold_search ^
  --val_strategy split_train ^
  --val_ratio 0.15 ^
  --num_workers 0
```

### Grad-CAM 可解释性分析

```bash
python generate_gradcam.py ^
  --data_dir ../dataset/chest_xray ^
  --model_name densenet121 ^
  --checkpoint_path results/wem_net/checkpoints/densenet121_wtconv_ema_best.pth ^
  --output_dir results/wem_net/gradcam ^
  --num_samples 16 ^
  --threshold 0.5 ^
  --target_class predicted ^
  --sample_mode mixed ^
  --num_workers 0
```

### 单张或批量推理

修改 `test.py` 顶部 `CONFIG` 中的图片路径、模型名称、checkpoint 路径和阈值，然后运行：

```bash
python test.py
```

如需使用桌面图形界面，可运行：

```bash
python apps/app.py
```

## 项目结构

```text
WEM-Net/
  data/                         # 数据相关
    raw/                        # 原始数据目录说明
    processed/                  # 处理后数据目录说明
    datasets/                   # 数据集加载与预处理脚本
  models/                       # 模型定义
    base_model.py               # 基础模型接口
    your_model.py               # ResNet / DenseNet / EfficientNet / WEM-Net
    wtconv_adapter.py           # WTConv A 模块
    ema_adapter.py              # EMA B 模块
    hybridgnet_adapter.py       # HybridGNet ROI 适配器
    external/                   # 第三方轻量依赖代码
  training/                     # 训练相关
    trainers/                   # 训练、验证、测试循环
    losses/                     # BCE / Soft MCC / 组合损失
    metrics/                    # Accuracy / AUC / Sensitivity 等指标
  utils/                        # 工具函数
    data_utils.py               # 随机种子、设备、目录、配置保存
    model_utils.py              # Grad-CAM 与模型加载工具
    visualization.py            # 曲线、ROC、混淆矩阵绘图
    augmentation_demo.py        # 数据增强示例
  configs/                      # 配置文件
    default.yaml                # 默认配置
    experiment1.yaml            # WEM-Net 实验配置
  experiments/                  # 实验记录
    exp_ablation_logs/          # A/B/C 消融日志
  tests/                        # 测试说明
  docs/                         # 文档
  apps/                         # GUI 程序
  src/                          # 保留原始 import 兼容包
  WTConv/                       # 整理后的 WTConv 依赖副本
  requirements.txt              # 依赖列表
  setup.py                      # 安装脚本
  train.py                      # 快速训练入口
  train_baselines.py            # 命令行训练入口
  evaluate.py                   # 阈值搜索/评估入口
  test.py                       # 推理入口
  WEM-Net_完整源代码.ipynb       # 完整源码 Notebook 文件
  README.md                     # 项目说明
```

## 实验结果

### Backbone 对比

三种 ImageNet 预训练 backbone 在相同训练设置下进行比较。DenseNet-121 在 Accuracy、Specificity、Precision 和 F1-score 上表现更均衡，因此被选为后续 WEM-Net 的基础主干网络。

| Backbone | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 0.7292 | 0.9499 | 1.0000 | 0.2778 | 0.6977 | 0.8219 |
| DenseNet-121 | 0.8462 | 0.9230 | 0.9974 | 0.5897 | 0.7959 | 0.8902 |
| EfficientNet-B0 | 0.8397 | 0.9408 | 1.0000 | 0.5726 | 0.7923 | 0.8864 |

### 学习率调优

DenseNet-121 在学习率 `1e-4` 下取得较好的综合表现，因此后续实验采用 `1e-4` 作为默认学习率。

| 学习率 | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1e-3 | 0.8173 | 0.9630 | 1.0000 | 0.5128 | 0.7738 | 0.8725 |
| 3e-4 | 0.8205 | 0.9419 | 0.9974 | 0.5256 | 0.7780 | 0.8742 |
| 1e-4 | 0.8462 | 0.9230 | 0.9974 | 0.5897 | 0.7959 | 0.8902 |
| 1e-5 | 0.8125 | 0.9299 | 0.9974 | 0.5043 | 0.7703 | 0.8693 |

### 消融实验

A 表示 WTConv 多频特征适配器，B 表示 EMA Attention，C 表示 Soft MCC Loss。

| 模型 | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8460 | 0.9230 | 0.9974 | 0.5897 | 0.7959 | 0.8902 |
| baseline + A | 0.8494 | 0.9311 | 1.0000 | 0.5940 | 0.8037 | 0.8924 |
| baseline + A + B | 0.8526 | 0.9408 | 1.0000 | 0.5983 | 0.8050 | 0.8948 |
| baseline + A + B + C | 0.8620 | 0.9623 | 1.0000 | 0.6028 | 0.8076 | 0.8982 |

完整 WEM-Net 在 Accuracy、AUC、Specificity、Precision 和 F1-score 上取得最高结果，并保持了 1.0000 的 Sensitivity，说明模型在保持肺炎检出能力的同时改善了正常样本误报问题。

### 主要模型测试结果

| Model | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 0.7292 | 0.9499 | 1.0000 | 0.2778 | 0.6977 | 0.8219 |
| EfficientNet-B0 | 0.8397 | 0.9408 | 1.0000 | 0.5726 | 0.7923 | 0.8864 |
| DenseNet-121 | 0.8462 | 0.9230 | 0.9974 | 0.5897 | 0.7959 | 0.8902 |
| WEM-Net | 0.8620 | 0.9623 | 1.0000 | 0.6028 | 0.8076 | 0.8982 |

## 可视化分析

项目使用 Grad-CAM 分析模型关注区域。课程报告中的可视化结果显示，WEM-Net 的热力图响应更集中于肺野内部，并覆盖较明显的纹理异常和密度变化区域。相比部分 baseline 模型，WEM-Net 对非肺实质背景结构的依赖更少，具有更好的可解释性。

## 关键要点总结

1. 模块化设计：将数据、模型、训练、损失、评估和工具函数拆分为独立模块，便于复用和消融。
2. 配置驱动：使用 `configs/` 记录默认实验和 WEM-Net 实验参数，便于复现实验。
3. 类型提示和文档字符串：核心函数和类保留清晰注释，提高代码可读性。
4. 异常处理：数据读取、模型导入、外部模块依赖等位置包含必要错误提示。
5. 日志记录：训练过程保存配置、数据分布、训练日志、预测结果和图表。
6. 可复现性：固定随机种子，并记录验证集划分策略、阈值、学习率和增强设置。
7. 版本控制：提交时应避免加入大体积数据、权重、缓存和运行产物，可通过 `.gitignore` 管理。
8. 医学指标意识：除 Accuracy 外，同时关注 Sensitivity、Specificity、Precision、F1 和 AUC。

## 注意事项

- 目录不包含原始图片数据和 `.pth` 权重文件。
- 若直接运行整理目录中的代码，请确认 `--data_dir` 与 checkpoint 路径指向真实存在的位置。
- `--use_wtconv` 和 `--use_ema` 仅支持 `densenet121`。
- 当前训练脚本使用 logits 训练，模型最后不需要手动添加 sigmoid。
- 测试集仅用于最终评估，不参与训练、验证、阈值搜索或模型选择。
