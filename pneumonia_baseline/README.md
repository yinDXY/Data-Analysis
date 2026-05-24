# Pneumonia Baseline

基于 PyTorch 的胸部 X 光肺炎二分类 Baseline 项目。  
使用 Kaggle **Chest X-Ray Images (Pneumonia)** 数据集，训练三种经典 CNN backbone。

---

## 类别定义

| 类别      | 标签 | 含义  |
|-----------|------|-------|
| NORMAL    | 0    | 正常  |
| PNEUMONIA | 1    | 肺炎（阳性类） |

---

## 环境安装

```bash
pip install -r requirements.txt
```

依赖：`torch>=2.0` · `torchvision>=0.15` · `scikit-learn` · `matplotlib` · `pandas` · `numpy` · `tqdm` · `Pillow`

---

## 数据集目录结构

将数据集放置为以下结构（`--data_dir` 指向 `chest_xray/`）：

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 训练命令

### 训练单个模型

```bash
python train_baselines.py \
  --data_dir ./dataset/chest_xray \
  --model_name resnet50 \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_strategy split_train
```

`model_name` 可选：`resnet50` / `densenet121` / `efficientnet_b0`

### 训练全部模型（依次运行三个 backbone）

```bash
python train_baselines.py \
  --data_dir ./dataset/chest_xray \
  --model_name all \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_strategy split_train
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | *(必填)* | chest_xray 根目录 |
| `--model_name` | `resnet50` | 模型名称，支持 `all` |
| `--epochs` | `10` | 训练轮数 |
| `--batch_size` | `32` | 批大小 |
| `--lr` | `1e-4` | 学习率（AdamW） |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--num_workers` | `4` | DataLoader 进程数 |
| `--val_strategy` | `split_train` | 验证集策略（见下） |
| `--val_ratio` | `0.15` | split_train 模式验证集比例 |
| `--threshold` | `0.5` | sigmoid 输出二值化阈值 |
| `--seed` | `42` | 随机种子 |
| `--output_dir` | `results` | 输出根目录 |

### 验证集策略

| `--val_strategy` | 验证集来源 | 说明 |
|---|---|---|
| `split_train` | 从 train 中分层划分 | **推荐**，原始 val 仅 16 张 |
| `original` | 使用原始 val 目录 | 适合对比原始论文结果 |

---

## 输出文件说明

```
results/
├── checkpoints/
│   └── {model}_best.pth              # Best checkpoint（按 val AUC 选择）
├── logs/
│   ├── config.json                   # 本次实验配置
│   ├── dataset_distribution.csv      # 数据集类别数量统计
│   ├── {model}_training_log.csv      # 每轮训练指标
│   └── baseline_summary.csv          # 所有模型测试集指标汇总
├── figures/
│   ├── {model}_training_curves.png   # Loss + AUC 训练曲线
│   ├── {model}_roc_curve.png         # 单模型 ROC 曲线
│   ├── all_models_roc_comparison.png # 多模型 ROC 对比（all 模式）
│   └── baseline_metrics_comparison.png # 多模型指标柱状图（all 模式）
├── confusion_matrices/
│   └── {model}_confusion_matrix.png  # 混淆矩阵
└── predictions/
    └── {model}_test_predictions.csv  # 逐样本预测结果
```

### `baseline_summary.csv` 列说明

| 列 | 说明 |
|---|---|
| model | 模型名称 |
| accuracy | 准确率 |
| sensitivity | **灵敏度 = Pneumonia Recall** = TP/(TP+FN) |
| specificity | **特异度 = Normal Recall** = TN/(TN+FP) |
| precision | 精确率 = TP/(TP+FP) |
| f1 | F1-score |
| auc | AUC-ROC（基于概率计算） |
| tn / fp / fn / tp | 混淆矩阵四项 |

### `{model}_test_predictions.csv` 列说明

| 列 | 说明 |
|---|---|
| image_path | 图片绝对路径 |
| true_label | 真实标签（0/1） |
| true_class | 真实类名（NORMAL/PNEUMONIA） |
| pred_prob | 预测为 PNEUMONIA 的概率 |
| pred_label | 预测标签（0/1） |
| pred_class | 预测类名 |
| correct | 预测是否正确（1/0） |

---

## 指标含义

| 指标 | 公式 | 临床意义 |
|---|---|---|
| **Accuracy** | $(TP+TN)/(TP+TN+FP+FN)$ | 整体正确率 |
| **Sensitivity** | $TP/(TP+FN)$ | Pneumonia Recall，漏诊率越低越好 |
| **Specificity** | $TN/(TN+FP)$ | Normal Recall，误诊率越低越好 |
| **Precision** | $TP/(TP+FP)$ | 预测为肺炎中真实肺炎的比例 |
| **F1** | $2 \cdot P \cdot R / (P+R)$ | Precision 与 Sensitivity 的调和平均 |
| **AUC-ROC** | sklearn.metrics.roc_auc_score | 综合排序能力，不依赖阈值 |

> **重要说明**  
> - **Sensitivity（灵敏度）= Pneumonia 的 Recall**，即模型找出肺炎患者的能力。  
> - **Specificity（特异度）= Normal 的 Recall**，即模型识别正常胸片的能力。  
> - **Best checkpoint** 以 **validation AUC** 为标准选择，与 test set 完全无关。  
> - **Test set 仅在训练完成后评估一次**，不参与任何模型选择或超参数调整。

---

## 项目结构

```
pneumonia_baseline/
├── train_baselines.py      # 主训练脚本
├── requirements.txt
├── README.md
├── src/
│   ├── dataset.py          # 数据加载、DataLoader 构建
│   ├── models.py           # CNN backbone（ResNet / DenseNet / EfficientNet）
│   ├── engine.py           # 训练 / 验证 / 测试循环
│   ├── metrics.py          # 评估指标计算
│   ├── utils.py            # 通用工具
│   └── plots.py            # 可视化
└── results/                # 自动生成
    ├── checkpoints/
    ├── logs/
    ├── figures/
    ├── confusion_matrices/
    └── predictions/
```


---

## Grad-CAM 可解释性分析

Grad-CAM 用于观察模型预测时关注的图像区域，辅助判断模型是否主要关注肺部区域，而非边框、文字或背景噪声。

### 额外依赖安装

```bash
pip install grad-cam opencv-python
```

### 生成 Grad-CAM 示例

```bash
python generate_gradcam.py \
  --data_dir ./dataset/chest_xray \
  --model_name resnet50 \
  --checkpoint_path results/checkpoints/resnet50_best.pth \
  --output_dir results/gradcam/resnet50 \
  --num_samples 16 \
  --threshold 0.5 \
  --target_class predicted \
  --sample_mode mixed
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | *(必填)* | chest_xray 根目录 |
| `--model_name` | *(必填)* | `resnet50` / `densenet121` / `efficientnet_b0` |
| `--checkpoint_path` | *(必填)* | best checkpoint .pth 文件路径 |
| `--output_dir` | `results/gradcam` | 热力图输出根目录 |
| `--num_samples` | `16` | 生成热力图的样本总数 |
| `--threshold` | `0.5` | 二值化阈值，**必须与训练评估时一致** |
| `--image_size` | `224` | 输入图像尺寸 |
| `--target_class` | `predicted` | Grad-CAM 目标类别（见下） |
| `--sample_mode` | `mixed` | 样本选择策略（见下） |

#### `--target_class` 说明

| 值 | 含义 |
|---|---|
| `predicted` | 对模型预测的类别生成热力图（推荐） |
| `pneumonia` | 固定对 PNEUMONIA 方向生成热力图 |
| `normal` | 固定对 NORMAL 方向生成热力图 |

#### `--sample_mode` 说明

| 值 | 含义 |
|---|---|
| `mixed` | 从 TP / TN / FP / FN 中各选若干张（置信度优先） |
| `tp` | 仅选真阳性（真实肺炎 & 预测肺炎） |
| `tn` | 仅选真阴性（真实正常 & 预测正常） |
| `fp` | 仅选假阳性（误报肺炎） |
| `fn` | 仅选假阴性（漏诊肺炎） |
| `all` | 测试集前 `num_samples` 张 |

### 输出目录结构

```
results/gradcam/resnet50/
├── gradcam_selected_predictions.csv   # 选中样本的预测信息
├── TP/
│   ├── sample_000_original.png        # 原图
│   ├── sample_000_heatmap.png         # Grad-CAM 热力图
│   ├── sample_000_overlay.png         # 原图 + 热力图叠加
│   └── sample_000_panel.png           # 1×3 对比图（含标题）
├── TN/
├── FP/
└── FN/
```

### 模型目标层

| 模型 | Grad-CAM 目标层 | 说明 |
|---|---|---|
| ResNet-50 | `model.layer4[-1]` | 最后一个 Bottleneck |
| DenseNet-121 | `model.features.denseblock4` | 最后一个 DenseBlock |
| EfficientNet-B0 | `model.features[-1]` | 最后一个 MBConv 组 |

---

## A 模块：HybridGNet 肺部 ROI 预处理

### 概述

A 模块使用 [HybridGNet](https://github.com/ngaggion/HybridGNet) 从胸部 X-ray 中
预测左右肺轮廓 landmarks，生成 binary lung mask，并裁剪出肺部 ROI，
输出与原始数据集目录结构完全一致的新数据集。

**注意**：HybridGNet 仅用于**离线预处理**，不参与 DenseNet 分类训练，不修改模型结构。

### 目录准备

HybridGNet 仓库需与 `pneumonia_baseline` 放在**同一根目录**：

```
project_root/
├── pneumonia_baseline/
└── HybridGNet/
```

### 额外依赖

```bash
pip install torch-geometric scipy opencv-python
```

HybridGNet 内部通过 `torch_geometric` 实现 Chebyshev 图卷积，必须安装。

### 权重文件

HybridGNet 不随仓库分发权重，需自行下载官方预训练权重并放置到
`../HybridGNet/weights/weights.pt`（路径可通过 `--weights_path` 自定义）。

### 小规模测试（20 张）

```bash
python precompute_hybridgnet_roi.py \
  --data_dir ./dataset/chest_xray \
  --hybridgnet_dir ../HybridGNet \
  --weights_path ../HybridGNet/weights/weights.pt \
  --output_dir ./dataset/chest_xray_hybridgnet_roi_test \
  --margin_ratio 0.08 \
  --fallback original \
  --save_masks \
  --save_visualizations \
  --max_samples 20 \
  --overwrite
```

### 全量预处理

```bash
python precompute_hybridgnet_roi.py \
  --data_dir ./dataset/chest_xray \
  --hybridgnet_dir ../HybridGNet \
  --weights_path ../HybridGNet/weights/weights.pt \
  --output_dir ./dataset/chest_xray_hybridgnet_roi \
  --margin_ratio 0.08 \
  --fallback original \
  --save_masks \
  --save_visualizations \
  --overwrite
```

### 使用预处理后的数据训练 DenseNet-121

```bash
python train_baselines.py \
  --data_dir ./dataset/chest_xray_hybridgnet_roi \
  --model_name densenet121
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | *(必填)* | 原始 chest_xray 根目录 |
| `--hybridgnet_dir` | *(必填)* | HybridGNet 仓库路径 |
| `--weights_path` | *(必填)* | HybridGNet 权重 .pt 路径 |
| `--output_dir` | `./dataset/chest_xray_hybridgnet_roi` | 输出数据集目录 |
| `--margin_ratio` | `0.08` | ROI bbox 外扩比例 |
| `--fallback` | `original` | 分割失败时策略：`original`=保留原图，`skip`=跳过 |
| `--device` | 自动检测 | `cuda` 或 `cpu` |
| `--overwrite` | False | 允许覆盖已有输出目录 |
| `--save_masks` | False | 保存 binary lung mask |
| `--save_visualizations` | False | 保存可视化对比图 |
| `--max_samples` | None（全量）| 处理图像数上限（用于测试） |

### 输出结构

```
output_dir/
├── train/NORMAL/           ← ROI 裁剪后的图像（与原始格式相同）
├── train/PNEUMONIA/
├── val/NORMAL/
├── val/PNEUMONIA/
├── test/NORMAL/
├── test/PNEUMONIA/
├── masks/                  ← 仅 --save_masks 时生成
│   ├── train/NORMAL/*.png
│   └── ...
├── visualizations/         ← 仅 --save_visualizations 时生成
│   ├── train/NORMAL/*_vis.jpg  （原图 | mask | ROI 并排）
│   └── ...
└── preprocessing_metadata.csv
```

`preprocessing_metadata.csv` 字段：

| 字段 | 说明 |
|---|---|
| split | train / val / test |
| class_name | NORMAL / PNEUMONIA |
| label | 0 / 1 |
| original_path | 原始图像路径 |
| output_path | 输出图像路径 |
| mask_path | mask 路径（不保存时为空） |
| status | success / fallback / skipped / failed |
| bbox_x1/y1/x2/y2 | 裁剪框坐标（失败时为 -1） |
| mask_area_ratio | 肺部 mask 占图像面积比（失败时为 -1.0） |
| error_message | 错误描述（成功时为空） |

---

## C 模块：类别不平衡损失（Soft MCC Loss）

### 目录准备

`address-class-imbalance` 仓库（来自 <https://github.com/daniel-scholz/address-class-imbalance>）
需与 `pneumonia_baseline` 放在**同一根目录**下：

```
project_root/
├── pneumonia_baseline/
└── address-class-imbalance/
```

> 本项目在 `src/losses.py` 中内联了 Soft MCC 的核心算法（与第三方仓库逻辑完全一致），
> **无需修改**第三方仓库源码，也无需将其加入 Python 路径。

### 可选损失函数

| `--loss_name` | 说明 |
|---|---|
| `bce` | BCEWithLogitsLoss（默认 baseline） |
| `soft_mcc` | Soft Matthews Correlation Coefficient Loss |
| `bce_soft_mcc` | `bce_weight × BCE + mcc_weight × SoftMCC` |

### 训练 BCE Baseline

```bash
python train_baselines.py \
  --data_dir ./dataset/chest_xray \
  --model_name densenet121 \
  --loss_name bce
```

### 训练 C 模块（Soft MCC 组合损失）

```bash
python train_baselines.py \
  --data_dir ./dataset/chest_xray \
  --model_name densenet121 \
  --loss_name bce_soft_mcc \
  --bce_weight 1.0 \
  --mcc_weight 1.0
```

### 相关参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--loss_name` | `bce` | 损失函数类型 |
| `--bce_weight` | `1.0` | `bce_soft_mcc` 时 BCE 项权重 |
| `--mcc_weight` | `1.0` | `soft_mcc` / `bce_soft_mcc` 时 Soft MCC 项权重 |

### 快速启动（train.py）

编辑 `train.py` 中的 `CONFIG`：

```python
loss_name  = "bce_soft_mcc"
bce_weight = 1.0
mcc_weight = 1.0
```

然后直接运行：

```bash
python train.py
```
