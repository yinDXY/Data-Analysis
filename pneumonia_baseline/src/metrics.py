"""
metrics.py — 二分类评估指标

约定：
  - 阳性类 (Positive) = PNEUMONIA = 1
  - 阴性类 (Negative) = NORMAL    = 0
  - 模型输出单个 logit，先 sigmoid 转概率，再按 threshold 二值化
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)


# ─────────────────────────────────────────────
# 1. Sigmoid（numpy）
# ─────────────────────────────────────────────

def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """将 logit 数组转换为概率（numerically stable）。

    Args:
        logits: 任意 shape 的 numpy 数组。

    Returns:
        与 logits 同 shape 的概率数组，值域 (0, 1)。
    """
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


# ─────────────────────────────────────────────
# 2. 二分类指标计算
# ─────────────────────────────────────────────

def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """计算二分类全套指标（阳性类 = PNEUMONIA = 1）。

    Args:
        y_true    : 真实标签数组，值为 0 或 1。
        y_prob    : 预测为 PNEUMONIA 的概率数组，值域 [0, 1]。
        threshold : 二值化阈值，默认 0.5。

    Returns:
        包含以下键的 dict：
            accuracy, sensitivity, specificity, precision,
            f1, auc, tn, fp, fn, tp
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # 混淆矩阵，labels=[0,1] 保证顺序：
    #   [[TN, FP],
    #    [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ── 各项指标（分母为 0 时返回 0.0）──────────
    accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall for PNEUMONIA
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Recall for NORMAL
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    # ── AUC-ROC（使用概率，而非预测标签）───────
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # 只有一个类别时 roc_auc_score 会抛 ValueError
        auc = float("nan")

    return {
        "accuracy":    round(float(accuracy),    4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision":   round(float(precision),   4),
        "f1":          round(float(f1),          4),
        "auc":         round(auc, 4) if not np.isnan(auc) else float("nan"),
        "tn":          int(tn),
        "fp":          int(fp),
        "fn":          int(fn),
        "tp":          int(tp),
    }


# ─────────────────────────────────────────────
# 3. 多模型指标汇总
# ─────────────────────────────────────────────

def metrics_to_dataframe(metrics_list: List[Dict]) -> pd.DataFrame:
    """将多个模型的指标 dict 列表转为 DataFrame。

    每个 dict 应包含 compute_binary_metrics 的返回键，
    可额外包含 "model" 字段作为行标识。

    Args:
        metrics_list: 指标 dict 列表，每个 dict 对应一个模型或一个 epoch。

    Returns:
        pandas DataFrame，每行一个模型 / epoch 的指标。

    Example:
        >>> metrics_list = [
        ...     {"model": "resnet50",      **compute_binary_metrics(y_true, prob_r)},
        ...     {"model": "densenet121",   **compute_binary_metrics(y_true, prob_d)},
        ... ]
        >>> df = metrics_to_dataframe(metrics_list)
    """
    if not metrics_list:
        raise ValueError("metrics_list 为空，无法构建 DataFrame")

    df = pd.DataFrame(metrics_list)

    # 若存在 "model" 列，将其置为第一列
    if "model" in df.columns:
        cols = ["model"] + [c for c in df.columns if c != "model"]
        df = df[cols]

    return df
