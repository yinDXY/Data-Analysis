"""
plots.py — 可视化工具

所有函数只负责绘图与保存，不包含训练或推理逻辑。
使用纯 matplotlib，不依赖 seaborn。
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 非交互后端，适合服务器/脚本环境
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ─────────────────────────────────────────────
# 1. 训练曲线
# ─────────────────────────────────────────────

def plot_training_curves(
    log_df: pd.DataFrame,
    save_path: str,
    model_name: str = "",
) -> None:
    """绘制并保存训练曲线（Loss / AUC / Accuracy）。

    Args:
        log_df     : 训练日志 DataFrame，必须含列：
                     epoch / train_loss / val_loss / val_auc / val_accuracy。
                     若含 test_auc / test_accuracy 列，则同时绘制测试曲线。
        save_path  : 图片保存路径（含文件名）。
        model_name : 显示在标题中的模型名称。
    """
    required = {"epoch", "train_loss", "val_loss", "val_auc", "val_accuracy"}
    missing = required - set(log_df.columns)
    if missing:
        raise ValueError(f"log_df 缺少列: {missing}")

    has_test = "test_auc" in log_df.columns and "test_accuracy" in log_df.columns
    epochs = log_df["epoch"].values
    title_prefix = f"{model_name} - " if model_name else ""

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax1, ax2, ax3 = axes

    # ── Loss 曲线 ──────────────────────────────
    ax1.plot(epochs, log_df["train_loss"].values, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, log_df["val_loss"].values,   "r-o", markersize=3, label="Val Loss")
    ax1.set_title(f"{title_prefix}Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── AUC 曲线 ───────────────────────────────
    ax2.plot(epochs, log_df["val_auc"].values, "g-o", markersize=3, label="Val AUC")
    if has_test:
        ax2.plot(epochs, log_df["test_auc"].values, "m-o", markersize=3, label="Test AUC")
    ax2.set_title(f"{title_prefix}AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ── Accuracy 曲线 ──────────────────────────
    if "train_accuracy" in log_df.columns:
        ax3.plot(epochs, log_df["train_accuracy"].values, "b-o", markersize=3, label="Train Acc")
    ax3.plot(epochs, log_df["val_accuracy"].values,
             color="orange", marker="o", markersize=3, linestyle="-", label="Val Acc")
    if has_test:
        ax3.plot(epochs, log_df["test_accuracy"].values, "m-o", markersize=3, label="Test Acc")
    ax3.set_title(f"{title_prefix}Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0.0, 1.05)
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 训练曲线已保存: {save_path}")


# ─────────────────────────────────────────────
# 2. 独立精度曲线
# ─────────────────────────────────────────────

def plot_accuracy_curve(
    log_df: pd.DataFrame,
    save_path: str,
    model_name: str = "",
) -> None:
    """绘制并保存独立的精度曲线图（Train / Val / Test Accuracy）。

    Args:
        log_df     : 训练日志 DataFrame，需含 epoch / val_accuracy 列。
                     若含 train_accuracy / test_accuracy 列则一并绘制。
        save_path  : 图片保存路径（含文件名）。
        model_name : 显示在标题中的模型名称。
    """
    epochs = log_df["epoch"].values
    title  = (
        f"{model_name} - Training and Validation Accuracy"
        if model_name else "Training and Validation Accuracy"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if "train_accuracy" in log_df.columns:
        ax.plot(epochs, log_df["train_accuracy"].values,
                "b-o", markersize=4, label="Training Accuracy")
    ax.plot(epochs, log_df["val_accuracy"].values,
            color="orange", marker="o", markersize=4, linestyle="-", label="Validation Accuracy")

    # 自动 y 轴下限（留 0.05 余量）
    all_acc = [log_df["val_accuracy"].values]
    if "train_accuracy" in log_df.columns:
        all_acc.append(log_df["train_accuracy"].values)
    y_min = max(0.0, float(np.concatenate(all_acc).min()) - 0.05)
    ax.set_ylim(y_min, 1.02)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 独立精度曲线已保存: {save_path}")


# ─────────────────────────────────────────────
# 3. 单模型 ROC 曲线
# ─────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    model_name: str = "",
) -> None:
    """绘制并保存单个模型的 ROC 曲线。

    Args:
        y_true    : 真实标签（0/1）。
        y_prob    : 预测为 PNEUMONIA 的概率。
        save_path : 图片保存路径（含文件名）。
        model_name: 显示在标题中的模型名称。
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    title = f"{model_name} - ROC Curve" if model_name else "ROC Curve"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] ROC 曲线已保存: {save_path}")


# ─────────────────────────────────────────────
# 3. 混淆矩阵
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    save_path: str,
    model_name: str = "",
) -> None:
    """绘制并保存 2×2 混淆矩阵热力图。

    矩阵布局（labels=[0, 1]，PNEUMONIA 为阳性类）：
        [[TN, FP],
         [FN, TP]]

    Args:
        tn, fp, fn, tp: 混淆矩阵四项。
        save_path     : 图片保存路径（含文件名）。
        model_name    : 显示在标题中的模型名称。
    """
    cm = np.array([[tn, fp], [fn, tp]])
    class_names = ["NORMAL (0)", "PNEUMONIA (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    title = f"{model_name} - Confusion Matrix" if model_name else "Confusion Matrix"
    ax.set_title(title)

    # 在格子内显示数值，颜色随背景自动对比
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 混淆矩阵已保存: {save_path}")


# ─────────────────────────────────────────────
# 4. 多模型 ROC 对比
# ─────────────────────────────────────────────

def plot_all_models_roc(
    roc_data: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
) -> None:
    """绘制多个模型的 ROC 曲线对比图。

    Args:
        roc_data : 字典，格式：
                   {model_name: {"y_true": ndarray, "y_prob": ndarray}, ...}
        save_path: 图片保存路径（含文件名）。
    """
    if not roc_data:
        raise ValueError("roc_data 为空，无法绘图")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

    colors = plt.cm.tab10.colors
    for idx, (model_name, data) in enumerate(roc_data.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"], pos_label=1)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model_name} (AUC = {roc_auc:.4f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 多模型 ROC 对比图已保存: {save_path}")


# ─────────────────────────────────────────────
# 5. 模型指标对比柱状图
# ─────────────────────────────────────────────

def plot_metrics_comparison(
    summary_df: pd.DataFrame,
    save_path: str,
) -> None:
    """绘制多模型指标对比柱状图。

    Args:
        summary_df: DataFrame，必须含列 "model" 以及至少：
                    accuracy / sensitivity / specificity / auc。
        save_path : 图片保存路径（含文件名）。
    """
    if "model" not in summary_df.columns:
        raise ValueError("summary_df 必须包含 'model' 列")

    metric_cols = [c for c in ["accuracy", "sensitivity", "specificity", "auc"]
                   if c in summary_df.columns]
    if not metric_cols:
        raise ValueError("summary_df 中未找到任何可绘制的指标列")

    models = summary_df["model"].tolist()
    n_models = len(models)
    n_metrics = len(metric_cols)

    x = np.arange(n_models)
    bar_width = 0.7 / n_metrics
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(max(6, n_models * 2), 5))

    for i, metric in enumerate(metric_cols):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            summary_df[metric].values,
            width=bar_width,
            color=colors[i % len(colors)],
            label=metric.capitalize(),
            edgecolor="white",
            linewidth=0.5,
        )
        # 在柱顶标注数值
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.005,
                    f"{height:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.10)
    ax.set_title("Model Metrics Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 指标对比图已保存: {save_path}")
