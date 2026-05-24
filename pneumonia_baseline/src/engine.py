"""
engine.py — 训练 / 验证 / 测试循环

约定：
  - 模型输出单个 logit，shape [batch_size]
  - Loss：BCEWithLogitsLoss
  - 评估指标来自 src.metrics.compute_binary_metrics
  - Best checkpoint 以 validation AUC 为标准
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_binary_metrics, sigmoid_np


# ─────────────────────────────────────────────
# 1. 单 epoch 训练
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    """在训练集上运行一个 epoch。

    Args:
        model       : 待训练模型。
        dataloader  : 训练集 DataLoader（每批返回 image, label, path）。
        criterion   : 损失函数（BCEWithLogitsLoss）。
        optimizer   : 优化器。
        device      : 运算设备。
        mixup_alpha : MixUp 的 Beta 分布参数，0.0 表示不使用 MixUp。

    Returns:
        (train_loss, train_accuracy) — 平均训练 Loss 与精度。
    """
    model.train()
    running_loss = 0.0
    n_correct    = 0
    n_samples    = 0

    pbar = tqdm(dataloader, desc="  [Train]", leave=False, dynamic_ncols=True)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()

        if mixup_alpha > 0.0:
            from .dataset import mixup_batch
            images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
            logits = model(images).squeeze(1)          # [batch_size]
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            ref_labels = labels_a   # MixUp 时以主标签计算精度
        else:
            logits = model(images).squeeze(1)          # [batch_size]
            loss = criterion(logits, labels)
            ref_labels = labels

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (logits > 0).float()
            n_correct += (preds == ref_labels).sum().item()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / n_samples, n_correct / n_samples


# ─────────────────────────────────────────────
# 2. 评估（验证集 / 测试集）
# ─────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict, np.ndarray, np.ndarray, List[str]]:
    """在验证集或测试集上评估模型。

    Args:
        model      : 待评估模型。
        dataloader : 评估 DataLoader（每批返回 image, label, path）。
        criterion  : 损失函数（BCEWithLogitsLoss）。
        device     : 运算设备。
        threshold  : 二值化阈值，默认 0.5。

    Returns:
        avg_loss   : 平均 Loss。
        metrics    : compute_binary_metrics 返回的指标 dict。
        y_true     : 真实标签数组。
        y_prob     : 预测为 PNEUMONIA 的概率数组。
        image_paths: 对应图片路径列表。
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths:  List[str]        = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  [Eval] ", leave=False, dynamic_ncols=True)
        for images, labels, paths in pbar:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True).float()

            logits = model(images).squeeze(1)       # [batch_size]
            loss = criterion(logits, labels_dev)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

    avg_loss = running_loss / n_samples
    y_logits = np.concatenate(all_logits, axis=0)
    y_true   = np.concatenate(all_labels, axis=0).astype(int)
    y_prob   = sigmoid_np(y_logits)                 # 概率，用于 AUC

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    return avg_loss, metrics, y_true, y_prob, all_paths


# ─────────────────────────────────────────────
# 3. 完整训练流程
# ─────────────────────────────────────────────

def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    device: torch.device,
    epochs: int,
    model_name: str,
    checkpoint_path: str,
    log_path: Optional[str] = None,
    threshold: float = 0.5,
    mixup_alpha: float = 0.0,
    test_loader: Optional[DataLoader] = None,
) -> pd.DataFrame:
    """端到端训练循环，保存 best checkpoint 与训练日志。

    Best checkpoint 标准：validation AUC（nan 时跳过更新）。

    Args:
        model           : 待训练模型。
        train_loader    : 训练集 DataLoader。
        val_loader      : 验证集 DataLoader。
        criterion       : 损失函数。
        optimizer       : 优化器。
        scheduler       : 学习率调度器（可为 None）。
        device          : 运算设备。
        epochs          : 训练轮数。
        model_name      : 模型名称，用于日志显示与 checkpoint 标记。
        checkpoint_path : best model 权重保存路径（.pth）。
        log_path        : 训练日志 CSV 保存路径。
        threshold       : 评估阈值，默认 0.5。

    Returns:
        训练日志 DataFrame。
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    best_val_acc = 0.0
    best_epoch   = -1
    log_records: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"  开始训练: {model_name}  |  epochs={epochs}  |  device={device}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch:>3d}/{epochs}]")

        # ── 训练 ──────────────────────────────
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha,
        )

        # ── 验证 ──────────────────────────────
        val_loss, val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, device, threshold
        )

        val_auc         = val_metrics["auc"]
        val_accuracy    = val_metrics["accuracy"]
        val_sensitivity = val_metrics["sensitivity"]
        val_specificity = val_metrics["specificity"]

        # ── 测试集（每 epoch 可选）────────────────────
        test_acc = float("nan")
        test_auc = float("nan")
        if test_loader is not None:
            _, test_metrics, _, _, _ = evaluate(
                model, test_loader, criterion, device, threshold
            )
            test_acc = test_metrics["accuracy"]
            test_auc = test_metrics["auc"]

        # ── 学习率调度 ─────────────────────────
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)
            else:
                scheduler.step()

        # ── 打印日志 ───────────────────────────
        lr_now = optimizer.param_groups[0]["lr"]
        test_str = (
            f"  test_acc={test_acc:.4f}  test_auc={test_auc:.4f}"
            if test_loader is not None else ""
        )
        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_auc={val_auc:.4f}  val_acc={val_accuracy:.4f}"
            f"  sens={val_sensitivity:.4f}  spec={val_specificity:.4f}"
            f"{test_str}"
            f"  lr={lr_now:.2e}"
        )

        # ── 保存 best checkpoint ───────────────
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch   = epoch
            torch.save(
                {
                    "model_name":       model_name,
                    "epoch":            epoch,
                    "best_val_acc":     best_val_acc,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            print(f"  >> Best checkpoint saved (val_acc={best_val_acc:.4f}, epoch={epoch})")

        # ── 记录日志 ───────────────────────────
        log_records.append({
            "epoch":           epoch,
            "train_loss":      round(train_loss,      4),
            "train_accuracy":  round(train_accuracy,  4),
            "val_loss":        round(val_loss,        4),
            "val_accuracy":    round(val_accuracy,    4),
            "val_sensitivity": round(val_sensitivity, 4),
            "val_specificity": round(val_specificity, 4),
            "val_auc":         round(val_auc, 4) if not np.isnan(val_auc) else float("nan"),
            "test_accuracy":   round(test_acc, 4) if not np.isnan(test_acc) else float("nan"),
            "test_auc":        round(test_auc, 4) if not np.isnan(test_auc) else float("nan"),
            "lr":              lr_now,
        })

    # ── 保存训练日志 CSV ───────────────────────
    log_df = pd.DataFrame(log_records)
    if log_path:
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  训练完成: {model_name}")
    print(f"  Best epoch={best_epoch}  Best val_acc={best_val_acc:.4f}")
    if log_path:
        print(f"  日志已保存: {log_path}")
    print(f"{'='*60}\n")

    return log_df


# ─────────────────────────────────────────────
# 4. 加载 Checkpoint
# ─────────────────────────────────────────────

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    """从文件加载 best checkpoint 的权重到模型。

    Args:
        model           : 结构与 checkpoint 一致的模型实例。
        checkpoint_path : .pth 文件路径。
        device          : 目标设备。

    Returns:
        checkpoint dict（包含 model_name / epoch / best_val_auc 等元信息）。

    Raises:
        FileNotFoundError: checkpoint 文件不存在。
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(
        f"[Engine] 已加载 checkpoint: {checkpoint_path}"
        f"  (model={checkpoint.get('model_name', '?')}"
        f", epoch={checkpoint.get('epoch', '?')}"
        f", best_val_acc={checkpoint.get('best_val_acc', checkpoint.get('best_val_auc', '?'))})"
    )
    return checkpoint
