"""
losses.py — 损失函数工厂（C 模块：类别不平衡损失）

支持：
  - bce           : BCEWithLogitsLoss（baseline）
  - soft_mcc      : Soft MCC Loss（直接吃 logits）
  - bce_soft_mcc  : BCE + Soft MCC 加权组合

Soft MCC 核心算法复现自：
  https://github.com/daniel-scholz/address-class-imbalance
  (torch_losses/soft_mcc.py — SoftMCCWithLogitsLoss)
不修改第三方仓库，此处为最小化内联实现。
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Soft MCC（内联复现，与第三方仓库算法完全一致）
# ─────────────────────────────────────────────

class SoftMCCWithLogitsLossWrapper(nn.Module):
    """Soft Matthews Correlation Coefficient Loss，直接接受 logits。

    内部先 sigmoid，再计算软混淆矩阵，最终返回 1 - soft_mcc。

    输入：
        logits : [B] 或 [B, 1]（自动 squeeze）
        labels : [B] 或 [B, 1]（自动 squeeze 并转 float）
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1) if logits.dim() == 2 else logits  # → [B]
        labels = labels.squeeze(1).float() if labels.dim() == 2 else labels.float()

        preds = torch.sigmoid(logits)

        tp = torch.sum(preds * labels)
        tn = torch.sum((1 - preds) * (1 - labels))
        fp = torch.sum(preds * (1 - labels))
        fn = torch.sum((1 - preds) * labels)

        numerator = tp * tn - fp * fn
        denom = (
            torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
        )
        soft_mcc = numerator / denom

        return 1.0 - soft_mcc


# ─────────────────────────────────────────────
# BCE + Soft MCC 加权组合
# ─────────────────────────────────────────────

class BCESoftMCCLoss(nn.Module):
    """loss = bce_weight * BCE + mcc_weight * SoftMCC

    两个子损失均直接接受 logits，内部不做提前 sigmoid。

    Args:
        bce_weight : BCE 项的权重（默认 1.0）
        mcc_weight : Soft MCC 项的权重（默认 1.0）
    """

    def __init__(self, bce_weight: float = 1.0, mcc_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.mcc_weight  = mcc_weight
        self._bce        = nn.BCEWithLogitsLoss()
        self._soft_mcc   = SoftMCCWithLogitsLossWrapper()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1) if logits.dim() == 2 else logits
        labels = labels.squeeze(1).float() if labels.dim() == 2 else labels.float()

        bce_loss = self._bce(logits, labels)
        mcc_loss = self._soft_mcc(logits, labels)

        return self.bce_weight * bce_loss + self.mcc_weight * mcc_loss


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

_SUPPORTED = ("bce", "soft_mcc", "bce_soft_mcc")


def get_loss_function(
    loss_name: str = "bce",
    bce_weight: float = 1.0,
    mcc_weight: float = 1.0,
) -> nn.Module:
    """返回指定的损失函数实例。

    Args:
        loss_name  : "bce" | "soft_mcc" | "bce_soft_mcc"
        bce_weight : BCE 项权重（仅 bce_soft_mcc 生效）
        mcc_weight : Soft MCC 项权重（soft_mcc / bce_soft_mcc 生效）

    Returns:
        nn.Module，可直接 criterion(logits, labels) 调用。

    Raises:
        ValueError : 不支持的 loss_name。
    """
    loss_name = loss_name.lower().strip()

    print(f"[Loss] loss_name={loss_name}, bce_weight={bce_weight}, mcc_weight={mcc_weight}")

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "soft_mcc":
        return SoftMCCWithLogitsLossWrapper()

    if loss_name == "bce_soft_mcc":
        return BCESoftMCCLoss(bce_weight=bce_weight, mcc_weight=mcc_weight)

    raise ValueError(
        f"不支持的 loss_name='{loss_name}'，可选：{_SUPPORTED}"
    )
