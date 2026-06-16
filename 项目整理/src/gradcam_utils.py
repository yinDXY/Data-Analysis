"""
gradcam_utils.py — Grad-CAM 可解释性工具模块

依赖：
    pip install grad-cam opencv-python

用途：
    为 pneumonia_baseline 中的单 logit 二分类模型生成 Grad-CAM 热力图，
    辅助判断模型是否主要关注肺部区域。

模型输出约定：
    - 单 logit，shape [B] 或 [B, 1]
    - PNEUMONIA class score = +logit
    - NORMAL    class score = -logit
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LABEL_NAME = {0: "NORMAL", 1: "PNEUMONIA"}


# ─────────────────────────────────────────────
# 1. 自定义 Grad-CAM 目标类别
# ─────────────────────────────────────────────

class BinaryClassifierOutputTarget:
    """单 logit 二分类 Grad-CAM 目标。

    模型输出 1 个 logit，代表 PNEUMONIA 的得分。
    - PNEUMONIA：梯度来自 +logit
    - NORMAL   ：梯度来自 -logit（对 logit 取反，使 NORMAL 方向梯度上升）

    兼容 output shape [B] 和 [B, 1]。

    Args:
        target_label: 0 = NORMAL, 1 = PNEUMONIA。
    """

    def __init__(self, target_label: int) -> None:
        if target_label not in (0, 1):
            raise ValueError(f"target_label 必须是 0 或 1，收到: {target_label}")
        self.target_label = target_label

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        # 兼容 [B] 和 [B, 1]
        if output.dim() == 2:
            logit = output[:, 0]
        else:
            logit = output

        if self.target_label == 1:   # PNEUMONIA
            return logit
        else:                        # NORMAL
            return -logit


# ─────────────────────────────────────────────
# 2. 获取目标层
# ─────────────────────────────────────────────

def get_target_layers(model: nn.Module, model_name: str) -> list:
    """返回 Grad-CAM 使用的目标层列表。

    每种模型选取最后一个卷积特征块：
      - resnet50      : model.layer4[-1]
                        最后一个 Bottleneck，包含 3×3 conv + BN，语义最丰富。
      - densenet121   : model.features.denseblock4
                        最后一个 DenseBlock，输出 1024 通道特征图。
      - efficientnet_b0: model.features[-1]
                        最后一个 MBConv 组（包含 Conv2d + BN + SiLU），
                        捕获高层语义特征。

    Args:
        model      : 已实例化的模型（未必加载权重）。
        model_name : "resnet50" / "densenet121" / "efficientnet_b0"。

    Returns:
        list of nn.Module，传给 GradCAM(target_layers=...)。

    Raises:
        ValueError: 不支持的 model_name。
    """
    model_name = model_name.lower()
    if model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "densenet121":
        return [model.features.denseblock4]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]
    else:
        raise ValueError(
            f"不支持的 model_name: '{model_name}'。"
            f"可选: resnet50 / densenet121 / efficientnet_b0"
        )


# ─────────────────────────────────────────────
# 3. 模型 Wrapper（使 GradCAM 兼容 [B] 输出）
# ─────────────────────────────────────────────

class GradCAMModelWrapper(nn.Module):
    """将 shape [B] 输出转为 [B, 1]，满足 pytorch-grad-cam 的期望格式。

    pytorch-grad-cam 内部对 output 做 output[:, class_idx] 索引，
    因此要求输出至少是二维张量。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)   # [B] → [B, 1]
        return out


# ─────────────────────────────────────────────
# 4. 图像反归一化
# ─────────────────────────────────────────────

def denormalize_image(
    tensor: torch.Tensor,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray  = IMAGENET_STD,
) -> np.ndarray:
    """将 ImageNet 归一化的 Tensor 反归一化回 [0, 1] RGB 浮点图像。

    Args:
        tensor : [C, H, W] 或 [1, C, H, W] float Tensor。
        mean   : 归一化均值，长度 3。
        std    : 归一化标准差，长度 3。

    Returns:
        np.ndarray, shape [H, W, 3], dtype float32, 值域 [0, 1]。
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy()         # [C, H, W]
    img = img.transpose(1, 2, 0)               # [H, W, C]
    img = img * std[None, None, :] + mean[None, None, :]
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


# ─────────────────────────────────────────────
# 5. Grad-CAM 热力图生成（单样本）
# ─────────────────────────────────────────────

def generate_cam_for_sample(
    cam: GradCAM,
    input_tensor: torch.Tensor,
    target_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """对单张图生成 Grad-CAM 热力图和叠加图。

    Args:
        cam          : 已初始化的 GradCAM 对象。
        input_tensor : [1, C, H, W] Tensor，归一化后的输入。
        target_label : 0 = NORMAL, 1 = PNEUMONIA。

    Returns:
        grayscale_cam : [H, W] float32，热力图（0~1）。
        rgb_img       : [H, W, 3] float32，反归一化原图（0~1）。
    """
    targets = [BinaryClassifierOutputTarget(target_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]             # [H, W]
    rgb_img = denormalize_image(input_tensor)    # [H, W, 3]
    return grayscale_cam, rgb_img


# ─────────────────────────────────────────────
# 6. 图像保存
# ─────────────────────────────────────────────

def save_gradcam_images(
    output_dir: str,
    sample_id: str,
    rgb_img: np.ndarray,
    grayscale_cam: np.ndarray,
    true_class: str,
    pred_class: str,
    pred_prob: float,
    correctness: str,
) -> None:
    """保存原图、热力图、叠加图和 1×3 panel 图。

    文件命名：
        {sample_id}_original.png
        {sample_id}_heatmap.png
        {sample_id}_overlay.png
        {sample_id}_panel.png

    Args:
        output_dir    : 保存目录（会自动创建）。
        sample_id     : 样本标识符，用于文件名前缀。
        rgb_img       : [H, W, 3] float32，反归一化原图，值域 [0, 1]。
        grayscale_cam : [H, W] float32，Grad-CAM 热力图，值域 [0, 1]。
        true_class    : "NORMAL" 或 "PNEUMONIA"。
        pred_class    : "NORMAL" 或 "PNEUMONIA"。
        pred_prob     : 预测为 PNEUMONIA 的概率（0~1）。
        correctness   : "TP" / "TN" / "FP" / "FN"。
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 生成热力图和叠加图 ────────────────────
    heatmap_colored = _cam_to_heatmap(grayscale_cam)   # [H, W, 3] uint8
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)  # [H, W, 3] uint8

    # ── 各自保存 ─────────────────────────────
    original_uint8 = (rgb_img * 255).astype(np.uint8)
    _save_rgb(os.path.join(output_dir, f"{sample_id}_original.png"), original_uint8)
    _save_rgb(os.path.join(output_dir, f"{sample_id}_heatmap.png"),  heatmap_colored)
    _save_rgb(os.path.join(output_dir, f"{sample_id}_overlay.png"),  overlay)

    # ── 1×3 panel ────────────────────────────
    title = (
        f"True: {true_class} | Pred: {pred_class} | "
        f"Prob: {pred_prob:.3f} | {correctness}"
    )
    _save_panel(
        path=os.path.join(output_dir, f"{sample_id}_panel.png"),
        original=original_uint8,
        heatmap=heatmap_colored,
        overlay=overlay,
        title=title,
    )


def _cam_to_heatmap(grayscale_cam: np.ndarray) -> np.ndarray:
    """将灰度 Grad-CAM（0~1）转为彩色热力图（RGB uint8）。"""
    cam_uint8 = (grayscale_cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def _save_rgb(path: str, img_uint8: np.ndarray) -> None:
    """保存 RGB uint8 图像到 PNG。"""
    Image.fromarray(img_uint8).save(path)


def _save_panel(
    path: str,
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    title: str,
) -> None:
    """保存 1×3 对比 panel。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, subtitle in zip(
        axes,
        [original, heatmap, overlay],
        ["Original", "Grad-CAM Heatmap", "Overlay"],
    ):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# 7. 样本选择
# ─────────────────────────────────────────────

def select_samples(
    predictions_df,          # pd.DataFrame
    sample_mode: str,
    num_samples: int,
):
    """根据 sample_mode 从预测结果中选取目标样本。

    Args:
        predictions_df : 含列 [image_path, true_label, true_class,
                          pred_prob, pred_label, pred_class, correctness] 的 DataFrame。
        sample_mode    : "mixed" / "tp" / "tn" / "fp" / "fn" / "all"。
        num_samples    : 最多选取的总样本数。

    Returns:
        pd.DataFrame，已选样本子集（保留原索引）。
    """
    import pandas as pd

    mode = sample_mode.lower()

    if mode == "all":
        return predictions_df.head(num_samples).copy()

    if mode in ("tp", "tn", "fp", "fn"):
        subset = predictions_df[
            predictions_df["correctness"].str.upper() == mode.upper()
        ]
        subset = _sort_by_confidence(subset, mode.upper())
        return subset.head(num_samples).copy()

    if mode == "mixed":
        per_class = max(1, num_samples // 4)
        parts = []
        for label in ("TP", "TN", "FP", "FN"):
            subset = predictions_df[predictions_df["correctness"] == label]
            subset = _sort_by_confidence(subset, label)
            parts.append(subset.head(per_class))

        selected = pd.concat(parts, ignore_index=True)

        # 若总数不足 num_samples，用剩余样本补齐
        if len(selected) < num_samples:
            selected_paths = set(selected["image_path"])
            rest = predictions_df[~predictions_df["image_path"].isin(selected_paths)]
            need = num_samples - len(selected)
            selected = pd.concat([selected, rest.head(need)], ignore_index=True)

        return selected.head(num_samples).copy()

    raise ValueError(
        f"不支持的 sample_mode: '{sample_mode}'。"
        f"可选: mixed / tp / tn / fp / fn / all"
    )


def _sort_by_confidence(df, label: str):
    """置信度优先排序。TP/FP 按 pred_prob 降序；TN/FN 按 pred_prob 升序。"""
    if label in ("TP", "FP"):
        return df.sort_values("pred_prob", ascending=False)
    else:
        return df.sort_values("pred_prob", ascending=True)


# ─────────────────────────────────────────────
# 8. 加载 Checkpoint
# ─────────────────────────────────────────────

def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """加载 best checkpoint 并返回 eval 模式的模型。

    兼容两种 checkpoint 格式：
      - dict 含 "model_state_dict" 键（标准格式）
      - 直接是 state_dict

    Args:
        model_name      : "resnet50" / "densenet121" / "efficientnet_b0"。
        checkpoint_path : .pth 文件路径。
        device          : 运行设备。

    Returns:
        nn.Module，已加载权重、eval 模式。
    """
    from src.models import get_model

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

    model = get_model(model_name, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        ckpt_info = (
            f"  epoch={checkpoint.get('epoch', '?')}  "
            f"best_val_auc={checkpoint.get('best_val_auc', '?')}"
        )
    else:
        # 兼容直接保存 state_dict 的情况
        state_dict = checkpoint
        ckpt_info = ""

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[GradCAM] 已加载 checkpoint: {checkpoint_path}{ckpt_info}")
    return model
