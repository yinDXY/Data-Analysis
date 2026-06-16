"""
roi_preprocessing.py — 肺部 ROI 裁剪工具

基于 HybridGNet 预测的 lung mask 对胸部 X-ray 进行 ROI 裁剪。

主要函数：
  postprocess_lung_mask  — mask 后处理（形态学闭运算 + 填洞）
  get_bbox_from_mask     — 从 mask 计算 bounding box（含 margin）
  apply_roi_crop         — 按 bbox 裁剪图像
  preprocess_one_image   — 完整单图预处理流程（含 fallback 逻辑）
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# mask 有效性阈值
# ─────────────────────────────────────────────────────────────────────────────
_MASK_RATIO_MIN = 0.05   # mask 占图像面积的最低比例
_MASK_RATIO_MAX = 0.80   # mask 占图像面积的最高比例（超出视为预测异常）


# ─────────────────────────────────────────────────────────────────────────────
# 1. mask 后处理
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_lung_mask(mask: np.ndarray) -> np.ndarray:
    """对 binary lung mask 进行轻量后处理。

    操作：
      1. 二值化（任何非零值 → 1）
      2. 形态学闭运算（填充轮廓小缺口）
      3. binary_fill_holes（填充肺内空洞）

    Args:
        mask: uint8 ndarray，shape (H, W)，值 0 或 1。

    Returns:
        cleaned: uint8 ndarray，shape (H, W)，值 0 或 1。
    """
    binary = (mask > 0).astype(np.uint8)

    # 闭运算：先膨胀再腐蚀，填补轮廓的小缺口
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 填充封闭区域内的空洞
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bounding box
# ─────────────────────────────────────────────────────────────────────────────

def get_bbox_from_mask(
    mask: np.ndarray,
    margin_ratio: float = 0.08,
) -> Optional[Tuple[int, int, int, int]]:
    """从 lung mask 计算带 margin 的 bounding box。

    Args:
        mask         : uint8 ndarray (H, W)，值 0 或 1。
        margin_ratio : 在 bbox 四边各添加边长的比例作为 margin。

    Returns:
        (x1, y1, x2, y2) 整数坐标，若 mask 全零或 bbox 退化则返回 None。
        坐标已裁剪到图像范围内。
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])

    bbox_h = y_max - y_min
    bbox_w = x_max - x_min

    # bbox 退化检查
    if bbox_h <= 0 or bbox_w <= 0:
        return None

    img_h, img_w = mask.shape

    margin_y = max(1, int(bbox_h * margin_ratio))
    margin_x = max(1, int(bbox_w * margin_ratio))

    x1 = max(0, x_min - margin_x)
    y1 = max(0, y_min - margin_y)
    x2 = min(img_w - 1, x_max + margin_x)
    y2 = min(img_h - 1, y_max + margin_y)

    # 最终合法性检查
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROI crop
# ─────────────────────────────────────────────────────────────────────────────

def apply_roi_crop(
    image: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.08,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """根据 lung mask 的 bounding box 裁剪图像。

    Args:
        image        : 原始图像 ndarray（任意 shape，至少 2D）。
        mask         : uint8 ndarray (H, W)，值 0 或 1。
        margin_ratio : 传递给 get_bbox_from_mask。

    Returns:
        (cropped_image, bbox)
        若 bbox 无效，返回 (原图, None)。
    """
    bbox = get_bbox_from_mask(mask, margin_ratio)
    if bbox is None:
        return image, None

    x1, y1, x2, y2 = bbox
    # numpy 切片：行=y，列=x
    if image.ndim == 2:
        cropped = image[y1:y2 + 1, x1:x2 + 1]
    else:
        cropped = image[y1:y2 + 1, x1:x2 + 1, :]

    return cropped, bbox


# ─────────────────────────────────────────────────────────────────────────────
# 4. 完整单图预处理
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_one_image(
    image_path: str,
    segmenter,                       # HybridGNetSegmenter 实例
    margin_ratio: float = 0.08,
    fallback: str = "original",      # "original" | "skip"
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """对单张图像执行 HybridGNet 分割 → mask 后处理 → ROI 裁剪。

    Args:
        image_path   : 输入图像路径。
        segmenter    : HybridGNetSegmenter 实例。
        margin_ratio : ROI bounding box 的外扩比例。
        fallback     : mask 异常或推理失败时的处理策略：
                       "original" — 返回原图；
                       "skip"     — 返回 None（该图不被写入输出集）。

    Returns:
        (processed_image, metadata)
        processed_image: ndarray 或 None（skip 时）。
        metadata: dict，字段详见下方 Returns 说明。

    metadata 字段：
        original_path  : 输入路径（str）
        status         : "success" / "fallback" / "skipped" / "failed"
        bbox_x1/y1/x2/y2 : 裁剪框坐标（int，失败时为 -1）
        mask_area_ratio: float，失败时为 -1.0
        error_message  : str，无错误时为 ""
    """
    image_path = str(image_path)
    meta: Dict[str, Any] = {
        "original_path":  image_path,
        "status":         "failed",
        "bbox_x1":        -1,
        "bbox_y1":        -1,
        "bbox_x2":        -1,
        "bbox_y2":        -1,
        "mask_area_ratio": -1.0,
        "error_message":  "",
    }

    # ── 读取原图（BGR，用于保存时保留原色彩空间）──────────────
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        meta["error_message"] = f"无法读取图像: {image_path}"
        if fallback == "skip":
            meta["status"] = "skipped"
            return None, meta
        meta["status"] = "fallback"
        return None, meta   # 连原图都读不到就真的失败

    orig_h, orig_w = orig_bgr.shape[:2]
    total_pixels = float(orig_h * orig_w)

    # ── HybridGNet 推理 ───────────────────────────────────────
    try:
        mask_raw = segmenter.predict_lung_mask(image_path)
    except Exception as e:
        meta["error_message"] = f"HybridGNet 推理失败: {e}"
        if fallback == "skip":
            meta["status"] = "skipped"
            return None, meta
        meta["status"] = "fallback"
        return orig_bgr, meta

    # ── mask 后处理 ───────────────────────────────────────────
    mask = postprocess_lung_mask(mask_raw)

    # ── 有效性检查 ────────────────────────────────────────────
    mask_area_ratio = float(mask.sum()) / total_pixels
    meta["mask_area_ratio"] = round(mask_area_ratio, 4)

    invalid_ratio = (
        mask_area_ratio < _MASK_RATIO_MIN
        or mask_area_ratio > _MASK_RATIO_MAX
    )

    bbox = get_bbox_from_mask(mask, margin_ratio)
    invalid_bbox = bbox is None

    if invalid_ratio or invalid_bbox:
        reason_parts = []
        if invalid_ratio:
            reason_parts.append(
                f"mask_area_ratio={mask_area_ratio:.4f} "
                f"(期望 [{_MASK_RATIO_MIN}, {_MASK_RATIO_MAX}])"
            )
        if invalid_bbox:
            reason_parts.append("bbox 无效（mask 全零或退化）")
        meta["error_message"] = "; ".join(reason_parts)

        if fallback == "skip":
            meta["status"] = "skipped"
            return None, meta
        meta["status"] = "fallback"
        return orig_bgr, meta

    # ── ROI 裁剪 ──────────────────────────────────────────────
    x1, y1, x2, y2 = bbox
    meta["bbox_x1"] = x1
    meta["bbox_y1"] = y1
    meta["bbox_x2"] = x2
    meta["bbox_y2"] = y2

    cropped, _ = apply_roi_crop(orig_bgr, mask, margin_ratio)
    meta["status"] = "success"

    return cropped, meta
