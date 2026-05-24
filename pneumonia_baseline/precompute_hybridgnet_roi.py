#!/usr/bin/env python
"""
precompute_hybridgnet_roi.py — HybridGNet 肺部 ROI 批量预处理脚本

使用 HybridGNet 从 Chest X-Ray 数据集中提取左右肺 ROI，
生成与原始数据集目录结构相同的新数据集，可直接传给 train_baselines.py。

用法（小规模测试）：
    python precompute_hybridgnet_roi.py \\
      --data_dir ./dataset/chest_xray \\
      --hybridgnet_dir ../HybridGNet \\
      --weights_path ../HybridGNet/weights/weights.pt \\
      --output_dir ./dataset/chest_xray_hybridgnet_roi_test \\
      --max_samples 20 --save_masks --save_visualizations --overwrite

用法（全量）：
    python precompute_hybridgnet_roi.py \\
      --data_dir ./dataset/chest_xray \\
      --hybridgnet_dir ../HybridGNet \\
      --weights_path ../HybridGNet/weights/weights.pt \\
      --output_dir ./dataset/chest_xray_hybridgnet_roi \\
      --save_masks --save_visualizations --overwrite
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# 确保项目根在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybridgnet_adapter import HybridGNetSegmenter
from src.roi_preprocessing import (
    apply_roi_crop,
    get_bbox_from_mask,
    postprocess_lung_mask,
    preprocess_one_image,
)


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HybridGNet 肺部 ROI 批量预处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",       type=str, required=True,
                        help="原始 chest_xray 根目录（含 train/val/test）")
    parser.add_argument("--hybridgnet_dir", type=str, required=True,
                        help="HybridGNet 仓库根目录路径")
    parser.add_argument("--weights_path",   type=str, required=True,
                        help="HybridGNet 预训练权重 (.pt) 路径")
    parser.add_argument("--output_dir",     type=str,
                        default="./dataset/chest_xray_hybridgnet_roi",
                        help="输出数据集根目录")
    parser.add_argument("--margin_ratio",   type=float, default=0.08,
                        help="ROI bounding box 外扩比例")
    parser.add_argument("--fallback",       type=str,   default="original",
                        choices=["original", "skip"],
                        help="预处理失败时的处理策略")
    parser.add_argument("--device",         type=str,   default=None,
                        help="推理设备（cuda/cpu），默认自动检测")
    parser.add_argument("--overwrite",      action="store_true",
                        help="若输出目录已存在则清空后重建")
    parser.add_argument("--save_masks",     action="store_true",
                        help="保存 binary lung mask 到 output_dir/masks/")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="保存可视化（原图 | mask | 裁剪）到 output_dir/visualizations/")
    parser.add_argument("--max_samples",    type=int,   default=None,
                        help="处理图像总数上限（用于测试，None=全量）")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 目录工具
# ─────────────────────────────────────────────────────────────────────────────

SPLITS      = ["train", "val", "test"]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LABEL_MAP   = {"NORMAL": 0, "PNEUMONIA": 1}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def collect_image_list(data_dir: str) -> List[Tuple[str, str, str]]:
    """遍历 data_dir/{split}/{class}/ 收集所有图像路径。

    Returns:
        [(image_path, split, class_name), ...]
    """
    records = []
    for split in SPLITS:
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(data_dir, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                    records.append((
                        os.path.join(cls_dir, fname),
                        split,
                        cls,
                    ))
    return records


def setup_output_dir(output_dir: str, overwrite: bool) -> None:
    """创建输出目录结构，视 overwrite 标志决定是否清空已有目录。"""
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}\n"
                "使用 --overwrite 标志以清空并重建。"
            )
        print(f"[setup] 清空已有输出目录: {output_dir}")
        shutil.rmtree(output_dir)

    for split in SPLITS:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_visualization(
    orig_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    cropped_bgr: np.ndarray,
    vis_path: str,
) -> None:
    """保存三格对比图：原图 | lung mask | ROI 裁剪。"""
    # 目标高度统一
    target_h = 320

    def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        w = max(1, int(img.shape[1] * ratio))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # 原图（灰度→RGB）
    if orig_bgr.ndim == 2:
        orig_vis = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)
    else:
        orig_vis = orig_bgr.copy()
    orig_vis = _resize_to_height(orig_vis, target_h)

    # Mask 可视化（叠加热力色）
    if mask is not None:
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_color = cv2.applyColorMap(mask_u8, cv2.COLORMAP_BONE)
        mask_color = _resize_to_height(mask_color, target_h)
    else:
        mask_color = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        cv2.putText(mask_color, "no mask", (10, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 裁剪图
    if cropped_bgr.ndim == 2:
        crop_vis = cv2.cvtColor(cropped_bgr, cv2.COLOR_GRAY2BGR)
    else:
        crop_vis = cropped_bgr.copy()
    crop_vis = _resize_to_height(crop_vis, target_h)

    # 水平拼接
    panel = cv2.hconcat([orig_vis, mask_color, crop_vis])

    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    cv2.imwrite(vis_path, panel)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 设备 ───────────────────────────────────────────────────
    import torch
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] {device}")

    # ── 检查原始数据目录 ──────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir 不存在: {args.data_dir}")

    # ── 设置输出目录 ─────────────────────────────────────────
    setup_output_dir(args.output_dir, args.overwrite)

    # ── 构建 segmenter ────────────────────────────────────────
    print("[init] 正在加载 HybridGNet 模型…")
    segmenter = HybridGNetSegmenter(
        hybridgnet_dir=args.hybridgnet_dir,
        weights_path=args.weights_path,
        device=device,
    )
    print("[init] 模型加载完毕")

    # ── 收集图像列表 ──────────────────────────────────────────
    image_list = collect_image_list(args.data_dir)
    if not image_list:
        raise RuntimeError(f"在 {args.data_dir} 下未找到任何图像文件。")

    if args.max_samples is not None:
        image_list = image_list[:args.max_samples]
        print(f"[info] max_samples={args.max_samples}，共处理 {len(image_list)} 张图像")
    else:
        print(f"[info] 共发现 {len(image_list)} 张图像")

    # ── 批量处理 ──────────────────────────────────────────────
    metadata_rows = []
    t0 = time.time()
    n_success = n_fallback = n_skipped = n_failed = 0

    for i, (img_path, split, cls_name) in enumerate(image_list, 1):
        fname = os.path.basename(img_path)
        stem  = os.path.splitext(fname)[0]
        ext   = os.path.splitext(fname)[1].lower()

        # 输出路径
        out_img_path  = os.path.join(args.output_dir, split, cls_name, fname)
        mask_path_rel = ""
        if args.save_masks:
            mask_path_rel = os.path.join(
                args.output_dir, "masks", split, cls_name, stem + "_mask.png"
            )

        # ── 预处理 ──────────────────────────────────────────
        processed, meta = preprocess_one_image(
            image_path=img_path,
            segmenter=segmenter,
            margin_ratio=args.margin_ratio,
            fallback=args.fallback,
        )

        status = meta["status"]

        # ── 计数 ────────────────────────────────────────────
        if status == "success":
            n_success += 1
        elif status == "fallback":
            n_fallback += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            n_failed += 1

        # ── 保存输出图像 ─────────────────────────────────────
        saved_output_path = ""
        if processed is not None and status != "skipped":
            cv2.imwrite(out_img_path, processed)
            saved_output_path = out_img_path
        elif status == "skipped":
            # skip 不写任何文件
            pass
        else:
            # 其他 failed 不保存
            pass

        # ── 保存 mask ────────────────────────────────────────
        saved_mask_path = ""
        if args.save_masks and status == "success":
            # 重新预测一次以获取 mask（复用之前的 mask）
            # 为避免重复推理，在 fallback 情况下 mask 可能不可用
            # 此处只在 success 时保存 mask（mask 已通过后处理）
            try:
                mask_raw  = segmenter.predict_lung_mask(img_path)
                mask_proc = postprocess_lung_mask(mask_raw)
                mask_save = (mask_proc * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(mask_path_rel), exist_ok=True)
                cv2.imwrite(mask_path_rel, mask_save)
                saved_mask_path = mask_path_rel
            except Exception:
                pass  # mask 保存失败不中断流程

        # ── 保存可视化 ────────────────────────────────────────
        if args.save_visualizations and processed is not None:
            vis_path = os.path.join(
                args.output_dir, "visualizations", split, cls_name,
                stem + "_vis.jpg"
            )
            try:
                orig_bgr = cv2.imread(img_path)
                mask_for_vis = None
                if status == "success":
                    try:
                        mv = segmenter.predict_lung_mask(img_path)
                        mask_for_vis = postprocess_lung_mask(mv)
                    except Exception:
                        pass
                save_visualization(orig_bgr, mask_for_vis, processed, vis_path)
            except Exception:
                pass  # 可视化失败不中断流程

        # ── 记录 metadata ─────────────────────────────────────
        metadata_rows.append({
            "split":           split,
            "class_name":      cls_name,
            "label":           LABEL_MAP.get(cls_name, -1),
            "original_path":   meta["original_path"],
            "output_path":     saved_output_path,
            "mask_path":       saved_mask_path,
            "status":          status,
            "bbox_x1":         meta["bbox_x1"],
            "bbox_y1":         meta["bbox_y1"],
            "bbox_x2":         meta["bbox_x2"],
            "bbox_y2":         meta["bbox_y2"],
            "mask_area_ratio": meta["mask_area_ratio"],
            "error_message":   meta["error_message"],
        })

        # ── 进度打印 ─────────────────────────────────────────
        if i % 50 == 0 or i == len(image_list):
            elapsed = time.time() - t0
            speed   = i / elapsed if elapsed > 0 else 0
            print(
                f"  [{i}/{len(image_list)}] "
                f"success={n_success} fallback={n_fallback} "
                f"skipped={n_skipped} failed={n_failed}  "
                f"({speed:.1f} img/s)"
            )

    # ── 保存 metadata ─────────────────────────────────────────
    meta_df   = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(args.output_dir, "preprocessing_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"\n[done] metadata 已保存到: {meta_path}")

    # ── 汇总 ─────────────────────────────────────────────────
    total   = len(image_list)
    elapsed = time.time() - t0
    print(
        f"\n{'─' * 55}\n"
        f"  总计处理: {total} 张\n"
        f"  success : {n_success}  ({100 * n_success / total:.1f}%)\n"
        f"  fallback: {n_fallback}  ({100 * n_fallback / total:.1f}%)\n"
        f"  skipped : {n_skipped}  ({100 * n_skipped / total:.1f}%)\n"
        f"  failed  : {n_failed}  ({100 * n_failed / total:.1f}%)\n"
        f"  耗时    : {elapsed:.1f}s\n"
        f"  输出目录: {args.output_dir}\n"
        f"{'─' * 55}"
    )


if __name__ == "__main__":
    main()
