#!/usr/bin/env python
"""
test.py — 单张 / 批量图像推理 + 可视化面板

直接修改下方 CONFIG，然后运行：
    python test.py

可视化面板内容：
  - 原始 X-Ray 图像
  - Grad-CAM 热力图（标注模型关注区域）
  - Grad-CAM 叠加图
  - 预测信息面板（结果、概率条、真实标签）
  - 底部概率仪表盘（连续渐变 + 指针 + 阈值线）

批量模式（input_path 为目录）时额外输出 predictions.csv。
"""

import io
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.gradcam_utils import (
    GradCAMModelWrapper,
    BinaryClassifierOutputTarget,
    get_target_layers,
    denormalize_image,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_NAME,
)

# ─── 中文字体（Windows / Linux 均可）─────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                    "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════
#  CONFIG：修改此处，直接 python test.py 运行
# ══════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── 输入 ────────────────────────────────────────────────────
    # 单张图像路径  OR  包含图像的目录（递归搜索 jpg/png/bmp）
    input_path      = r"D:\数据挖掘课设\dataset\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg",

    # ── 模型 ────────────────────────────────────────────────────
    # resnet50 / densenet121 / efficientnet_b0
    model_name      = "densenet121",

    # 训练好的 best checkpoint 路径（相对路径或绝对路径均可）
    checkpoint_path = r"results_split\checkpoints\densenet121_best.pth",

    # ── 推理参数 ─────────────────────────────────────────────────
    # sigmoid 输出 >= threshold → PNEUMONIA（建议与训练/threshold search 一致）
    threshold       = 0.5,

    # 模型输入尺寸（必须与训练时一致）
    image_size      = 224,

    # ── 输出 ────────────────────────────────────────────────────
    # 结果保存目录（None = 不保存文件，仅打印）
    output_dir      = r"results_split\single_test",

    # ── Grad-CAM ─────────────────────────────────────────────────
    # True = 生成热力图（需安装 grad-cam 库：pip install grad-cam）
    use_gradcam     = True,

    # predicted  → 对模型预测类别生成热力图（推荐）
    # pneumonia  → 固定对 PNEUMONIA 方向
    # normal     → 固定对 NORMAL 方向
    target_class    = "predicted",

    # ── 设备 ────────────────────────────────────────────────────
    # auto / cpu / cuda
    device          = "auto",
)


# ══════════════════════════════════════════════════════════════
#  主题色
# ══════════════════════════════════════════════════════════════

_BG    = "#1A1A2E"   # 背景
_PANEL = "#16213E"   # 面板底色
_ACCE  = "#0F3460"   # 强调色（框）
_FG    = "#E0E0E0"   # 主文字
_GRAY  = "#888888"   # 次要文字
_RED   = "#E74C3C"   # PNEUMONIA
_GREEN = "#2ECC71"   # NORMAL

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def resolve_device(d: str) -> torch.device:
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def load_model(model_name: str, ckpt_path: str, device: torch.device) -> nn.Module:
    model = get_model(model_name, pretrained=False)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[Model] {model_name}  ←  {ckpt_path}")
    return model


def get_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def collect_images(path: str) -> list:
    if os.path.isfile(path):
        return [path]
    imgs = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                imgs.append(os.path.join(root, f))
    return imgs


def infer_true_label(image_path: str):
    """从父目录名自动推断真实标签（NORMAL=0 / PNEUMONIA=1 / None=未知）。"""
    parent = os.path.basename(os.path.dirname(image_path)).upper()
    return {"NORMAL": 0, "PNEUMONIA": 1}.get(parent)


def infer_single(model, transform, image_path: str, device):
    """单张图像推理，返回 (input_tensor, logit, prob)。"""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor).item()
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return tensor, logit, prob


def get_gradcam(model, model_name: str, input_tensor, target_label: int):
    """生成 Grad-CAM，返回 (grayscale_cam, rgb_img) 或 (None, rgb_img)。"""
    try:
        from pytorch_grad_cam import GradCAM
        wrapped = GradCAMModelWrapper(model)
        target_layers = get_target_layers(wrapped.model, model_name)
        targets = [BinaryClassifierOutputTarget(target_label)]
        with GradCAM(model=wrapped, target_layers=target_layers) as cam:
            gray_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        return gray_cam, denormalize_image(input_tensor)
    except Exception as e:
        print(f"  [Grad-CAM] 生成失败: {e}")
        return None, denormalize_image(input_tensor)


# ══════════════════════════════════════════════════════════════
#  子图绘制函数
# ══════════════════════════════════════════════════════════════

def _draw_image_panel(ax, rgb_img: np.ndarray, title: str):
    """显示图像，深色面板风格。"""
    ax.imshow(rgb_img)
    ax.set_title(title, color=_FG, fontsize=11, pad=5, fontweight="bold")
    ax.axis("off")
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(_ACCE)


def _draw_heatmap_panel(ax, grayscale_cam: np.ndarray):
    """绘制 Grad-CAM 纯热力图 + colorbar。"""
    im = ax.imshow(grayscale_cam, cmap="jet", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color=_FG, labelcolor=_FG, labelsize=8)
    cbar.outline.set_edgecolor(_FG)
    ax.set_title("Grad-CAM 热力图", color=_FG, fontsize=11, pad=5, fontweight="bold")
    ax.axis("off")
    ax.set_facecolor(_PANEL)


def _draw_overlay_panel(ax, rgb_img: np.ndarray, grayscale_cam: np.ndarray,
                        pred_label: int):
    """绘制 Grad-CAM 叠加图，带预测标签角标。"""
    from pytorch_grad_cam.utils.image import show_cam_on_image
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    ax.imshow(overlay)
    # 左上角标签角标
    label_color = _RED if pred_label == 1 else _GREEN
    label_text  = "PNEUMONIA" if pred_label == 1 else "NORMAL"
    ax.text(0.03, 0.97, label_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=label_color,
                      edgecolor="none", alpha=0.85))
    ax.set_title("Grad-CAM 叠加图", color=_FG, fontsize=11, pad=5, fontweight="bold")
    ax.axis("off")
    ax.set_facecolor(_PANEL)


def _draw_info_panel(ax, prob: float, pred_label: int, true_label,
                     threshold: float, model_name: str):
    """绘制预测信息面板：结果大字 + 概率条 + 真实标签。"""
    ax.set_facecolor(_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pred_c    = _RED if pred_label == 1 else _GREEN
    pred_name = "PNEUMONIA\n（肺炎）" if pred_label == 1 else "NORMAL\n（正常）"

    # ── 预测结果大字 ─────────────────────────────────
    ax.text(0.5, 0.87, pred_name,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=pred_c,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=_ACCE,
                      edgecolor=pred_c, linewidth=2.2))

    # ── PNEUMONIA 概率条 ──────────────────────────────
    ax.text(0.05, 0.70, "PNEUMONIA", ha="left", va="center",
            transform=ax.transAxes, fontsize=8.5, color=_RED)
    ax.add_patch(Rectangle((0.05, 0.62), 0.90, 0.055,
                            transform=ax.transAxes,
                            facecolor="#2a2a2a", zorder=1))
    ax.add_patch(Rectangle((0.05, 0.62), 0.90 * prob, 0.055,
                            transform=ax.transAxes,
                            facecolor=_RED, alpha=0.85, zorder=2))
    ax.text(0.96, 0.647, f"{prob:.4f}", ha="right", va="center",
            transform=ax.transAxes, fontsize=9, color=_RED,
            fontweight="bold", zorder=3)

    # ── NORMAL 概率条 ─────────────────────────────────
    ax.text(0.05, 0.56, "NORMAL", ha="left", va="center",
            transform=ax.transAxes, fontsize=8.5, color=_GREEN)
    ax.add_patch(Rectangle((0.05, 0.48), 0.90, 0.055,
                            transform=ax.transAxes,
                            facecolor="#2a2a2a", zorder=1))
    ax.add_patch(Rectangle((0.05, 0.48), 0.90 * (1 - prob), 0.055,
                            transform=ax.transAxes,
                            facecolor=_GREEN, alpha=0.85, zorder=2))
    ax.text(0.96, 0.507, f"{1 - prob:.4f}", ha="right", va="center",
            transform=ax.transAxes, fontsize=9, color=_GREEN,
            fontweight="bold", zorder=3)

    # ── 阈值 ──────────────────────────────────────────
    ax.text(0.5, 0.38, f"Threshold = {threshold:.3f}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color=_GRAY)

    # ── 真实标签 & 判断结果 ───────────────────────────
    if true_label is not None:
        correct    = pred_label == true_label
        result_str = "✓  预测正确" if correct else "✗  预测错误"
        result_c   = _GREEN if correct else _RED
        ax.text(0.5, 0.27, f"真实标签: {LABEL_NAME[true_label]}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=_FG)
        ax.text(0.5, 0.16, result_str,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=result_c)
    else:
        ax.text(0.5, 0.22, "真实标签: 未知",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=_GRAY)

    # ── 模型名 ────────────────────────────────────────
    ax.text(0.5, 0.05, f"Model: {model_name}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color=_GRAY)


def _draw_gauge(ax, prob: float, threshold: float, pred_label: int):
    """绘制底部概率仪表盘：渐变色条 + 阈值竖线 + 概率指针。"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(_PANEL)
    ax.axis("off")

    BAR_L, BAR_R = 0.07, 0.93       # 条形左右端（data 坐标）
    BAR_B, BAR_T = 0.30, 0.72       # 条形上下端
    BAR_W = BAR_R - BAR_L           # 0.86

    # ── 渐变色条 ─────────────────────────────────────
    grad = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(grad, aspect="auto",
              extent=[BAR_L, BAR_R, BAR_B, BAR_T],
              cmap="RdYlGn_r", vmin=0, vmax=1, zorder=1)

    # ── 刻度线 + 标签 ─────────────────────────────────
    for frac, lbl in [(0.0, "0.00"), (0.25, "0.25"), (0.50, "0.50"),
                      (0.75, "0.75"), (1.00, "1.00")]:
        x = BAR_L + frac * BAR_W
        ax.plot([x, x], [BAR_B, BAR_T], color="#333333", lw=0.9, zorder=2)
        ax.text(x, BAR_B - 0.06, lbl, ha="center", va="top",
                color=_FG, fontsize=8.5, zorder=3)

    # ── 阈值竖线 ─────────────────────────────────────
    thr_x = BAR_L + threshold * BAR_W
    ax.plot([thr_x, thr_x], [BAR_B - 0.08, BAR_T + 0.10],
            color="white", lw=1.8, ls="--", zorder=4)
    ax.text(thr_x, BAR_T + 0.13,
            f"Threshold\n{threshold:.2f}",
            ha="center", va="bottom", color="white",
            fontsize=8.5, zorder=5)

    # ── 概率指针（三角箭头） ───────────────────────────
    prob_x  = BAR_L + prob * BAR_W
    pred_c  = _RED if pred_label == 1 else _GREEN
    ax.annotate("",
                xy=(prob_x, BAR_B),
                xytext=(prob_x, BAR_B - 0.18),
                arrowprops=dict(arrowstyle="-|>", color=pred_c,
                                lw=2.5, mutation_scale=18),
                zorder=6)
    ax.text(prob_x, BAR_B - 0.22, f"{prob:.4f}",
            ha="center", va="top", color=pred_c,
            fontsize=12, fontweight="bold", zorder=7)

    # ── 两端标签 ─────────────────────────────────────
    ax.text(BAR_L - 0.02, (BAR_B + BAR_T) / 2, "NORMAL",
            ha="right", va="center", color=_GREEN,
            fontsize=9, fontweight="bold")
    ax.text(BAR_R + 0.02, (BAR_B + BAR_T) / 2, "PNEUMONIA",
            ha="left", va="center", color=_RED,
            fontsize=9, fontweight="bold")

    # ── 标题 ─────────────────────────────────────────
    ax.text(0.5, 0.97, "PNEUMONIA 概率仪表盘",
            ha="center", va="top", color=_FG, fontsize=10)


# ══════════════════════════════════════════════════════════════
#  可视化主函数
# ══════════════════════════════════════════════════════════════

def visualize_and_save(
    image_path: str,
    input_tensor,
    prob: float,
    pred_label: int,
    true_label,
    threshold: float,
    model_name: str,
    grayscale_cam,
    rgb_img: np.ndarray,
    save_path: str,
):
    """生成完整可视化面板并保存。"""
    has_cam = grayscale_cam is not None

    # ── 布局 ─────────────────────────────────────────────────
    # Row 0 (高): 原图 | [热图 | 叠加图] | 信息面板
    # Row 1 (矮): 概率仪表盘（全宽）
    fig = plt.figure(figsize=(18, 9), facecolor=_BG)
    gs  = gridspec.GridSpec(
        2, 4,
        figure=fig,
        height_ratios=[4, 2],
        hspace=0.10, wspace=0.06,
        left=0.03, right=0.98, top=0.92, bottom=0.04,
    )

    if has_cam:
        ax_orig    = fig.add_subplot(gs[0, 0])
        ax_heat    = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_info    = fig.add_subplot(gs[0, 3])
    else:
        ax_orig = fig.add_subplot(gs[0, 0:2])   # 原图占 2 列
        ax_info = fig.add_subplot(gs[0, 2:4])   # 信息占 2 列

    # ── 各子图 ────────────────────────────────────────────────
    _draw_image_panel(ax_orig, rgb_img, "原始 X-Ray 图像")

    if has_cam:
        _draw_heatmap_panel(ax_heat, grayscale_cam)
        _draw_overlay_panel(ax_overlay, rgb_img, grayscale_cam, pred_label)

    _draw_info_panel(ax_info, prob, pred_label, true_label, threshold, model_name)

    # ── 底部仪表盘 ────────────────────────────────────────────
    ax_gauge = fig.add_subplot(gs[1, :])
    _draw_gauge(ax_gauge, prob, threshold, pred_label)

    # ── 总标题 ────────────────────────────────────────────────
    fname    = os.path.basename(image_path)
    pred_str = "PNEUMONIA（肺炎）" if pred_label == 1 else "NORMAL（正常）"
    conf_pct = max(prob, 1 - prob) * 100
    correct_str = ""
    if true_label is not None:
        correct_str = "  ✓" if pred_label == true_label else "  ✗"
    fig.suptitle(
        f"{fname}   →   {pred_str}  （置信度 {conf_pct:.1f}%）{correct_str}",
        color=_FG, fontsize=13, fontweight="bold", y=0.97,
    )
    fig.patch.set_facecolor(_BG)

    # ── 渲染为 PIL Image（BytesIO，避免磁盘 I/O）────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_BG)
    buf.seek(0)
    result_img = Image.open(buf)
    result_img.load()   # 在 close(fig) 前强制加载像素数据
    plt.close(fig)

    # ── 保存到文件（可选）────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        result_img.save(save_path, dpi=(150, 150))
        print(f"  → 已保存: {save_path}")
    return result_img


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG

    device = resolve_device(cfg["device"])
    print(f"[Device]  {device}")

    model     = load_model(cfg["model_name"], cfg["checkpoint_path"], device)
    transform = get_eval_transform(cfg["image_size"])

    image_paths = collect_images(cfg["input_path"])
    if not image_paths:
        print(f"[Error] 未找到任何图像: {cfg['input_path']}")
        return
    print(f"[Images]  找到 {len(image_paths)} 张图像\n")

    output_dir = cfg.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    records = []

    for idx, image_path in enumerate(image_paths):
        fname = os.path.basename(image_path)
        print(f"[{idx+1}/{len(image_paths)}]  {fname}")

        # ── 推理 ──────────────────────────────────────────────
        input_tensor, logit, prob = infer_single(
            model, transform, image_path, device)
        pred_label = int(prob >= cfg["threshold"])
        true_label = infer_true_label(image_path)

        status = ""
        if true_label is not None:
            status = "  ✓" if pred_label == true_label else "  ✗"
        print(f"  logit = {logit:+.4f}   prob = {prob:.4f}"
              f"   →  {LABEL_NAME[pred_label]}"
              + (f"   (真实: {LABEL_NAME[true_label]}){status}"
                 if true_label is not None else ""))

        # ── Grad-CAM ──────────────────────────────────────────
        grayscale_cam = None
        rgb_img       = denormalize_image(input_tensor)
        if cfg["use_gradcam"]:
            tc  = cfg["target_class"]
            tgt = (pred_label if tc == "predicted"
                   else (1 if tc == "pneumonia" else 0))
            grayscale_cam, rgb_img = get_gradcam(
                model, cfg["model_name"], input_tensor, tgt)

        # ── 可视化保存 ─────────────────────────────────────────
        stem      = os.path.splitext(fname)[0]
        save_path = (os.path.join(output_dir, f"{stem}_result.png")
                     if output_dir else None)
        visualize_and_save(
            image_path, input_tensor, prob, pred_label, true_label,
            cfg["threshold"], cfg["model_name"],
            grayscale_cam, rgb_img, save_path,
        )

        # ── 记录 ──────────────────────────────────────────────
        records.append({
            "image_path":     image_path,
            "filename":       fname,
            "logit":          round(logit, 6),
            "prob_pneumonia": round(prob, 6),
            "prob_normal":    round(1 - prob, 6),
            "pred_label":     pred_label,
            "pred_class":     LABEL_NAME[pred_label],
            "true_label":     true_label if true_label is not None else "",
            "true_class":     LABEL_NAME[true_label] if true_label is not None
                              else "Unknown",
            "correct":        (int(pred_label == true_label)
                               if true_label is not None else ""),
        })

    # ── 汇总 ──────────────────────────────────────────────────
    df = pd.DataFrame(records)
    if len(records) > 1:
        known = df[df["correct"] != ""]
        if len(known) > 0:
            acc    = known["correct"].astype(int).mean()
            n_corr = known["correct"].astype(int).sum()
            print(f"\n[Summary]  Accuracy = {acc:.4f}"
                  f"  ({n_corr}/{len(known)} 正确)"
                  f"  |  共处理 {len(image_paths)} 张")
        else:
            print(f"\n[Summary]  共处理 {len(image_paths)} 张（真实标签未知）")

    if output_dir and records:
        csv_path = os.path.join(output_dir, "predictions.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[CSV]     汇总结果 → {csv_path}")

    print("\n✓ 完成")


if __name__ == "__main__":
    main()
