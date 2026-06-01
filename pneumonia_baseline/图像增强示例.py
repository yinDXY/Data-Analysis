"""
图像增强示例.py
生成用于 PPT 展示的数据增强可视化图像

输出文件（保存到 augmentation_demo/ 目录）：
  01_augmentation_grid.png     — 单张原图 + 8 种增强变体对比网格
  02_each_method.png           — 每种增强方法单独展示（2 行 × 5 列）
  03_normal_vs_pneumonia.png   — Normal / Pneumonia 各 4 种增强对比
  04_mixup_demo.png            — MixUp 混合过程（5 个 lambda 步骤）
  05_cutout_demo.png           — Cutout 遮挡大小对比
"""

import os
import random
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # 无需显示器
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
DATA_ROOT  = r"D:\数据挖掘课设\dataset\chest_xray\train"
OUT_DIR    = "augmentation_demo"
SEED       = 42
DPI        = 180         # 高分辨率，适合 PPT
IMG_SIZE   = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 配色方案（深色 PPT 友好）
# ──────────────────────────────────────────────
BG      = "#FFFFFF"
SURFACE = "#F5F5F5"
FG      = "#1A1A2E"
ACCENT  = "#1565C0"
GREEN   = "#2E7D32"
RED     = "#C62828"
YELLOW  = "#E65100"
MUTED   = "#757575"
BORDER  = "#BDBDBD"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "text.color":        FG,
    "axes.titlecolor":   FG,
    "axes.labelcolor":   FG,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Microsoft YaHei", "DejaVu Sans"],
})


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _collect(class_name: str, n: int = 8) -> list:
    """随机抽取 n 张指定类别的 PIL Image。"""
    d = os.path.join(DATA_ROOT, class_name)
    files = glob.glob(os.path.join(d, "*.jpeg")) + \
            glob.glob(os.path.join(d, "*.jpg")) + \
            glob.glob(os.path.join(d, "*.png"))
    chosen = random.sample(files, min(n, len(files)))
    return [Image.open(p).convert("RGB") for p in chosen]


def _to_display(img_pil: Image.Image) -> np.ndarray:
    """PIL → 224×224 numpy uint8，用于展示（不归一化）。"""
    return np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))


def _tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """归一化 Tensor → 反归一化 numpy uint8。"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = t.clone().cpu() * std + mean
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


def _ax_img(ax, img, title="", title_color=FG, border_color=None):
    """在 ax 上显示图像，去掉坐标轴，加标题。"""
    if isinstance(img, torch.Tensor):
        img = _tensor_to_display(img)
    elif isinstance(img, Image.Image):
        img = _to_display(img)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, color=title_color, pad=4)
    if border_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)


def _base_transform(pil: Image.Image) -> Image.Image:
    """仅 Resize，保持原比例，不增强。"""
    return pil.resize((IMG_SIZE, IMG_SIZE))


# ──────────────────────────────────────────────
# 各增强方法定义（均作用于 PIL Image，返回 PIL Image）
# ──────────────────────────────────────────────

def aug_random_crop(pil):
    t = T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 0.95))
    return T.ToPILImage()(t(T.ToTensor()(pil.resize((256, 256)))))

def aug_rotation(pil):
    return TF.rotate(pil.resize((IMG_SIZE, IMG_SIZE)),
                     angle=random.uniform(-12, 12))

def aug_hflip(pil):
    return TF.hflip(pil.resize((IMG_SIZE, IMG_SIZE)))

def aug_brightness(pil):
    return TF.adjust_brightness(pil.resize((IMG_SIZE, IMG_SIZE)),
                                brightness_factor=random.uniform(0.6, 1.5))

def aug_contrast(pil):
    return TF.adjust_contrast(pil.resize((IMG_SIZE, IMG_SIZE)),
                              contrast_factor=random.uniform(0.6, 1.6))

def aug_color_jitter(pil):
    t = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
    return t(pil.resize((IMG_SIZE, IMG_SIZE)))

def aug_all(pil):
    """组合所有基础增强（项目中实际使用的 pipeline）。"""
    t = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
        T.RandomRotation(degrees=10),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    return t(pil)

def aug_cutout(pil, size=64):
    """先走完 pipeline 再做 Cutout，返回可视化（反归一化后贴黑块）。"""
    t = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = t(pil)
    _, h, w = tensor.shape
    cx = random.randint(size // 2, w - size // 2)
    cy = random.randint(size // 2, h - size // 2)
    x1, x2 = max(0, cx - size // 2), min(w, cx + size // 2)
    y1, y2 = max(0, cy - size // 2), min(h, cy + size // 2)
    tensor = tensor.clone()
    tensor[:, y1:y2, x1:x2] = 0.0
    return tensor   # 返回 Tensor，由 _ax_img 反归一化


# ──────────────────────────────────────────────
# 图 1：单张原图 + 8 种增强变体
# ──────────────────────────────────────────────

def make_augmentation_grid():
    print("[1/5] 生成增强网格图...")
    pil_orig = _collect("PNEUMONIA", 1)[0]

    methods = [
        ("原始图像",       pil_orig,                       ACCENT),
        ("随机裁剪",       aug_random_crop(pil_orig),      FG),
        ("随机旋转 ±12°",  aug_rotation(pil_orig),         FG),
        ("水平翻转",       aug_hflip(pil_orig),            FG),
        ("亮度调整",       aug_brightness(pil_orig),       FG),
        ("对比度调整",     aug_contrast(pil_orig),         FG),
        ("颜色抖动",       aug_color_jitter(pil_orig),     FG),
        ("组合增强",       aug_all(pil_orig),              YELLOW),
        ("Cutout 遮挡",    aug_cutout(pil_orig, size=64),  RED),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("数据增强策略总览  |  Chest X-Ray", fontsize=14,
                 color=FG, fontweight="bold", y=0.97)

    for ax, (title, img, tc) in zip(axes.flatten(), methods):
        border = ACCENT if tc == ACCENT else (YELLOW if tc == YELLOW else (RED if tc == RED else None))
        _ax_img(ax, img, title=title, title_color=tc, border_color=border)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "01_augmentation_grid.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   → 已保存: {out}")


# ──────────────────────────────────────────────
# 图 2：每种增强方法独立展示
# ──────────────────────────────────────────────

def make_each_method():
    print("[2/5] 生成各方法独立展示图...")
    pil_orig = _collect("NORMAL", 1)[0]
    orig_224 = _base_transform(pil_orig)

    rows_def = [
        [
            ("原始图像",      orig_224,                      ACCENT),
            ("随机裁剪",      aug_random_crop(pil_orig),     FG),
            ("随机旋转",      aug_rotation(pil_orig),        FG),
            ("水平翻转",      aug_hflip(pil_orig),           FG),
            ("亮度 ↑↓",       aug_brightness(pil_orig),      FG),
        ],
        [
            ("原始图像",      orig_224,                      ACCENT),
            ("对比度调整",    aug_contrast(pil_orig),        FG),
            ("颜色抖动",      aug_color_jitter(pil_orig),    FG),
            ("组合增强",      aug_all(pil_orig),             YELLOW),
            ("Cutout 遮挡",   aug_cutout(pil_orig, size=56), RED),
        ],
    ]

    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("各数据增强方法（上：几何变换  |  下：像素/正则化）",
                 fontsize=12, color=FG, fontweight="bold", y=1.01)

    for r, row_def in enumerate(rows_def):
        for c, (title, img, tc) in enumerate(row_def):
            ax = axes[r, c]
            border = ACCENT if tc == ACCENT else (YELLOW if tc == YELLOW else (RED if tc == RED else None))
            _ax_img(ax, img, title=title, title_color=tc, border_color=border)
            if c == 0:
                ax.set_ylabel("Normal 样本" if r == 0 else "Normal 样本",
                              color=MUTED, fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_each_method.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   → 已保存: {out}")


# ──────────────────────────────────────────────
# 图 3：Normal vs Pneumonia 增强对比
# ──────────────────────────────────────────────

def make_normal_vs_pneumonia():
    print("[3/5] 生成 Normal vs Pneumonia 对比图...")
    norm_pil  = _collect("NORMAL",    1)[0]
    pneu_pil  = _collect("PNEUMONIA", 1)[0]

    augs = [
        ("原始图像",   lambda p: _base_transform(p)),
        ("组合增强①",  aug_all),
        ("组合增强②",  aug_all),
        ("Cutout",     lambda p: aug_cutout(p, size=60)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Normal vs Pneumonia  ×  数据增强",
                 fontsize=13, color=FG, fontweight="bold", y=1.01)

    class_labels = [("NORMAL",    norm_pil,  GREEN),
                    ("PNEUMONIA", pneu_pil,  RED)]

    for r, (cls, pil, row_color) in enumerate(class_labels):
        for c, (title, fn) in enumerate(augs):
            ax = axes[r, c]
            img = fn(pil)
            border = ACCENT if c == 0 else None
            _ax_img(ax, img, title=title if r == 0 else "",
                    title_color=FG, border_color=border)
            if c == 0:
                ax.set_ylabel(cls, color=row_color,
                              fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "03_normal_vs_pneumonia.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   → 已保存: {out}")


# ──────────────────────────────────────────────
# 图 4：MixUp 混合过程展示
# ──────────────────────────────────────────────

def make_mixup_demo():
    print("[4/5] 生成 MixUp 展示图...")

    to_tensor = T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    norm_pil = _collect("NORMAL",    1)[0]
    pneu_pil = _collect("PNEUMONIA", 1)[0]
    t_n = to_tensor(norm_pil)
    t_p = to_tensor(pneu_pil)

    lambdas = [1.0, 0.75, 0.50, 0.25, 0.0]
    # λ=1.0 → 纯 Normal；λ=0.0 → 纯 Pneumonia

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    fig.patch.set_facecolor(BG)
    fig.suptitle("MixUp 数据增强  |  Normal(λ) + Pneumonia(1-λ)",
                 fontsize=12, color=FG, fontweight="bold", y=1.04)

    for ax, lam in zip(axes, lambdas):
        mixed = lam * t_n + (1 - lam) * t_p
        label = f"λ = {lam:.2f}"
        if lam == 1.0:
            color = GREEN
        elif lam == 0.0:
            color = RED
        else:
            # 渐变色：绿 → 黄 → 红
            r = int(0xA6 + (0xF3 - 0xA6) * (1 - lam))
            g = int(0xE3 + (0x8B - 0xE3) * (1 - lam))
            b = int(0xA1 + (0xA8 - 0xA1) * (1 - lam))
            color = f"#{r:02X}{g:02X}{b:02X}"
        _ax_img(ax, mixed, title=label, title_color=color,
                border_color=color)

    # 添加箭头标注
    fig.text(0.09, -0.04, "← 纯 Normal",  color=GREEN, fontsize=9, ha="left")
    fig.text(0.91, -0.04, "纯 Pneumonia →", color=RED, fontsize=9, ha="right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "04_mixup_demo.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   → 已保存: {out}")


# ──────────────────────────────────────────────
# 图 5：Cutout 不同遮挡大小对比
# ──────────────────────────────────────────────

def make_cutout_demo():
    print("[5/5] 生成 Cutout 对比图...")

    to_tensor = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    pil = _collect("PNEUMONIA", 1)[0]

    def apply_cutout(pil_img, size):
        tensor = to_tensor(pil_img).clone()
        _, h, w = tensor.shape
        cx = random.randint(size // 2, w - size // 2)
        cy = random.randint(size // 2, h - size // 2)
        x1, x2 = max(0, cx - size // 2), min(w, cx + size // 2)
        y1, y2 = max(0, cy - size // 2), min(h, cy + size // 2)
        tensor[:, y1:y2, x1:x2] = 0.0
        return tensor

    sizes = [0, 32, 48, 64, 96]
    labels = ["无 Cutout", "size = 32", "size = 48", "size = 64\n(训练配置)", "size = 96"]
    colors = [FG,         FG,          FG,           YELLOW,                  MUTED]

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Cutout 遮挡大小对比  |  Pneumonia 样本",
                 fontsize=12, color=FG, fontweight="bold", y=1.04)

    for ax, size, label, color in zip(axes, sizes, labels, colors):
        if size == 0:
            t = T.Compose([
                T.Resize(256), T.CenterCrop(IMG_SIZE),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            img = t(pil)
        else:
            img = apply_cutout(pil, size)
        border = YELLOW if size == 64 else None
        _ax_img(ax, img, title=label, title_color=color, border_color=border)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "05_cutout_demo.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   → 已保存: {out}")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print(" 数据增强可视化  →  输出目录:", OUT_DIR)
    print("=" * 52)

    make_augmentation_grid()
    make_each_method()
    make_normal_vs_pneumonia()
    make_mixup_demo()
    make_cutout_demo()

    print()
    print("=" * 52)
    print(" 全部完成！生成文件：")
    for f in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(p) // 1024
        print(f"   {f}  ({size_kb} KB)")
    print("=" * 52)
