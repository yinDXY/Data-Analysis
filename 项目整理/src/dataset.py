"""
dataset.py — 数据集加载与预处理

支持两种验证集策略：
  - "original"     : 使用原始 train / val / test 目录
  - "split_train"  : 从 train 中按类别分层划分验证集，忽略原始 val 目录
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

LABEL_MAP: Dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 1}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# 1. 自定义 Transform：Cutout
# ─────────────────────────────────────────────

class CutoutTransform:
    """随机遮挡图像中一个正方形区域（Cutout 正则化）。

    在 ToTensor + Normalize 之后作用于 Tensor，将遮挡区域置为 0。

    Args:
        size: 遮挡正方形的边长（像素）。
    """

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        half = self.size // 2
        cx = torch.randint(0, w, (1,)).item()
        cy = torch.randint(0, h, (1,)).item()
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self) -> str:
        return f"CutoutTransform(size={self.size})"


# ─────────────────────────────────────────────
# 2. MixUp 工具函数
# ─────────────────────────────────────────────

def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """对一个 batch 执行 MixUp。

    混合后的 loss = lam * BCE(pred, labels_a) + (1-lam) * BCE(pred, labels_b)

    Args:
        images : [B, C, H, W] 图像 Tensor。
        labels : [B] 标签 Tensor（float）。
        alpha  : Beta 分布参数，典型值 0.2~0.4。

    Returns:
        mixed_images, labels_a, labels_b, lam
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────
# 3. Transforms
# ─────────────────────────────────────────────

def get_transforms(
    image_size: int = 224,
    augment: bool = True,
    cutout: bool = False,
    cutout_size: int = 64,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """返回 (train_transform, eval_transform)。

    Args:
        image_size  : 最终裁剪尺寸，默认 224。
        augment     : 是否对训练集使用随机增强，False 则训练集与验证集用同一 pipeline。
        cutout      : 是否在训练集上叠加 Cutout（需 augment=True 才生效）。
        cutout_size : Cutout 遮挡边长（像素）。

    Returns:
        train_transform, eval_transform
    """
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if not augment:
        return eval_transform, eval_transform

    train_ops = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if cutout:
        train_ops.append(CutoutTransform(size=cutout_size))

    train_transform = transforms.Compose(train_ops)
    return train_transform, eval_transform


# ─────────────────────────────────────────────
# 2. 路径收集
# ─────────────────────────────────────────────

def collect_image_paths(root_dir: str) -> Tuple[List[str], List[int]]:
    """读取某个 split 目录下所有图片的路径与标签。

    目录结构要求：
        root_dir/
        ├── NORMAL/
        └── PNEUMONIA/

    Args:
        root_dir: split 目录路径（如 .../chest_xray/train）。

    Returns:
        paths  : 图片绝对路径列表。
        labels : 对应标签列表（NORMAL=0, PNEUMONIA=1）。

    Raises:
        FileNotFoundError: root_dir 不存在。
        ValueError: root_dir 中找不到任何类别子目录。
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    paths: List[str] = []
    labels: List[int] = []

    found_any = False
    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # 允许某个 split 缺失某个类（容错）
        found_any = True
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    if not found_any:
        raise ValueError(
            f"在 {root_dir} 中未找到 NORMAL / PNEUMONIA 子目录，"
            "请检查 data_dir 是否指向正确的 split 目录。"
        )

    return paths, labels


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """胸部 X 光二分类 Dataset。

    每个样本返回 (image_tensor, label, image_path)。
    图片以 RGB 模式读取，灰度图自动转换。

    Args:
        paths    : 图片路径列表。
        labels   : 对应标签列表（int）。
        transform: torchvision transforms，为 None 时只转 Tensor。
    """

    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        transform: transforms.Compose = None,
    ) -> None:
        assert len(paths) == len(labels), "paths 与 labels 长度不一致"
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法读取图片 {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_path


# ─────────────────────────────────────────────
# 4. DataLoaders
# ─────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_strategy: str = "split_train",
    val_ratio: float = 0.15,
    seed: int = 42,
    augment: bool = True,
    cutout: bool = False,
    cutout_size: int = 64,
) -> Dict[str, DataLoader]:
    """构建 train / val / test DataLoader。

    Args:
        data_dir     : chest_xray 根目录，包含 train / val / test 子目录。
        batch_size   : 批大小。
        num_workers  : DataLoader 工作进程数。
        image_size   : 图片裁剪尺寸。
        val_strategy : 验证集策略，"original" 或 "split_train"。
        val_ratio    : split_train 模式下验证集比例（相对于 train）。
        seed         : 随机种子，用于分层划分。
        augment      : 是否对训练集启用随机增强。
        cutout       : 是否在训练增强中加入 Cutout（需 augment=True）。
        cutout_size  : Cutout 遮挡边长（像素）。

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}

    Raises:
        FileNotFoundError: data_dir 不存在。
        ValueError: val_strategy 不合法，或 val_ratio 超出范围。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据根目录不存在: {data_dir}")
    if val_strategy not in ("original", "split_train"):
        raise ValueError(f"val_strategy 必须为 'original' 或 'split_train'，收到: {val_strategy}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio 必须在 (0, 1) 范围内，收到: {val_ratio}")

    train_transform, eval_transform = get_transforms(
        image_size=image_size,
        augment=augment,
        cutout=cutout,
        cutout_size=cutout_size,
    )
    aug_info = "on" if augment else "off"
    if augment and cutout:
        aug_info += f" + Cutout(size={cutout_size})"
    print(f"[Dataset] 数据增强     : {aug_info}")

    # ── 测试集（两种策略共用）────────────────────
    test_paths, test_labels = collect_image_paths(os.path.join(data_dir, "test"))

    # ── 训练集 / 验证集 ──────────────────────────
    if val_strategy == "original":
        train_paths, train_labels = collect_image_paths(os.path.join(data_dir, "train"))
        val_paths,   val_labels   = collect_image_paths(os.path.join(data_dir, "val"))

    else:  # split_train
        all_paths, all_labels = collect_image_paths(os.path.join(data_dir, "train"))
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths,
            all_labels,
            test_size=val_ratio,
            stratify=all_labels,
            random_state=seed,
        )

    # ── 构建 Dataset ─────────────────────────────
    train_dataset = ChestXrayDataset(train_paths, train_labels, transform=train_transform)
    val_dataset   = ChestXrayDataset(val_paths,   val_labels,   transform=eval_transform)
    test_dataset  = ChestXrayDataset(test_paths,  test_labels,  transform=eval_transform)

    # ── 构建 DataLoader ──────────────────────────
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # ── 统计信息打印 ─────────────────────────────
    print(f"[Dataset] 策略       : {val_strategy}")
    print(f"[Dataset] train 样本 : {len(train_dataset)}")
    print(f"[Dataset] val   样本 : {len(val_dataset)}")
    print(f"[Dataset] test  样本 : {len(test_dataset)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ─────────────────────────────────────────────
# 5. 数据分布统计
# ─────────────────────────────────────────────

def count_dataset_distribution(data_dir: str) -> pd.DataFrame:
    """统计 train / val / test 中各类别的样本数量。

    Args:
        data_dir: chest_xray 根目录。

    Returns:
        DataFrame，列：split / class / count，以及 total 汇总行。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据根目录不存在: {data_dir}")

    records = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in LABEL_MAP:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                count = 0
            else:
                count = sum(
                    1 for f in os.listdir(class_dir)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
                )
            records.append({"split": split, "class": class_name, "count": count})

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────
# 6. 保存分布 CSV
# ─────────────────────────────────────────────

def save_dataset_distribution(df: pd.DataFrame, output_path: str) -> None:
    """将类别统计 DataFrame 保存为 CSV 文件。

    Args:
        df          : count_dataset_distribution 返回的 DataFrame。
        output_path : 目标 CSV 路径（含文件名）。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[Dataset] 分布统计已保存: {output_path}")
