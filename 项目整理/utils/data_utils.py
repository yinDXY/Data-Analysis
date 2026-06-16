"""
utils.py — 通用工具函数
包含：随机种子固定、设备获取、目录管理、配置保存、参数统计等。
"""

import os
import json
import random
import numpy as np
import torch


# ─────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多 GPU 场景
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# 设备
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    """优先返回 CUDA 设备，否则返回 CPU。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 使用设备: {device}")
    if device.type == "cuda":
        print(f"         GPU 名称 : {torch.cuda.get_device_name(0)}")
    return device


# ─────────────────────────────────────────────
# 目录管理
# ─────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """若目录不存在则递归创建。"""
    os.makedirs(path, exist_ok=True)


def create_output_dirs(output_dir: str) -> None:
    """在 output_dir 下创建标准子目录结构。

    子目录：
        checkpoints/   — 模型权重
        logs/          — 训练日志
        figures/       — 学习曲线等图表
        confusion_matrices/ — 混淆矩阵
        predictions/   — 推理输出
    """
    sub_dirs = [
        "checkpoints",
        "logs",
        "figures",
        "confusion_matrices",
        "predictions",
    ]
    for sub in sub_dirs:
        ensure_dir(os.path.join(output_dir, sub))
    print(f"[Output] 输出目录已就绪: {output_dir}")


# ─────────────────────────────────────────────
# 配置持久化
# ─────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    """将字典序列化为 JSON 文件并保存。

    Args:
        data: 待保存的字典（需可 JSON 序列化）。
        path: 目标文件路径（含文件名）。
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[JSON ] 已保存: {path}")


# ─────────────────────────────────────────────
# 模型信息
# ─────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数总量并打印。

    Returns:
        可训练参数数量（int）。
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数量: {total:,}")
    return total


# ─────────────────────────────────────────────
# 配置打印
# ─────────────────────────────────────────────

def print_config(args) -> None:
    """格式化打印命令行 / argparse 配置项。

    Args:
        args: argparse.Namespace 或任意含 __dict__ 属性的对象。
    """
    print("=" * 50)
    print("  实验配置")
    print("=" * 50)
    config_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    for key, value in config_dict.items():
        print(f"  {key:<25}: {value}")
    print("=" * 50)
