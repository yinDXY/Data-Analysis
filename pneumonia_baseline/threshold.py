"""
threshold.py — Threshold Search 快速启动脚本

直接修改下方 CONFIG 中的参数，然后运行：
    python threshold.py
"""

import sys
import os
import types

# ─────────────────────────────────────────────
# 在此修改所有参数
# ─────────────────────────────────────────────

CONFIG = dict(
    # 数据集根目录（包含 train / val / test 子目录）
    data_dir          = r"D:\数据挖掘课设\dataset\chest_xray",

    # 模型：resnet50 / densenet121 / efficientnet_b0
    model_name        = "efficientnet_b0",

    # best checkpoint 路径（必须与训练时 val_strategy / seed 一致）
    checkpoint_path   = r"results_split\checkpoints\efficientnet_b0_best.pth",

    # 结果输出目录
    output_dir        = r"results_split\threshold_search\efficientnet_b0",

    # 验证集策略（必须与训练时完全一致，保证划分可复现）
    val_strategy      = "split_train",
    val_ratio         = 0.15,

    # 图像尺寸
    image_size        = 224,

    # DataLoader 参数
    batch_size        = 32,
    num_workers       = 0,      # Windows 建议 0

    # 随机种子（必须与训练时一致，保证 split_train 划分可复现）
    seed              = 42,

    # 阈值搜索范围和步长
    threshold_min     = 0.05,
    threshold_max     = 0.95,
    threshold_step    = 0.01,   # 共约 91 个阈值点
)

# ─────────────────────────────────────────────
# 以下无需修改
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import find_threshold
    _orig_parse = find_threshold.parse_args

    def _patched_parse():
        return types.SimpleNamespace(**CONFIG)

    find_threshold.parse_args = _patched_parse
    find_threshold.main()
    find_threshold.parse_args = _orig_parse
