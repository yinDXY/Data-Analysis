"""
gradcam.py — Grad-CAM 可解释性分析快速启动脚本

直接修改下方 CONFIG 中的参数，然后运行：
    python gradcam.py
"""

import sys
import os
import types

# ─────────────────────────────────────────────
# 在此修改所有 Grad-CAM 参数
# ─────────────────────────────────────────────

CONFIG = dict(
    # 数据集根目录（包含 test/ 子目录）
    data_dir          = r"D:\数据挖掘课设\dataset\chest_xray",

    # 模型：resnet50 / densenet121 / efficientnet_b0
    model_name        = "densenet121",

    # best checkpoint 路径
    checkpoint_path   = r"results_split\checkpoints\densenet121_best.pth",

    # 热力图输出根目录
    output_dir        = r"results_split\gradcam\densenet121",

    # 生成热力图的样本总数
    num_samples       = 16,

    # 二值化阈值（必须与训练评估时一致）
    threshold         = 0.5,

    # 输入图像尺寸
    image_size        = 224,

    # 遍历测试集时的批大小
    batch_size        = 32,

    # DataLoader 工作进程数（Windows 建议 0）
    num_workers       = 0,

    # 随机种子
    seed              = 42,

    # Grad-CAM 目标类别：
    #   predicted  — 对模型预测的类别生成热力图（推荐）
    #   pneumonia  — 固定对 PNEUMONIA 方向生成
    #   normal     — 固定对 NORMAL 方向生成
    target_class      = "predicted",

    # 样本选择策略：
    #   mixed      — TP / TN / FP / FN 各取若干（置信度优先，推荐）
    #   tp         — 仅真阳性
    #   tn         — 仅真阴性
    #   fp         — 仅假阳性（误报）
    #   fn         — 仅假阴性（漏诊）
    #   all        — 测试集前 num_samples 张
    sample_mode       = "mixed",
)

# ─────────────────────────────────────────────
# 以下无需修改
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 将项目根目录加入 sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import generate_gradcam
    _orig_parse = generate_gradcam.parse_args

    def _patched_parse():
        return types.SimpleNamespace(**CONFIG)

    generate_gradcam.parse_args = _patched_parse
    generate_gradcam.main()
    generate_gradcam.parse_args = _orig_parse
