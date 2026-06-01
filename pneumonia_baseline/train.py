"""
train.py — 快速启动脚本

直接修改下方 CONFIG 中的参数，然后运行：
    python train.py
"""

import sys
import os
import types

# ─────────────────────────────────────────────
# 在此修改所有训练参数
# ─────────────────────────────────────────────

CONFIG = dict(
    # 数据集根目录（包含 train / val / test 子目录）
    data_dir      = r"D:\数据挖掘课设\dataset\chest_xray",

    # 模型：resnet50 / densenet121 / efficientnet_b0 / all
    model_name    = "densenet121",

    # 训练轮数
    epochs        = 100,

    # 批大小
    batch_size    = 64,

    # 学习率（AdamW）
    lr            = 1e-4,

    # 权重衰减
    weight_decay  = 1e-4,

    # DataLoader 工作进程数（Windows 建议 0 或 2）
    num_workers   = 0,

    # 验证集策略：split_train（推荐） / original
    val_strategy  = "split_train",

    # split_train 模式下验证集占训练集比例
    val_ratio     = 0.15,

    # sigmoid 二值化阈值
    threshold     = 0.5,

    # 随机种子
    seed          = 42,

    # 输出根目录
    output_dir    = "results_modules/A+B+C/densenet_enhanced",

    # ── 数据增强 ───────────────────────────────
    # 是否开启基础随机增强（随机裁剪 / 旋转 / 翻转 / 颜色抖动）
    augment       = True,

    # 是否加入 Cutout 正则化（需 augment=True）
    cutout        = True,

    # Cutout 遮挡方块边长（像素）
    cutout_size   = 64,

    # MixUp alpha 参数（0.0 = 不使用 MixUp，推荐 0.2~0.4）
    mixup_alpha   = 0.2,

    # ── 分层解冻（三阶段；全为 0 则普通训练） ─────────────
    #
    # 示例 A — 普通训练（不分层）：
    #   freeze_epochs=0, last_block_epochs=0
    #
    # 示例 B — DenseNet-121 推荐配置（epochs=15, lr=1e-3）：
    #   freeze_epochs=3      → Stage 1: 只训练 classifier，lr=args.lr
    #   last_block_epochs=7  → Stage 2: 解冻 denseblock4+norm5+cls，lr=finetune_lr
    #   剩余 5 epoch         → Stage 3: 全部解冻，lr=finetune_lr/10
    #   finetune_lr=1e-4     → Stage 2 用 1e-4，Stage 3 自动用 1e-5
    #
    # Stage 1 建议 lr=1e-3（快速拟合分类头），Stage 2/3 由 finetune_lr 控制。
    freeze_epochs     = 0,
    last_block_epochs = 0,
    finetune_lr       = -1,   # -1 = 自动取 args.lr / 10

    # ── 损失函数（消融实验切换指南） ──────────────────────
    #
    # ① Baseline（BCE，原始行为，完全等价于以前的训练）：
    #   loss_name = "bce"
    #   bce_weight / mcc_weight 两行保留原值即可，不生效
    #
    # ② 仅 Soft MCC（C 模块，替换 BCE）：
    #   loss_name = "soft_mcc"
    #   mcc_weight = 1.0
    #
    # ③ BCE + Soft MCC 联合（C 模块主配置）：
    #   loss_name  = "bce_soft_mcc"
    #   bce_weight = 1.0   ← 调整各项相对权重做消融
    #   mcc_weight = 1.0
    #
    # 消融对比建议：
    #   - 保持其他所有参数（lr/epochs/seed/model_name）完全一致
    #   - 只改 loss_name，output_dir 用不同名称区分结果
    #     例如 output_dir = "results/ablation_bce"
    #          output_dir = "results/ablation_bce_mcc"
    #
    # 切回 BCE baseline 只需把下面一行改为 loss_name = "bce"
    loss_name   = "soft_mcc",
    bce_weight  = 1.0,
    mcc_weight  = 1.0,

    # ── A 模块：WTConv 多频特征增强 ───────────────────────────
    #
    # 启用 WTConv A 模块（仅支持 model_name="densenet121"）：
    #   use_wtconv = True
    #
    # 消融建议：
    #   Baseline:    use_wtconv=False, output_dir="results/densenet_baseline"
    #   + WTConv A:  use_wtconv=True,  output_dir="results/densenet_wtconv"
    #   其他参数（lr/epochs/loss_name/seed）保持完全一致
    use_wtconv  = True,

    # ── B 模块：EMA 多尺度空间注意力 ─────────────────────────
    #
    # 启用 EMA B 模块（仅支持 model_name="densenet121"）：
    #   use_ema = True
    #
    # 消融建议：
    #   Baseline:        use_ema=False, output_dir="results/densenet_baseline"
    #   + EMA B:         use_ema=True,  output_dir="results/densenet_ema"
    #   + WTConv A+B:    use_wtconv=True, use_ema=True, output_dir="results/densenet_wtconv_ema"
    use_ema     = True,
)

# ─────────────────────────────────────────────
# 以下无需修改
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 将项目根目录加入 sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from train_baselines import main

    # 将 CONFIG dict 转成 argparse.Namespace 注入
    import train_baselines
    _orig_parse = train_baselines.parse_args

    def _patched_parse():
        # 将 augment 转成 train_baselines 期望的 no_augment 属性
        ns = dict(CONFIG)
        ns["no_augment"] = not ns.pop("augment")

        # 若输出目录已存在，自动追加序号 (2), (3), ... 防止覆盖
        base_dir = ns["output_dir"]
        candidate = base_dir
        idx = 2
        while os.path.exists(candidate):
            candidate = f"{base_dir} ({idx})"
            idx += 1
        if candidate != base_dir:
            print(f"[train.py] 输出目录已存在，改用: {candidate}")
        ns["output_dir"] = candidate

        return types.SimpleNamespace(**ns)

    train_baselines.parse_args = _patched_parse
    main()
    train_baselines.parse_args = _orig_parse
