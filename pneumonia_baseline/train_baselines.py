#!/usr/bin/env python
"""
train_baselines.py — 胸部 X 光肺炎二分类 Baseline 主训练脚本

支持模型：resnet50 / densenet121 / efficientnet_b0 / all

用法示例：
    # 单模型
    python train_baselines.py --data_dir ./dataset/chest_xray --model_name resnet50

    # 全部模型
    python train_baselines.py --data_dir ./dataset/chest_xray --model_name all --epochs 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 确保项目根目录在 sys.path 中，支持直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (
    build_dataloaders,
    count_dataset_distribution,
    save_dataset_distribution,
)
from src.engine import evaluate, fit_model, load_checkpoint
from src.metrics import metrics_to_dataframe
from src.models import freeze_backbone, get_model, list_supported_models, unfreeze_all, unfreeze_last_block
from src.plots import (
    plot_accuracy_curve,
    plot_all_models_roc,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curve,
    plot_training_curves,
)
from src.losses import get_loss_function
from src.utils import (
    count_parameters,
    create_output_dirs,
    get_device,
    print_config,
    save_json,
    set_seed,
)


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Pneumonia 二分类 Baseline 训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="chest_xray 根目录（包含 train / val / test 子目录）",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet50",
        help="模型：resnet50 / densenet121 / efficientnet_b0 / all",
    )
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument(
        "--val_strategy", type=str, default="split_train",
        choices=["original", "split_train"],
        help="验证集策略：original=使用原始 val 目录；split_train=从 train 分层划分",
    )
    parser.add_argument("--val_ratio",  type=float, default=0.15,
                        help="split_train 模式下验证集占训练集的比例")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="二值化阈值（sigmoid 输出 >= threshold → PNEUMONIA）")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", type=str,   default="results",
                        help="输出根目录")
    # ── 数据增强 ────────────────────────────────────────────
    parser.add_argument("--no_augment", action="store_true", default=False,
                        help="关闭所有随机增强（训练集与验证集使用相同 pipeline）")
    parser.add_argument("--cutout",     action="store_true", default=False,
                        help="在训练增强中加入 Cutout（需 augment 开启）")
    parser.add_argument("--cutout_size", type=int, default=64,
                        help="Cutout 遮挡方块边长（像素）")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="MixUp 的 Beta 分布参数，0.0 表示不使用 MixUp（建议 0.2~0.4）")
    # ── 分层解冻 ──────────────────────────────────────
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="Stage 1 冒结 backbone 训练的 epoch 数，0=不使用分层解冻")
    parser.add_argument("--last_block_epochs", type=int, default=0,
                        help="Stage 2 解冻最后一个 block 训练的 epoch 数，0=跳过此阶段")
    parser.add_argument("--finetune_lr", type=float, default=-1.0,
                        help="Stage 2 解冻后的学习率，-1=自动取 lr/10")
    # ── 损失函数 ────────────────────────────────────────────
    parser.add_argument("--loss_name", type=str, default="bce",
                        choices=["bce", "soft_mcc", "bce_soft_mcc"],
                        help="损失函数：bce / soft_mcc / bce_soft_mcc")
    parser.add_argument("--bce_weight", type=float, default=1.0,
                        help="bce_soft_mcc 时 BCE 项的权重")
    parser.add_argument("--mcc_weight", type=float, default=1.0,
                        help="soft_mcc / bce_soft_mcc 时 Soft MCC 项的权重")
    # ── A 模块：WTConv 多频特征增强 ──────────────────────────────
    parser.add_argument("--use_wtconv", action="store_true", default=False,
                        help="启用 WTConv A 模块（仅支持 model_name=densenet121）")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 单模型训练 + 测试
# ─────────────────────────────────────────────

def _train_and_evaluate(
    model_name: str,
    args: argparse.Namespace,
    dataloaders: dict,
    device: torch.device,
    summary_rows: list,
    roc_data: dict,
) -> None:
    """训练单个模型，完成测试评估并保存所有产物。"""

    print(f"\n{'#' * 60}")
    print(f"  模型: {model_name}")
    print(f"{'#' * 60}")

    # ── 输出路径 ──────────────────────────────
    checkpoint_path = os.path.join(args.output_dir, "checkpoints",
                                   f"{model_name}_best.pth")
    log_path        = os.path.join(args.output_dir, "logs",
                                   f"{model_name}_training_log.csv")
    fig_curve_path  = os.path.join(args.output_dir, "figures",
                                   f"{model_name}_training_curves.png")
    fig_roc_path    = os.path.join(args.output_dir, "figures",
                                   f"{model_name}_roc_curve.png")
    fig_cm_path     = os.path.join(args.output_dir, "confusion_matrices",
                                   f"{model_name}_confusion_matrix.png")
    pred_path       = os.path.join(args.output_dir, "predictions",
                                   f"{model_name}_test_predictions.csv")

    # ── 构建模型 ──────────────────────────────
    use_wtconv = getattr(args, "use_wtconv", False)
    if use_wtconv and model_name != "densenet121":
        raise ValueError(
            f"--use_wtconv 仅支持 densenet121，当前模型为 '{model_name}'"
        )
    model = get_model(model_name, pretrained=True, use_wtconv=use_wtconv).to(device)
    count_parameters(model)

    # ── 损失 / 优化器 / 调度器 ────────────────
    criterion = get_loss_function(
        loss_name=getattr(args, "loss_name", "bce"),
        bce_weight=getattr(args, "bce_weight", 1.0),
        mcc_weight=getattr(args, "mcc_weight", 1.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # ReduceLROnPlateau 监控 val AUC（越大越好）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # ── 训练（支持三阶段分层解冻） ──────────────────────
    freeze_epochs     = getattr(args, "freeze_epochs",     0)
    last_block_epochs = getattr(args, "last_block_epochs", 0)
    raw_finetune_lr   = getattr(args, "finetune_lr",       -1.0)
    use_progressive   = (freeze_epochs > 0 or last_block_epochs > 0)

    if use_progressive:
        finetune_lr_s2 = raw_finetune_lr if raw_finetune_lr > 0 else args.lr * 0.1
        # 有 Stage 2 则 Stage 3 再缩 1/10；否则与 Stage 2 相同（兴容旧两阶段配置）
        finetune_lr_s3 = (finetune_lr_s2 / 10) if last_block_epochs > 0 else finetune_lr_s2

        phase_logs   = []
        epoch_offset = 0

        # ── Stage 1: 冒结 backbone，仅训练分类头 ────────
        if freeze_epochs > 0:
            print(f"\n[分层解冻] Stage 1 — 冒结 backbone ({freeze_epochs} epoch, lr={args.lr:.2e})")
            freeze_backbone(model, model_name)
            log_s = fit_model(
                model=model,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=freeze_epochs,
                model_name=f"{model_name} [S1:head]",
                checkpoint_path=checkpoint_path,
                log_path=None,
                threshold=args.threshold,
                mixup_alpha=args.mixup_alpha,
                test_loader=dataloaders["test"],
            )
            log_s = log_s.copy(); log_s["epoch"] += epoch_offset
            phase_logs.append(log_s)
            epoch_offset += freeze_epochs

        # ── Stage 2: 解冻最后一个 block + 分类头 ─────
        if last_block_epochs > 0:
            print(f"\n[分层解冻] Stage 2 — 解冻最后 block ({last_block_epochs} epoch, lr={finetune_lr_s2:.2e})")
            unfreeze_last_block(model, model_name)
            opt2 = torch.optim.AdamW(model.parameters(), lr=finetune_lr_s2, weight_decay=args.weight_decay)
            sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="max", factor=0.5, patience=3)
            log_s = fit_model(
                model=model,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                criterion=criterion,
                optimizer=opt2,
                scheduler=sch2,
                device=device,
                epochs=last_block_epochs,
                model_name=f"{model_name} [S2:last_block]",
                checkpoint_path=checkpoint_path,
                log_path=None,
                threshold=args.threshold,
                mixup_alpha=args.mixup_alpha,
                test_loader=dataloaders["test"],
            )
            log_s = log_s.copy(); log_s["epoch"] += epoch_offset
            phase_logs.append(log_s)
            epoch_offset += last_block_epochs

        # ── Stage 3: 全部解冻，全局微调 ───────────
        remaining = args.epochs - freeze_epochs - last_block_epochs
        if remaining > 0:
            print(f"\n[分层解冻] Stage 3 — 解冻所有层 ({remaining} epoch, lr={finetune_lr_s3:.2e})")
            unfreeze_all(model)
            opt3 = torch.optim.AdamW(model.parameters(), lr=finetune_lr_s3, weight_decay=args.weight_decay)
            sch3 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt3, mode="max", factor=0.5, patience=3)
            log_s = fit_model(
                model=model,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                criterion=criterion,
                optimizer=opt3,
                scheduler=sch3,
                device=device,
                epochs=remaining,
                model_name=f"{model_name} [S3:full]",
                checkpoint_path=checkpoint_path,
                log_path=None,
                threshold=args.threshold,
                mixup_alpha=args.mixup_alpha,
                test_loader=dataloaders["test"],
            )
            log_s = log_s.copy(); log_s["epoch"] += epoch_offset
            phase_logs.append(log_s)

        # 合并各阶段日志并保存
        log_df = pd.concat(phase_logs, ignore_index=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"[Log] 合并训练日志已保存: {log_path}")

    else:
        # 标准单阶段训练
        log_df = fit_model(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            threshold=args.threshold,
            mixup_alpha=args.mixup_alpha,
            test_loader=dataloaders["test"],
        )

    # ── 训练曲线 ──────────────────────────────
    plot_training_curves(log_df, save_path=fig_curve_path, model_name=model_name)
    # ── 独立精度图 ──────────────────────────
    fig_acc_path = os.path.join(args.output_dir, "figures",
                                f"{model_name}_accuracy_curve.png")
    plot_accuracy_curve(log_df, save_path=fig_acc_path, model_name=model_name)
    # ── 加载 best checkpoint → 测试集评估 ─────
    load_checkpoint(model, checkpoint_path, device)
    test_loss, test_metrics, y_true, y_prob, image_paths = evaluate(
        model, dataloaders["test"], criterion, device, threshold=args.threshold,
    )

    print(f"\n[Test] {model_name}  test_loss={test_loss:.4f}")
    for k, v in test_metrics.items():
        print(f"  {k:<15}: {v}")

    # ── 测试集 ROC & 混淆矩阵 ─────────────────
    plot_roc_curve(y_true, y_prob,
                   save_path=fig_roc_path, model_name=model_name)
    plot_confusion_matrix(
        tn=test_metrics["tn"], fp=test_metrics["fp"],
        fn=test_metrics["fn"], tp=test_metrics["tp"],
        save_path=fig_cm_path, model_name=model_name,
    )

    # ── 逐样本预测结果 CSV ────────────────────
    _label_name = {0: "NORMAL", 1: "PNEUMONIA"}
    pred_labels = (y_prob >= args.threshold).astype(int)
    pred_df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": y_true,
        "true_class": [_label_name[int(l)] for l in y_true],
        "pred_prob":  np.round(y_prob, 6),
        "pred_label": pred_labels,
        "pred_class": [_label_name[int(l)] for l in pred_labels],
        "correct":    (pred_labels == y_true).astype(int),
    })
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"[Pred] 预测结果已保存: {pred_path}")

    # ── 汇总 ──────────────────────────────────
    summary_rows.append({"model": model_name, **test_metrics})
    roc_data[model_name] = {"y_true": y_true, "y_prob": y_prob}


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print_config(args)

    # 1. 随机种子
    set_seed(args.seed)

    # 2. 设备
    device = get_device()

    # 3. 创建输出目录
    create_output_dirs(args.output_dir)

    # 4. 保存本次实验配置
    save_json(vars(args), os.path.join(args.output_dir, "logs", "config.json"))

    # 5. 数据集分布统计
    dist_df = count_dataset_distribution(args.data_dir)
    print("\n[Dataset] 类别分布：")
    print(dist_df.to_string(index=False))
    save_dataset_distribution(
        dist_df,
        os.path.join(args.output_dir, "logs", "dataset_distribution.csv"),
    )

    # 6. 构建 DataLoaders（所有模型共用同一份数据划分）
    dataloaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_strategy=args.val_strategy,
        val_ratio=args.val_ratio,
        seed=args.seed,
        augment=not args.no_augment,
        cutout=args.cutout,
        cutout_size=args.cutout_size,
    )

    # 7. 确定待训练模型列表
    supported = list_supported_models()
    if args.model_name == "all":
        model_names = supported
    elif args.model_name in supported:
        model_names = [args.model_name]
    else:
        raise ValueError(
            f"不支持的 model_name: '{args.model_name}'，"
            f"可选: {supported + ['all']}"
        )

    # 8. 逐模型训练与评估
    summary_rows: list = []
    roc_data:     dict = {}

    for model_name in model_names:
        _train_and_evaluate(
            model_name, args, dataloaders, device, summary_rows, roc_data,
        )

    # 9. 保存汇总指标 CSV
    summary_path = os.path.join(args.output_dir, "logs", "baseline_summary.csv")
    summary_df   = metrics_to_dataframe(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[Summary] 汇总指标已保存: {summary_path}")
    print(summary_df.to_string(index=False))

    # 10. 多模型对比图（仅 all 模式）
    if args.model_name == "all":
        plot_all_models_roc(
            roc_data,
            save_path=os.path.join(
                args.output_dir, "figures", "all_models_roc_comparison.png"
            ),
        )
        plot_metrics_comparison(
            summary_df,
            save_path=os.path.join(
                args.output_dir, "figures", "baseline_metrics_comparison.png"
            ),
        )

    print("\n✓ 所有任务完成！")


if __name__ == "__main__":
    main()
