"""
find_threshold.py — 验证集 Threshold Search 脚本

在验证集上遍历不同阈值，找到各评估标准下的最优阈值，
并生成指标曲线图和 ROC 图。

用法：
    python find_threshold.py \\
        --data_dir ./dataset/chest_xray \\
        --model_name densenet121 \\
        --checkpoint_path results_split/checkpoints/densenet121_best.pth \\
        --output_dir results_split/threshold_search/densenet121 \\
        --val_strategy split_train \\
        --val_ratio 0.15 \\
        --seed 42
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import build_dataloaders
from src.metrics import compute_binary_metrics, sigmoid_np
from src.gradcam_utils import load_model_from_checkpoint   # 复用 checkpoint 加载


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证集 Threshold Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",         type=str, required=True)
    parser.add_argument("--model_name",       type=str, required=True,
                        choices=["resnet50", "densenet121", "efficientnet_b0"])
    parser.add_argument("--checkpoint_path",  type=str, required=True)
    parser.add_argument("--output_dir",       type=str, default="results/threshold_search")
    parser.add_argument("--val_strategy",     type=str, default="split_train",
                        choices=["original", "split_train"])
    parser.add_argument("--val_ratio",        type=float, default=0.15)
    parser.add_argument("--image_size",       type=int,   default=224)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--threshold_min",    type=float, default=0.05,
                        help="阈值搜索起始值")
    parser.add_argument("--threshold_max",    type=float, default=0.95,
                        help="阈值搜索终止值")
    parser.add_argument("--threshold_step",   type=float, default=0.01,
                        help="阈值步长")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 步骤 1：在验证集上推断，获取 y_true / y_prob
# ─────────────────────────────────────────────

def run_val_inference(
    model: torch.nn.Module,
    data_dir: str,
    val_strategy: str,
    val_ratio: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
):
    """加载验证集 DataLoader 并执行推断，返回真实标签和预测概率。

    使用 augment=False 保证验证集无随机增强。
    种子与训练时保持一致，确保 split_train 模式下划分可复现。

    Returns:
        y_true : np.ndarray [N]，真实标签（0/1）
        y_prob : np.ndarray [N]，预测为 PNEUMONIA 的概率
    """
    dataloaders = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_strategy=val_strategy,
        val_ratio=val_ratio,
        seed=seed,
        augment=False,
    )
    val_loader = dataloaders["val"]
    print(f"[Threshold] 验证集样本数: {len(val_loader.dataset)}")

    all_true:  list = []
    all_probs: list = []

    model.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="  [Inference]", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1).cpu().numpy()
            probs  = sigmoid_np(logits)
            all_true.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    return np.array(all_true, dtype=np.int32), np.array(all_probs, dtype=np.float32)


# ─────────────────────────────────────────────
# 步骤 2：遍历阈值，计算各指标
# ─────────────────────────────────────────────

def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> pd.DataFrame:
    """在 [threshold_min, threshold_max] 范围内按步长遍历阈值，
    对每个阈值计算完整评估指标。

    Returns:
        DataFrame，每行对应一个阈值，列为 threshold + 各指标。
    """
    thresholds = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)
    thresholds = np.round(thresholds, 6)

    records = []
    for thr in tqdm(thresholds, desc="  [Sweep]", leave=False):
        m = compute_binary_metrics(y_true, y_prob, threshold=float(thr))
        records.append({"threshold": float(thr), **m})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 步骤 3：确定各标准下的最优阈值
# ─────────────────────────────────────────────

def find_optimal_thresholds(df: pd.DataFrame) -> dict:
    """从扫描结果中找出各评估标准下的最优阈值。

    标准说明：
      - max_accuracy      : 验证集准确率最高
      - max_f1            : F1-score 最高（综合 Precision 和 Sensitivity）
      - max_youden        : Youden's J = Sensitivity + Specificity - 1 最高
                            （平衡灵敏度和特异度，常用于筛查场景）
      - sens90_max_spec   : Sensitivity ≥ 0.90 约束下 Specificity 最高
                            （医学筛查常用，尽量不漏诊同时减少误报）
      - sens95_max_spec   : Sensitivity ≥ 0.95 约束下 Specificity 最高
    """
    df = df.copy()
    df["youden"] = df["sensitivity"] + df["specificity"] - 1.0

    optimal = {}

    optimal["max_accuracy"] = {
        "threshold": float(df.loc[df["accuracy"].idxmax(), "threshold"]),
        "value":     float(df["accuracy"].max()),
        "desc":      "验证集准确率最高",
    }
    optimal["max_f1"] = {
        "threshold": float(df.loc[df["f1"].idxmax(), "threshold"]),
        "value":     float(df["f1"].max()),
        "desc":      "F1-score 最高",
    }
    optimal["max_youden"] = {
        "threshold": float(df.loc[df["youden"].idxmax(), "threshold"]),
        "value":     float(df["youden"].max()),
        "desc":      "Youden's J 最高（灵敏度 + 特异度 - 1）",
    }

    for sens_floor, key in [(0.90, "sens90_max_spec"), (0.95, "sens95_max_spec")]:
        subset = df[df["sensitivity"] >= sens_floor]
        if len(subset) > 0:
            best = subset.loc[subset["specificity"].idxmax()]
            optimal[key] = {
                "threshold": float(best["threshold"]),
                "value":     float(best["specificity"]),
                "desc":      f"Sensitivity ≥ {sens_floor:.0%} 时 Specificity 最高",
                "sensitivity": float(best["sensitivity"]),
            }
        else:
            optimal[key] = {
                "threshold": None,
                "value":     None,
                "desc":      f"无满足 Sensitivity ≥ {sens_floor:.0%} 的阈值",
            }

    return optimal


# ─────────────────────────────────────────────
# 步骤 4：可视化
# ─────────────────────────────────────────────

def plot_threshold_curves(
    df: pd.DataFrame,
    optimal: dict,
    save_path: str,
    model_name: str = "",
) -> None:
    """绘制指标 vs 阈值曲线，并标注各最优阈值点。"""
    thresholds = df["threshold"].values
    title_prefix = f"{model_name} — " if model_name else ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：各指标随阈值变化 ────────────────
    ax = axes[0]
    ax.plot(thresholds, df["accuracy"].values,    label="Accuracy",    color="steelblue")
    ax.plot(thresholds, df["sensitivity"].values, label="Sensitivity", color="tomato")
    ax.plot(thresholds, df["specificity"].values, label="Specificity", color="seagreen")
    ax.plot(thresholds, df["f1"].values,          label="F1",          color="darkorchid")
    ax.plot(thresholds, df["precision"].values,   label="Precision",   color="goldenrod", linestyle="--")

    # 标注最优点
    _colors = {
        "max_accuracy":    "steelblue",
        "max_f1":          "darkorchid",
        "max_youden":      "black",
        "sens90_max_spec": "tomato",
        "sens95_max_spec": "firebrick",
    }
    for key, color in _colors.items():
        thr = optimal[key].get("threshold")
        if thr is not None:
            ax.axvline(thr, color=color, linestyle=":", alpha=0.7,
                       label=f"{key} @ {thr:.2f}")

    ax.set_title(f"{title_prefix}Metrics vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── 右图：Youden's J ──────────────────────
    ax2 = axes[1]
    youden = df["sensitivity"].values + df["specificity"].values - 1.0
    ax2.plot(thresholds, youden, color="teal", label="Youden's J")
    thr_j = optimal["max_youden"].get("threshold")
    if thr_j is not None:
        ax2.axvline(thr_j, color="black", linestyle=":", alpha=0.8,
                    label=f"best @ {thr_j:.2f}")
    ax2.set_title(f"{title_prefix}Youden's J = Sensitivity + Specificity − 1")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Youden's J")
    ax2.set_xlim(thresholds.min(), thresholds.max())
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 指标曲线已保存: {save_path}")


def plot_roc_with_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal: dict,
    save_path: str,
    model_name: str = "",
) -> None:
    """绘制 ROC 曲线，并在曲线上标注各最优阈值对应的工作点。"""
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    title_prefix = f"{model_name} — " if model_name else ""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    _markers = {
        "max_accuracy":    ("s", "steelblue",  "max Acc"),
        "max_f1":          ("D", "darkorchid", "max F1"),
        "max_youden":      ("*", "black",      "max Youden"),
        "sens90_max_spec": ("^", "tomato",     "Sens≥0.90"),
        "sens95_max_spec": ("v", "firebrick",  "Sens≥0.95"),
    }
    for key, (marker, color, label) in _markers.items():
        thr = optimal[key].get("threshold")
        if thr is None:
            continue
        # 在 ROC 曲线上找最近的 (FPR, TPR) 工作点
        idx = np.argmin(np.abs(roc_thr - thr))
        ax.scatter(fpr[idx], tpr[idx], marker=marker, color=color, s=100, zorder=5,
                   label=f"{label} (thr={thr:.2f})")

    ax.set_title(f"{title_prefix}ROC Curve with Optimal Thresholds")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] ROC 图已保存: {save_path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Threshold] 设备: {device}")
    print(f"[Threshold] 模型: {args.model_name}  |  Checkpoint: {args.checkpoint_path}")
    print(f"[Threshold] 验证集策略: {args.val_strategy}"
          + (f"  val_ratio={args.val_ratio}" if args.val_strategy == "split_train" else ""))
    print(f"[Threshold] 阈值范围: [{args.threshold_min}, {args.threshold_max}]  "
          f"步长={args.threshold_step}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    model = load_model_from_checkpoint(args.model_name, args.checkpoint_path, device)

    # 2. 验证集推断
    print("\n[Threshold] 正在推断验证集 ...")
    y_true, y_prob = run_val_inference(
        model=model,
        data_dir=args.data_dir,
        val_strategy=args.val_strategy,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    print(f"[Threshold] y_true 分布  NORMAL={int((y_true==0).sum())}  "
          f"PNEUMONIA={int((y_true==1).sum())}")

    # 3. 阈值扫描
    print("\n[Threshold] 正在扫描阈值 ...")
    metrics_df = sweep_thresholds(
        y_true, y_prob,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )

    # 4. 保存全量指标 CSV
    csv_path = os.path.join(args.output_dir, "val_threshold_metrics.csv")
    metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Threshold] 全量指标已保存: {csv_path}")

    # 5. 找最优阈值
    optimal = find_optimal_thresholds(metrics_df)

    # 6. 打印最优阈值汇总
    print(f"\n{'─'*55}")
    print("  各评估标准下的最优阈值")
    print(f"{'─'*55}")
    for key, info in optimal.items():
        thr = info.get("threshold")
        val = info.get("value")
        desc = info.get("desc", "")
        if thr is not None:
            print(f"  {key:<22}: threshold={thr:.3f}  value={val:.4f}  ({desc})")
        else:
            print(f"  {key:<22}: 无满足条件的阈值  ({desc})")
    print(f"{'─'*55}\n")

    # 7. 保存最优阈值 JSON
    json_path = os.path.join(args.output_dir, "val_optimal_thresholds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(optimal, f, ensure_ascii=False, indent=2)
    print(f"[Threshold] 最优阈值已保存: {json_path}")

    # 8. 绘图
    plot_threshold_curves(
        metrics_df, optimal,
        save_path=os.path.join(args.output_dir, "val_threshold_curves.png"),
        model_name=args.model_name,
    )
    plot_roc_with_thresholds(
        y_true, y_prob, optimal,
        save_path=os.path.join(args.output_dir, "val_roc_with_thresholds.png"),
        model_name=args.model_name,
    )

    print(f"\n[Threshold] 完成。输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
