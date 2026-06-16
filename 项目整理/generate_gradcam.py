"""
generate_gradcam.py — Grad-CAM 可解释性分析脚本

用法示例：
    python generate_gradcam.py \\
        --data_dir ./dataset/chest_xray \\
        --model_name resnet50 \\
        --checkpoint_path results/checkpoints/resnet50_best.pth \\
        --output_dir results/gradcam/resnet50 \\
        --num_samples 16 \\
        --threshold 0.5 \\
        --target_class predicted \\
        --sample_mode mixed

依赖：
    pip install grad-cam opencv-python
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将项目根目录加入 sys.path，支持从任意位置运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (
    ChestXrayDataset,
    collect_image_paths,
    get_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.metrics import sigmoid_np
from src.gradcam_utils import (
    BinaryClassifierOutputTarget,
    GradCAMModelWrapper,
    denormalize_image,
    generate_cam_for_sample,
    get_target_layers,
    load_model_from_checkpoint,
    save_gradcam_images,
    select_samples,
    LABEL_NAME,
)

from pytorch_grad_cam import GradCAM


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM 可解释性分析 — Chest X-Ray Pneumonia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="chest_xray 根目录（包含 test/ 子目录）",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        choices=["resnet50", "densenet121", "efficientnet_b0"],
        help="模型架构",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="best checkpoint .pth 文件路径",
    )
    parser.add_argument("--output_dir",   type=str,   default="results/gradcam",
                        help="Grad-CAM 图像输出根目录")
    parser.add_argument("--num_samples",  type=int,   default=16,
                        help="生成 Grad-CAM 的样本数量")
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="二值化阈值，需与训练评估时一致")
    parser.add_argument("--image_size",   type=int,   default=224,
                        help="输入图像尺寸")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="遍历测试集时的批大小")
    parser.add_argument("--num_workers",  type=int,   default=4,
                        help="DataLoader 工作进程数")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="随机种子")
    parser.add_argument(
        "--target_class", type=str, default="predicted",
        choices=["predicted", "pneumonia", "normal"],
        help=(
            "Grad-CAM 目标类别: "
            "predicted=按预测结果; "
            "pneumonia=固定 PNEUMONIA; "
            "normal=固定 NORMAL"
        ),
    )
    parser.add_argument(
        "--sample_mode", type=str, default="mixed",
        choices=["mixed", "tp", "tn", "fp", "fn", "all"],
        help=(
            "样本选择策略: "
            "mixed=TP/TN/FP/FN 各取若干; "
            "tp/tn/fp/fn=仅选指定类型; "
            "all=测试集前 num_samples 张"
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# 步骤 1：遍历测试集，收集所有预测结果
# ─────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    threshold: float,
    device: torch.device,
) -> pd.DataFrame:
    """在测试集上执行前向推断，返回每样本预测结果 DataFrame。

    Returns:
        DataFrame with columns:
            image_path, true_label, true_class,
            pred_prob, pred_label, pred_class, correctness
    """
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    image_paths, labels = collect_image_paths(test_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"测试集中没有找到图像文件: {test_dir}")

    _, eval_transform = get_transforms(image_size=image_size, augment=False)
    dataset = ChestXrayDataset(image_paths, labels, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_paths: list  = []
    all_true:  list  = []
    all_probs: list  = []

    model.eval()
    with torch.no_grad():
        for images, lbls, paths in tqdm(loader, desc="[Inference] 测试集推断", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1).cpu().numpy()   # [B]
            probs  = sigmoid_np(logits)
            all_paths.extend(paths)
            all_true.extend(lbls.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_probs = np.array(all_probs, dtype=np.float32)
    all_true  = np.array(all_true,  dtype=np.int32)
    pred_labels = (all_probs >= threshold).astype(np.int32)

    # 计算 TP / TN / FP / FN
    correctness = []
    for t, p in zip(all_true, pred_labels):
        if t == 1 and p == 1:
            correctness.append("TP")
        elif t == 0 and p == 0:
            correctness.append("TN")
        elif t == 0 and p == 1:
            correctness.append("FP")
        else:
            correctness.append("FN")

    df = pd.DataFrame({
        "image_path": all_paths,
        "true_label": all_true.tolist(),
        "true_class":  [LABEL_NAME[int(l)] for l in all_true],
        "pred_prob":   np.round(all_probs, 6).tolist(),
        "pred_label":  pred_labels.tolist(),
        "pred_class":  [LABEL_NAME[int(l)] for l in pred_labels],
        "correctness": correctness,
    })
    return df


# ─────────────────────────────────────────────
# 步骤 2：确定每样本的 Grad-CAM 目标类别
# ─────────────────────────────────────────────

def resolve_target_label(
    pred_label: int,
    pred_prob: float,
    threshold: float,
    target_class: str,
) -> int:
    """根据 target_class 策略决定当前样本的 Grad-CAM 目标类别。

    Returns:
        0 (NORMAL) 或 1 (PNEUMONIA)。
    """
    if target_class == "pneumonia":
        return 1
    elif target_class == "normal":
        return 0
    else:  # "predicted"
        return int(pred_prob >= threshold)


# ─────────────────────────────────────────────
# 步骤 3：对选定样本生成 Grad-CAM
# ─────────────────────────────────────────────

def generate_gradcam_for_samples(
    model: torch.nn.Module,
    model_name: str,
    selected_df: pd.DataFrame,
    image_size: int,
    threshold: float,
    target_class: str,
    output_dir: str,
    device: torch.device,
) -> None:
    """对已选样本逐张生成 Grad-CAM 并保存可视化结果。

    每张样本单独以 batch_size=1 前向，避免多样本 target 不一致带来的复杂性。
    图像保存在 output_dir/{correctness}/ 子目录下。

    Args:
        model       : 已加载权重的模型（将被包装为 GradCAMModelWrapper）。
        model_name  : 用于 get_target_layers。
        selected_df : 含 image_path / pred_prob / pred_label / correctness 等列。
        image_size  : 图像尺寸。
        threshold   : 预测概率阈值。
        target_class: "predicted" / "pneumonia" / "normal"。
        output_dir  : 输出根目录。
        device      : 运算设备。
    """
    _, eval_transform = get_transforms(image_size=image_size, augment=False)

    # 用 Wrapper 使输出变为 [B, 1]，兼容 pytorch-grad-cam
    wrapped_model = GradCAMModelWrapper(model).to(device)
    target_layers = get_target_layers(wrapped_model.model, model_name)

    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    counters = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for _, row in tqdm(
        selected_df.iterrows(),
        total=len(selected_df),
        desc="[GradCAM] 生成热力图",
    ):
        image_path  = row["image_path"]
        pred_prob   = float(row["pred_prob"])
        pred_label  = int(row["pred_label"])
        correctness = str(row["correctness"])
        true_class  = str(row["true_class"])
        pred_class  = str(row["pred_class"])

        # ── 加载单张图 ────────────────────────
        from PIL import Image as PILImage
        try:
            pil_img = PILImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] 无法读取图像，跳过: {image_path} ({e})")
            continue

        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)  # [1, C, H, W]

        # ── 确定 Grad-CAM 目标类别 ────────────
        target_label = resolve_target_label(
            pred_label, pred_prob, threshold, target_class
        )

        # ── 生成热力图 ────────────────────────
        grayscale_cam, rgb_img = generate_cam_for_sample(
            cam, input_tensor, target_label
        )

        # ── 保存 ─────────────────────────────
        sub_dir    = os.path.join(output_dir, correctness)
        sample_id  = f"sample_{counters[correctness]:03d}"
        counters[correctness] += 1

        save_gradcam_images(
            output_dir=sub_dir,
            sample_id=sample_id,
            rgb_img=rgb_img,
            grayscale_cam=grayscale_cam,
            true_class=true_class,
            pred_class=pred_class,
            pred_prob=pred_prob,
            correctness=correctness,
        )

    cam.__exit__(None, None, None)   # 释放 hook

    print("\n[GradCAM] 各类型生成数量：")
    for label, cnt in counters.items():
        if cnt > 0:
            print(f"  {label}: {cnt} 张")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1. 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradCAM] 设备: {device}")

    # 3. 加载模型
    model = load_model_from_checkpoint(
        args.model_name, args.checkpoint_path, device
    )

    # 4. 遍历测试集，得到全量预测结果
    print(f"\n[GradCAM] 正在对测试集进行推断 ...")
    predictions_df = run_inference(
        model=model,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        device=device,
    )

    print(f"[GradCAM] 测试集共 {len(predictions_df)} 张图像")
    dist = predictions_df["correctness"].value_counts().to_dict()
    for k in ("TP", "TN", "FP", "FN"):
        print(f"  {k}: {dist.get(k, 0)}")

    # 5. 保存全量预测 CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "gradcam_selected_predictions.csv")

    # 6. 根据 sample_mode 选样本
    selected_df = select_samples(predictions_df, args.sample_mode, args.num_samples)
    selected_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[GradCAM] 已选 {len(selected_df)} 张样本，预测结果已保存: {csv_path}")

    # 7. 生成 Grad-CAM
    generate_gradcam_for_samples(
        model=model,
        model_name=args.model_name,
        selected_df=selected_df,
        image_size=args.image_size,
        threshold=args.threshold,
        target_class=args.target_class,
        output_dir=args.output_dir,
        device=device,
    )

    print(f"\n[GradCAM] 全部完成。输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
