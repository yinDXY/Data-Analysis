"""
models.py — CNN 模型构建

支持的预训练 backbone：
    resnet50 / densenet121 / efficientnet_b0

所有模型输出 shape [batch_size]（单 logit），配合 BCEWithLogitsLoss 使用。
评估阶段对输出 sigmoid 即可得到概率。
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
)

# ─────────────────────────────────────────────
# 支持的模型列表
# ─────────────────────────────────────────────

_SUPPORTED_MODELS = ["resnet50", "densenet121", "efficientnet_b0"]


def list_supported_models():
    """返回当前支持的模型名称列表。"""
    return list(_SUPPORTED_MODELS)


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────

def get_model(
    model_name: str,
    pretrained: bool = True,
    use_wtconv: bool = False,
) -> nn.Module:
    """构建并返回指定 backbone 的二分类模型。

    最后分类层替换为输出 1 个 logit 的线性层，
    配合 BCEWithLogitsLoss 使用（无需手动加 sigmoid）。

    Args:
        model_name : 模型名称，见 list_supported_models()。
        pretrained : 是否加载 ImageNet 预训练权重，默认 True。
        use_wtconv : 是否启用 WTConv A 模块（仅支持 densenet121）。

    Returns:
        nn.Module，forward 输出 shape [batch_size]。

    Raises:
        ValueError: model_name 不在支持列表中，或 use_wtconv=True 且 model_name != densenet121。
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"不支持的模型: '{model_name}'。"
            f"可选模型: {_SUPPORTED_MODELS}"
        )

    if use_wtconv and model_name != "densenet121":
        raise ValueError(
            "WTConv A module currently supports only densenet121."
        )

    if use_wtconv:
        model = DenseNet121WTConv(pretrained=pretrained)
        print(f"[Model] 已加载: {model_name}  |  pretrained={pretrained}  |  use_wtconv=True")
        return model

    if model_name == "resnet50":
        model = _build_resnet50(pretrained)
    elif model_name == "densenet121":
        model = _build_densenet121(pretrained)
    elif model_name == "efficientnet_b0":
        model = _build_efficientnet_b0(pretrained)

    print(f"[Model] 已加载: {model_name}  |  pretrained={pretrained}  |  use_wtconv=False")
    return model


# ─────────────────────────────────────────────
# 各 backbone 构建（内部函数）
# ─────────────────────────────────────────────

def _build_resnet50(pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Linear(in_features, 1)        # 替换 fc
    return model


def _build_densenet121(pretrained: bool) -> nn.Module:
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Linear(in_features, 1)  # 替换 classifier
    return model


def _build_efficientnet_b0(pretrained: bool) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)  # 替换 classifier[1]
    return model


# ─────────────────────────────────────────────
# A 模块：DenseNet-121 + WTConv Feature Adapter
# ─────────────────────────────────────────────

class DenseNet121WTConv(nn.Module):
    """DenseNet-121 + WTConv 多频特征增强适配器（A 模块）。

    结构：
        DenseNet-121 features  →  ReLU
        → WTConvFeatureAdapter (1024→256→1024, residual)
        → AdaptiveAvgPool2d(1)
        → flatten
        → Linear(1024, 1)

    输出 shape [B]（单 logit），兼容 BCEWithLogitsLoss。
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # 延迟导入，避免未安装 WTConv 时影响 baseline 使用
        from src.wtconv_adapter import WTConvFeatureAdapter

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet121(weights=weights)

        self.features   = base.features          # 输出 [B, 1024, H, W]
        self.relu       = nn.ReLU(inplace=True)
        self.adapter    = WTConvFeatureAdapter()  # [B,1024,H,W] → [B,1024,H,W]
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.features(x))   # [B, 1024, H, W]
        x = self.adapter(x)               # [B, 1024, H, W]
        x = self.pool(x)                  # [B, 1024, 1, 1]
        x = x.flatten(1)                  # [B, 1024]
        return self.classifier(x)         # [B, 1]（engine 会再 .squeeze(1)）


# ─────────────────────────────────────────────
# 冻结 / 解冻工具（可选）
# ─────────────────────────────────────────────

def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """冻结 backbone 所有参数，只训练分类头。

    适用于迁移学习第一阶段（先训练分类头，再全局微调）。
    默认 baseline 训练不调用此函数。

    Args:
        model     : get_model 返回的模型。
        model_name: 模型名称，用于定位分类头。
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: '{model_name}'")

    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 再解冻分类头
    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "densenet121":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "efficientnet_b0":
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    print(f"[Model] backbone 已冻结，仅分类头参与训练")


def unfreeze_all(model: nn.Module) -> None:
    """解冻模型所有参数，用于全局微调阶段。

    Args:
        model: get_model 返回的模型。
    """
    for param in model.parameters():
        param.requires_grad = True
    print(f"[Model] 所有参数已解冻")


def unfreeze_last_block(model: nn.Module, model_name: str) -> None:
    """冒结所有参数，再只解冻最后一个 block + 分类头。

    适用于迁移学习中间阶段（下沉程度介于 freeze_backbone 与 unfreeze_all 之间）。

    Args:
        model     : get_model 返回的模型。
        model_name: 模型名称，用于定位各 backbone 的 block 位置。
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: '{model_name}'")

    # 先冒结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 再解冻最后一个 block + 分类头
    if model_name == "resnet50":
        for module in [model.layer4, model.fc]:
            for param in module.parameters():
                param.requires_grad = True
    elif model_name == "densenet121":
        for module in [model.features.denseblock4, model.features.norm5, model.classifier]:
            for param in module.parameters():
                param.requires_grad = True
    elif model_name == "efficientnet_b0":
        for module in list(model.features[-2:]) + [model.classifier]:
            for param in module.parameters():
                param.requires_grad = True

    print(f"[Model] 最后 block + 分类头已解冻")
