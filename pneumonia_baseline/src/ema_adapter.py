"""
ema_adapter.py — B 模块：EMA 多尺度空间注意力

EMA（Efficient Multi-scale Attention）通过沿空间维度进行多尺度通道注意力
重标定，使模型聚焦于胸片中与肺炎相关的局部纹理、浸润阴影和肺野密度变化区域。

EMA 类来源（最小实现复制自）：
    https://github.com/YOLOonMe/EMA-attention-module
    原始文件：EMA-attention-module/EMA_attention_module.py
    原始类名：EMA
    本文件不修改 EMA-attention-module 仓库源码，仅在此处维护一份最小副本。

依赖：无额外依赖（纯 PyTorch）
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# EMA 最小实现（复制自 EMA-attention-module 仓库，未做任何修改）
# 原始作者：YOLOonMe  https://github.com/YOLOonMe/EMA-attention-module
# ─────────────────────────────────────────────────────────────────────────────

class _EMA(nn.Module):
    """EMA 核心实现（内部类，外部请使用 EMAAttention）。

    原始论文：Efficient Multi-Scale Attention Module with Cross-Spatial Learning
    来源仓库：https://github.com/YOLOonMe/EMA-attention-module
    """

    def __init__(self, channels: int, c2=None, factor: int = 32):
        super().__init__()
        self.groups = factor
        if channels % self.groups != 0:
            raise ValueError(
                f"EMA: channels ({channels}) 必须能被 factor ({factor}) 整除。\n"
                f"请调整 factor，使 channels % factor == 0。\n"
                f"例如：channels=1024, factor=32 → 1024 % 32 = 0 ✓"
            )
        assert channels // self.groups > 0, (
            f"EMA: channels // factor 必须 > 0，"
            f"当前 channels={channels}, factor={factor}"
        )
        self.softmax  = nn.Softmax(-1)
        self.agp      = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h   = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w   = nn.AdaptiveAvgPool2d((1, None))
        self.gn       = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1  = nn.Conv2d(
            channels // self.groups, channels // self.groups,
            kernel_size=1, stride=1, padding=0,
        )
        self.conv3x3  = nn.Conv2d(
            channels // self.groups, channels // self.groups,
            kernel_size=3, stride=1, padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)        # [b*g, c//g, h, w]
        x_h = self.pool_h(group_x)                            # [b*g, c//g, h, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)        # [b*g, c//g, 1, w] → [b*g, c//g, w, 1]
        hw  = self.conv1x1(torch.cat([x_h, x_w], dim=2))      # [b*g, c//g, h+w, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(
            group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        )
        x2  = self.conv3x3(group_x)
        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )                                                      # [b*g, 1, c//g]
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)   # [b*g, c//g, h*w]
        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )                                                      # [b*g, 1, c//g]
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)   # [b*g, c//g, h*w]
        weights = (
            torch.matmul(x11, x12) + torch.matmul(x21, x22)
        ).reshape(b * self.groups, 1, h, w)                   # [b*g, 1, h, w]
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# ─────────────────────────────────────────────────────────────────────────────
# EMAAttention — B 模块主体
# ─────────────────────────────────────────────────────────────────────────────

class EMAAttention(nn.Module):
    """B 模块：EMA 多尺度空间注意力适配器。

    封装 EMA 模块，默认用于 DenseNet-121 最后一层特征图 [B, 1024, H, W]。
    输入输出 shape 完全一致，不含 pooling、分类或 sigmoid。

    Args:
        channels : 特征图通道数，默认 1024（DenseNet-121 输出）。
        factor   : EMA 分组因子，默认 32。要求 channels % factor == 0。

    Raises:
        ValueError: channels 不能被 factor 整除时抛出，附带清晰说明。

    Example::

        attn = EMAAttention(channels=1024, factor=32)
        x = torch.randn(2, 1024, 7, 7)
        out = attn(x)          # [2, 1024, 7, 7]
        assert out.shape == x.shape
    """

    def __init__(self, channels: int = 1024, factor: int = 32) -> None:
        super().__init__()
        self.ema = _EMA(channels=channels, factor=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, channels, H, W]

        Returns:
            注意力加权后的特征图，shape 与输入相同。
        """
        return self.ema(x)
