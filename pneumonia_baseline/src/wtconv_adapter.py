"""
wtconv_adapter.py — A 模块：WTConv 多频特征增强适配器

利用小波卷积（WTConv）增强 DenseNet-121 高层特征图的多频表达能力，
扩大有效感受野，使模型能同时捕获局部纹理变化与大范围肺部密度变化。

依赖：
  - WTConv 仓库（与 pneumonia_baseline 同级目录）
  - PyWavelets（pip install PyWavelets）

目录要求：
  project_root/
  ├── pneumonia_baseline/
  └── WTConv/

如果导入失败，请检查：
  1. WTConv 仓库是否与 pneumonia_baseline 位于同一根目录
  2. 是否已安装 PyWavelets（pip install PyWavelets）
  3. WTConv 仓库结构是否完整（应包含 wtconv/wtconv2d.py）
"""

import os
import sys

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 导入 WTConv2d（仅在此文件中处理 sys.path，不向其他模块扩散）
# ─────────────────────────────────────────────────────────────────────────────

def _import_wtconv2d():
    """定位并导入 WTConv 仓库的 WTConv2d 类。"""
    # pneumonia_baseline 的父目录 = project_root
    _this_dir   = os.path.dirname(os.path.abspath(__file__))          # src/
    _proj_dir   = os.path.dirname(_this_dir)                          # pneumonia_baseline/
    _root_dir   = os.path.dirname(_proj_dir)                          # project_root/
    _wtconv_dir = os.path.join(_root_dir, "WTConv")                   # project_root/WTConv/

    if not os.path.isdir(_wtconv_dir):
        raise ImportError(
            f"找不到 WTConv 仓库目录: {_wtconv_dir}\n"
            "请确认：\n"
            "  1. WTConv 仓库已克隆到与 pneumonia_baseline 同级的目录中\n"
            "  2. 目录结构为：project_root/WTConv/wtconv/wtconv2d.py\n"
            "  3. PyWavelets 已安装（pip install PyWavelets）"
        )

    if _wtconv_dir not in sys.path:
        sys.path.insert(0, _wtconv_dir)

    try:
        from wtconv import WTConv2d
        return WTConv2d
    except ImportError as e:
        raise ImportError(
            f"无法从 WTConv 仓库导入 WTConv2d: {e}\n"
            "请确认：\n"
            "  1. WTConv 仓库结构完整（包含 wtconv/__init__.py 和 wtconv/wtconv2d.py）\n"
            "  2. PyWavelets 已安装（pip install PyWavelets）\n"
            "  3. WTConv 目录路径：{_wtconv_dir}"
        ) from e


WTConv2d = _import_wtconv2d()


# ─────────────────────────────────────────────────────────────────────────────
# WTConvFeatureAdapter — A 模块主体
# ─────────────────────────────────────────────────────────────────────────────

class WTConvFeatureAdapter(nn.Module):
    """基于小波卷积的多频特征增强适配器（A 模块）。

    接在 DenseNet-121 features 输出之后，对 [B, 1024, H, W] 特征图进行
    多频域增强后返回相同 shape，后续再接 GlobalAvgPool + Linear 完成分类。

    结构（带残差连接）：
        input [B, in_channels, H, W]
        → 1×1 Conv  in_channels → adapter_channels
        → BN → ReLU
        → WTConv2d  adapter_channels → adapter_channels  (kernel_size, wt_levels)
        → BN → ReLU
        → 1×1 Conv  adapter_channels → in_channels
        → + input（残差连接）
        output [B, in_channels, H, W]

    Args:
        in_channels      : 输入/输出通道数，默认 1024（DenseNet-121 输出）。
        adapter_channels : 适配器内部中间通道数，默认 256（降维减少计算量）。
        kernel_size      : WTConv2d 的卷积核大小，默认 5。
        wt_levels        : WTConv2d 的小波分解层数，默认 3。

    注意：
        - 输入输出 shape 完全一致（[B, in_channels, H, W]）。
        - 不含 pooling、sigmoid 或分类逻辑。
        - WTConv2d 要求 in_channels == out_channels（depthwise 小波卷积）。
    """

    def __init__(
        self,
        in_channels:      int = 1024,
        adapter_channels: int = 256,
        kernel_size:      int = 5,
        wt_levels:        int = 3,
    ) -> None:
        super().__init__()

        # ── 降维：1×1 Conv ──────────────────────────────────
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, adapter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(inplace=True),
        )

        # ── 小波卷积：多频增强核心 ────────────────────────────
        # WTConv2d 要求 in_channels == out_channels
        self.wtconv = WTConv2d(
            in_channels=adapter_channels,
            out_channels=adapter_channels,
            kernel_size=kernel_size,
            wt_levels=wt_levels,
        )
        self.wt_bn  = nn.BatchNorm2d(adapter_channels)
        self.wt_act = nn.ReLU(inplace=True)

        # ── 升维：1×1 Conv ──────────────────────────────────
        self.expand = nn.Conv2d(adapter_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            同 shape 的增强特征图 [B, in_channels, H, W]。
        """
        residual = x

        out = self.compress(x)           # [B, adapter_channels, H, W]
        out = self.wtconv(out)           # [B, adapter_channels, H, W]
        out = self.wt_act(self.wt_bn(out))
        out = self.expand(out)           # [B, in_channels, H, W]

        return residual + out            # 残差连接
