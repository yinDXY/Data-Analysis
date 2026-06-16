import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from .util import wavelet


class WTConv1d(nn.Module):
    """
    1维小波变换卷积层 (Wavelet Transform 1D Convolution)
    将小波变换与卷积结合，在多尺度小波域进行卷积操作后逆变换融合
    
    Args:
        in_channels (int): 输入通道数（需等于输出通道数）
        out_channels (int): 输出通道数（需等于输入通道数）
        kernel_size (int): 卷积核大小，默认5
        stride (int): 输出步幅，默认1
        bias (bool): 是否使用偏置，默认True
        wt_levels (int): 小波变换的层数，默认1
        wt_type (str): 小波基类型，默认'db1'
    
    Shape:
        - Input: (N, C, L)
        - Output: (N, C, L//stride) (stride>1时) 或 (N, C, L) (stride=1时)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
        wt_type: str = 'db1'
    ):
        super().__init__()

        # 校验输入输出通道一致性
        if in_channels != out_channels:
            raise ValueError(f"in_channels ({in_channels}) must equal out_channels ({out_channels})")
        if wt_levels < 1:
            raise ValueError(f"wt_levels must be ≥1, got {wt_levels}")
        if stride < 1:
            raise ValueError(f"stride must be ≥1, got {stride}")

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.kernel_size = kernel_size

        # 创建小波变换/逆变换滤波器（不可训练）
        wt_filter, iwt_filter = wavelet.create_1d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        # 计算same padding值（兼容旧版PyTorch）
        self.padding = (kernel_size - 1) // 2

        # 基础卷积分支
        self.base_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1])

        # 小波域卷积分支（多层）
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels * 2,
                out_channels=in_channels * 2,
                kernel_size=kernel_size,
                padding=self.padding,
                stride=1,
                dilation=1,
                groups=in_channels * 2,
                bias=False
            ) for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 2, 1], init_scale=0.1) 
            for _ in range(wt_levels)
        ])

        # 下采样层（步幅>1时）
        self.do_stride = nn.AvgPool1d(kernel_size=stride, stride=stride) if stride > 1 else None

    def _wavelet_forward_level(self, x: torch.Tensor, level_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单层级小波变换前向处理"""
        curr_shape = x.shape
        # 补零确保长度为偶数
        if curr_shape[2] % 2 != 0:
            x = F.pad(x, (0, 1))  # 仅在最后一维右侧补零
        
        # 小波变换
        x_wt = wavelet.wavelet_1d_transform(x, self.wt_filter)
        x_ll = x_wt[:, :, 0, :]  # 低频分量
        
        # 小波域卷积处理
        batch, chan, coeff, length = x_wt.shape
        x_wt_reshaped = x_wt.reshape(batch, chan * 2, length)
        x_wt_reshaped = self.wavelet_scale[level_idx](self.wavelet_convs[level_idx](x_wt_reshaped))
        x_wt_processed = x_wt_reshaped.reshape(batch, chan, coeff, length)
        
        return x_wt_processed[:, :, 0, :], x_wt_processed[:, :, 1:2, :], curr_shape

    def _inverse_wavelet_level(
        self, 
        x_ll: torch.Tensor, 
        x_h: torch.Tensor, 
        next_x_ll: torch.Tensor, 
        orig_shape: torch.Size
    ) -> torch.Tensor:
        """单层级逆小波变换处理"""
        x_ll = x_ll + next_x_ll
        x_cat = torch.cat([x_ll.unsqueeze(2), x_h], dim=2)
        x_iwt = wavelet.inverse_1d_wavelet_transform(x_cat, self.iwt_filter)
        # 裁剪回原始形状
        x_iwt = x_iwt[:, :, :orig_shape[2]]
        return x_iwt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始化存储容器
        x_ll_levels: List[torch.Tensor] = []
        x_h_levels: List[torch.Tensor] = []
        shape_levels: List[torch.Size] = []
        curr_x_ll = x

        # 小波变换+卷积处理（前向逐层）
        for level in range(self.wt_levels):
            x_ll, x_h, curr_shape = self._wavelet_forward_level(curr_x_ll, level)
            x_ll_levels.append(x_ll)
            x_h_levels.append(x_h)
            shape_levels.append(curr_shape)
            curr_x_ll = x_ll

        # 逆小波变换（反向逐层）
        next_x_ll = torch.zeros_like(x_ll_levels[-1]) if self.wt_levels > 0 else 0
        for level in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_levels.pop()
            curr_x_h = x_h_levels.pop()
            curr_shape = shape_levels.pop()
            next_x_ll = self._inverse_wavelet_level(curr_x_ll, curr_x_h, next_x_ll, curr_shape)

        # 基础卷积分支 + 小波分支融合
        x_base = self.base_scale(self.base_conv(x))
        x_fused = x_base + next_x_ll

        # 步幅下采样
        if self.do_stride is not None:
            x_fused = self.do_stride(x_fused)

        # 校验所有层级都已处理
        assert len(x_ll_levels) == 0 and len(x_h_levels) == 0 and len(shape_levels) == 0

        return x_fused


class _ScaleModule(nn.Module):
    """
    缩放模块：对输入张量进行逐通道缩放（无偏置）
    
    Args:
        dims (List[int]): 缩放权重的维度
        init_scale (float): 初始缩放值，默认1.0
    """
    def __init__(self, dims: List[int], init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight