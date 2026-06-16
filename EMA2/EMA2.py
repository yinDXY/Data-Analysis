import torch
import torch.nn as nn
from typing import Optional


class EMA(nn.Module):
    """
    EMA (Efficient Multi-Scale Attention) 高效多尺度注意力模块
    结合通道分组、双方向池化、局部卷积与跨空间交互，增强图像关键区域特征表达
    适配医学影像（胸片）特征重标定，抑制无关背景、突出病灶区域

    Args:
        channels: 输入/输出特征通道数
        c2: 预留输出通道参数（兼容接口，当前未使用）
        factor: 通道分组数，默认 32，分组数决定特征子空间划分粒度
    """
    def __init__(self, channels: int, c2: Optional[int] = None, factor: int = 32):
        super().__init__()

        # 输入参数合法性校验
        if channels <= 0:
            raise ValueError(f"通道数必须大于0，当前输入 channels={channels}")
        if factor <= 0:
            raise ValueError(f"分组数必须大于0，当前输入 factor={factor}")
        if channels % factor != 0:
            raise ValueError(
                f"通道数必须可以被分组数整除，channels={channels}, factor={factor}, 余数={channels % factor}"
            )

        self.channels = channels
        self.groups = factor
        self.group_channels = channels // self.groups  # 单分组通道数

        # 基础激活与池化层
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d(output_size=(None, 1))  # 垂直方向一维池化
        self.pool_w = nn.AdaptiveAvgPool2d(output_size=(1, None))  # 水平方向一维池化

        # 归一化 & 卷积层
        self.gn = nn.GroupNorm(num_groups=self.group_channels, num_channels=self.group_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入特征图, 维度 [batch, channels, height, width]
        Returns:
            注意力重标定后的特征图, 维度与输入一致
        """
        # 解析输入维度
        b, c, h, w = x.shape
        assert c == self.channels, f"输入通道不匹配，预期{self.channels}，实际{c}"

        # 1. 通道分组：将通道维度划分为多组，降低计算量
        group_x = x.reshape(b * self.groups, self.group_channels, h, w)

        # 2. 水平、垂直双方向一维池化，提取空间方向上下文
        feat_h = self.pool_h(group_x)    # [b*g, gc, h, 1]
        feat_w = self.pool_w(group_x)    # [b*g, gc, 1, w]
        feat_w = feat_w.permute(0, 1, 3, 2)  # 维度对齐 [b*g, gc, w, 1]

        # 3. 拼接双方向特征 + 1x1卷积融合
        hw_concat = torch.cat([feat_h, feat_w], dim=2)
        hw_fused = self.conv1x1(hw_concat)

        # 拆分融合特征，得到两个方向注意力权重
        att_h, att_w = torch.split(hw_fused, split_size_or_sections=[h, w], dim=2)
        att_w = att_w.permute(0, 1, 3, 2)  # [b*g, gc, 1, w]

        # 4. 分支1：方向感知全局特征（空间长程依赖）
        branch1 = group_x * att_h.sigmoid() * att_w.sigmoid()
        branch1 = self.gn(branch1)

        # 5. 分支2：3x3卷积局部特征（局部邻域上下文）
        branch2 = self.conv3x3(group_x)

        # 6. 跨分支空间交互，计算全局注意力权重
        # 分支1 全局描述
        g1 = self.agp(branch1).reshape(b * self.groups, self.group_channels, 1)
        g1 = g1.permute(0, 2, 1)
        score1 = self.softmax(g1)

        # 分支2 全局描述
        g2 = self.agp(branch2).reshape(b * self.groups, self.group_channels, 1)
        g2 = g2.permute(0, 2, 1)
        score2 = self.softmax(g2)

        # 特征展平为空间序列
        flat_branch1 = branch1.reshape(b * self.groups, self.group_channels, h * w)
        flat_branch2 = branch2.reshape(b * self.groups, self.group_channels, h * w)

        # 矩阵乘法计算跨空间注意力
        cross_att1 = torch.matmul(score1, flat_branch2)
        cross_att2 = torch.matmul(score2, flat_branch1)
        total_weights = cross_att1 + cross_att2

        # 还原空间维度并生成最终注意力掩码
        total_weights = total_weights.reshape(b * self.groups, 1, h, w)
        att_mask = total_weights.sigmoid()

        # 7. 注意力加权 + 还原原始维度
        out = group_x * att_mask
        out = out.reshape(b, c, h, w)

        return out