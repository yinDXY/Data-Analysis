"""基础模型接口。

当前项目主要使用 torchvision backbone，并在 `your_model.py` 中完成模型构建。
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseClassifier(nn.Module, ABC):
    """二分类模型基础类。

    子类应输出单个 logit，shape 为 `[B]` 或 `[B, 1]`。
    损失函数使用 logits，评估阶段再做 sigmoid。
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

