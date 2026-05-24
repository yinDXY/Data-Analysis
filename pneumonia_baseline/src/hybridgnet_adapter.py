"""
hybridgnet_adapter.py — HybridGNet 肺部分割适配器

使用 HybridGNet 从胸部 X-ray 中预测肺部轮廓 landmarks，
并生成 binary lung mask，用于离线 ROI 裁剪预处理。

依赖：
  - HybridGNet 仓库（与 pneumonia_baseline 同级目录）
  - torch-geometric（HybridGNet 内部依赖）
  - opencv-python
  - scipy

推理流程：
  1. 读取灰度图并归一化到 [0, 1]
  2. Center-pad 到正方形
  3. Resize 到 1024×1024
  4. 输入 HybridGNet，输出 landmarks（归一化到 [0,1] 的 120 点）
  5. 坐标还原到原始图像尺寸
  6. cv2.drawContours 生成 binary lung mask

Landmark 分布（genMatrixesLH，Lungs + Heart）：
  Right Lung : [0:44]   — 44 points
  Left Lung  : [44:94]  — 50 points
  Heart      : [94:120] — 26 points（仅用于模型推理，不生成 mask）
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# HybridGNetSegmenter
# ─────────────────────────────────────────────────────────────────────────────

class HybridGNetSegmenter:
    """HybridGNet 推理封装，输出 binary lung mask。

    Args:
        hybridgnet_dir : HybridGNet 仓库根目录路径。
        weights_path   : HybridGNet 预训练权重 (.pt) 路径。
        device         : 推理设备，"cuda" 或 "cpu"。
                         若指定 "cuda" 但 CUDA 不可用，自动降级到 CPU。
    """

    # Landmark 索引边界
    RLUNG_END = 44      # Right Lung: [0, RLUNG_END)
    LLUNG_END = 94      # Left Lung:  [RLUNG_END, LLUNG_END)
    INPUT_SIZE = 1024

    def __init__(
        self,
        hybridgnet_dir: str,
        weights_path: str,
        device: str = "cuda",
    ) -> None:
        # ── 检查权重文件 ──────────────────────────────────────
        weights_path = str(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"HybridGNet 权重文件不存在: {weights_path}\n"
                "请下载官方预训练权重并通过 --weights_path 指定路径。"
            )

        # ── 注入 HybridGNet 到 sys.path ───────────────────────
        hybridgnet_dir = str(hybridgnet_dir)
        if not os.path.isdir(hybridgnet_dir):
            raise FileNotFoundError(
                f"HybridGNet 仓库目录不存在: {hybridgnet_dir}"
            )
        if hybridgnet_dir not in sys.path:
            sys.path.insert(0, hybridgnet_dir)

        # ── 导入 HybridGNet 内部模块 ─────────────────────────
        try:
            import scipy.sparse as sp
            from utils.utils import scipy_to_torch_sparse, genMatrixesLH
            from models.hybrid import Hybrid
            import models.hybrid as _hybrid_module
            from models.chebConv import Pool as _PoolClass
        except ImportError as e:
            raise ImportError(
                f"无法导入 HybridGNet 模块: {e}\n"
                "请确认：\n"
                "  1. hybridgnet_dir 路径正确\n"
                "  2. torch-geometric 已安装（pip install torch-geometric）\n"
                "  3. scipy 已安装（pip install scipy）"
            ) from e

        # ── Monkey-patch：解决 PyG 2.4+ 与 HybridGNet 的兼容性问题 ──
        #
        # 问题 1：ChebConv（PyG 2.4+ Jinja 模板生成器）
        #   HybridGNet 的 ChebConv 是 PyG ChebConv 的子类，只重写了
        #   reset_parameters()。PyG 2.4+ 的 _set_jittable_templates() 在
        #   子类中无法通过 Inspector 检测到继承的 message() 参数，导致生成
        #   的 propagate() 签名为空（不接受 x/norm），引发 TypeError。
        #   修复：在调用 Hybrid() 构造函数之前，将 models.hybrid 模块全局
        #   命名空间中的 ChebConv 替换为 PyG 的原生实现（包含正确的 propagate）。
        #   由于只重写了初始化方式（训练时权重随机初始化），加载预训练权重时
        #   参数结构完全相同，weights 可以正常加载。
        #
        # 问题 2：Pool（自定义 MessagePassing 子类）
        #   Pool.forward() 调用 self.propagate(..., x=x, ...)，在 PyG 2.4+ 中
        #   对于没有声明 x 的自定义 MessagePassing 子类会失败。
        #   修复：将 Pool.forward 替换为等价的 sparse matmul 实现。
        from torch_geometric.nn.conv.cheb_conv import ChebConv as _PyGChebConv

        # 修复 1：用 PyG 原生 ChebConv 替换 models.hybrid 里的绑定
        # （Hybrid.__init__ 执行时会在此命名空间中查找 ChebConv）
        _hybrid_module.ChebConv = _PyGChebConv

        # 修复 2：Pool.forward → sparse matmul（数学等价，避免旧 propagate API）
        def _pool_forward_compat(_self, x, pool_mat, dtype=None):
            # pool_mat: [N_out, N_in] sparse（例如 U: [120, 60]，用于上采样）
            # x       : [B, N_in, features]
            # 等价运算: pool_mat @ x → [B, N_out, features]
            b, n_in, f = x.shape
            x_2d = x.permute(1, 0, 2).reshape(n_in, b * f).contiguous().float()
            out_2d = torch.sparse.mm(pool_mat.coalesce(), x_2d)  # [N_out, B*f]
            n_out = pool_mat.shape[0]
            return out_2d.reshape(n_out, b, f).permute(1, 0, 2).contiguous()

        _PoolClass.forward = _pool_forward_compat

        # ── 确定推理设备 ──────────────────────────────────────
        if device == "cuda" and not torch.cuda.is_available():
            print("[hybridgnet_adapter] 警告: CUDA 不可用，自动切换到 CPU。")
            device = "cpu"
        self.device = torch.device(device)

        # ── 构建图矩阵（Lungs + Heart，120 nodes）─────────────
        A, AD, D, U = genMatrixesLH()
        N1 = A.shape[0]   # 120 (full resolution)
        N2 = AD.shape[0]  # ~60 (downsampled)

        A_sp  = sp.csc_matrix(A).tocoo()
        AD_sp = sp.csc_matrix(AD).tocoo()
        D_sp  = sp.csc_matrix(D).tocoo()
        U_sp  = sp.csc_matrix(U).tocoo()

        # 6 层邻接矩阵（3 层全分辨率 + 3 层下采样）
        A_list = [A_sp.copy(), A_sp.copy(), A_sp.copy(),
                  AD_sp.copy(), AD_sp.copy(), AD_sp.copy()]
        D_list = [D_sp.copy()]
        U_list = [U_sp.copy()]

        A_t = [scipy_to_torch_sparse(x).to(self.device) for x in A_list]
        D_t = [scipy_to_torch_sparse(x).to(self.device) for x in D_list]
        U_t = [scipy_to_torch_sparse(x).to(self.device) for x in U_list]

        # ── 构建模型 ─────────────────────────────────────────
        config = {
            "inputsize": self.INPUT_SIZE,
            "latents":   64,
            "filters":   [2, 32, 32, 32, 16, 16, 16],
            "K":         6,
            "n_nodes":   [N1, N1, N1, N2, N2, N2],
        }
        model = Hybrid(config, D_t, U_t, A_t)

        # ── 加载权重 ─────────────────────────────────────────
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        self.model = model

    # ──────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────

    def _pad_to_square(
        self,
        image: np.ndarray,
    ):
        """Center-pad 灰度图至正方形，返回 (padded, pad_top, pad_left)。

        Args:
            image: float32 灰度图，shape (H, W)，值域 [0, 1]。

        Returns:
            padded   : 正方形图像，shape (max(H,W), max(H,W))。
            pad_top  : 上方填充的行数。
            pad_left : 左方填充的列数。
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        pad_top    = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left   = (max_dim - w) // 2
        pad_right  = max_dim - w - pad_left
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0.0,
        )
        return padded, pad_top, pad_left

    # ──────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────

    def predict_landmarks(self, image_path: str) -> np.ndarray:
        """预测肺部 landmarks，坐标已还原到原始图像坐标系。

        处理流程：
          1. 读取灰度图，归一化到 [0, 1]
          2. Center-pad 至正方形（padded_size = max(H, W)）
          3. Resize 到 1024×1024
          4. 前向推理，得到归一化 landmarks（[0, 1] 相对 1024×1024）
          5. × padded_size → padded 坐标
          6. − [pad_left, pad_top] → 原始图像坐标

        Args:
            image_path: 输入图像路径（支持 JPEG/PNG）。

        Returns:
            landmarks : shape [N, 2]，其中 N≤120，坐标格式 (x, y)=(col, row)，
                        float32，已裁剪到 [0, orig_w-1] × [0, orig_h-1]。

        Raises:
            ValueError        : 图像无法读取。
            RuntimeError      : 模型推理失败。
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像（路径不存在或格式不支持）: {image_path}")

        orig_h, orig_w = image.shape

        # 归一化
        image_f = image.astype(np.float32) / 255.0

        # Pad to square
        padded, pad_top, pad_left = self._pad_to_square(image_f)
        padded_size = float(padded.shape[0])  # max(H, W)

        # Resize to 1024×1024
        resized = cv2.resize(
            padded, (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # 转为 tensor [1, 1, 1024, 1024]
        tensor = (
            torch.from_numpy(resized)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # 推理
        try:
            with torch.no_grad():
                out = self.model(tensor)
                if isinstance(out, tuple):
                    out = out[0]
        except Exception as e:
            raise RuntimeError(f"HybridGNet 推理失败: {e}") from e

        # out: [1, N, 2]，值域 [0, 1]（相对于 1024×1024 输入归一化）
        landmarks_norm = out.squeeze(0).cpu().numpy()  # [N, 2]

        # 还原到原始图像坐标
        # step 1: 乘以 padded_size → padded square 像素坐标
        # step 2: 减去 padding 偏移 → 原始图像坐标
        landmarks = landmarks_norm * padded_size
        landmarks[:, 0] -= pad_left   # x (col)
        landmarks[:, 1] -= pad_top    # y (row)

        # 裁剪到有效范围
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0.0, float(orig_w - 1))
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0.0, float(orig_h - 1))

        return landmarks.astype(np.float32)

    def landmarks_to_lung_mask(
        self,
        landmarks: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """将 landmarks 转换为 binary lung mask。

        只使用 Right Lung [0:44] 和 Left Lung [44:94]，不使用 Heart。

        Args:
            landmarks : [N, 2] float32，坐标格式 (x, y)=(col, row)。
            height    : 原始图像高度。
            width     : 原始图像宽度。

        Returns:
            mask: uint8 ndarray，shape (height, width)，值 0 或 1。
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        def _draw_lung(pts: np.ndarray) -> None:
            """将 N×2 float 轮廓点填充绘制到 mask。"""
            if len(pts) < 3:
                return
            contour = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(mask, [contour], contourIdx=-1,
                             color=1, thickness=cv2.FILLED)

        _draw_lung(landmarks[0:self.RLUNG_END])          # Right Lung
        _draw_lung(landmarks[self.RLUNG_END:self.LLUNG_END])  # Left Lung

        return mask

    def predict_lung_mask(self, image_path: str) -> np.ndarray:
        """一步完成 landmarks 预测和 mask 生成。

        Args:
            image_path: 输入图像路径。

        Returns:
            mask: uint8 ndarray，shape 与原始图像一致，值 0 或 1。

        Raises:
            ValueError   : 图像无法读取。
            RuntimeError : 模型推理失败。
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        orig_h, orig_w = image.shape
        landmarks = self.predict_landmarks(image_path)
        mask = self.landmarks_to_lung_mask(landmarks, orig_h, orig_w)
        return mask
