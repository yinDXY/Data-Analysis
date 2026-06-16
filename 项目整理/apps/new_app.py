#!/usr/bin/env python
"""
new_app.py — 胸部 X-Ray 肺炎检测系统  PySide6 + pyqtgraph 增强版

运行：
    python new_app.py

新特性（对比 app.py tkinter 版）：
  · QGraphicsView 图像查看器（滚轮缩放 / 拖拽平移 / 适配窗口）
  · 原图 / Grad-CAM 热力图 / 叠加图  三视图实时切换
  · 自定义 QPainter 概率仪表盘（带动画过渡效果）
  · pyqtgraph 概率柱状图（NORMAL vs PNEUMONIA，含阈值线）
  · pyqtgraph 推理历史折线（记录最近 20 次 P(PNEUMONIA)）
  · QThread 后台线程加载模型与推理，UI 永不冻结
  · 支持 use_wtconv / use_ema 可选模块

依赖：
    pip install PySide6 pyqtgraph
"""

from __future__ import annotations

import math
import os
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ─── PySide6 ─────────────────────────────────────────────────
from PySide6.QtCore import Qt, QRectF, QSize, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# ─── pyqtgraph ───────────────────────────────────────────────
import pyqtgraph as pg

pg.setConfigOptions(antialias=True, useOpenGL=False)
pg.setConfigOption("background", "#1B1B2B")
pg.setConfigOption("foreground", "#B8C0E0")

# ─── 项目模块 ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from src.models import get_model
from test import (
    LABEL_NAME,
    denormalize_image,
    get_eval_transform,
    get_gradcam,
    infer_single,
    infer_true_label,
    resolve_device,
)


# ══════════════════════════════════════════════════════════════
#  主题色 & Qt 样式表
# ══════════════════════════════════════════════════════════════

C = dict(
    bg="#141421",
    surface="#1B1B2B",
    panel="#202034",
    card="#272742",
    entry="#11111D",
    border="#343454",
    fg="#EEF2FF",
    fg2="#B8C0E0",
    muted="#7F849C",
    accent="#CBA6F7",
    accent2="#89B4FA",
    cyan="#89DCEB",
    green="#A6E3A1",
    red="#F38BA8",
    yellow="#F9E2AF",
)

_QSS = f"""
* {{
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 9pt;
}}
QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['fg']};
}}
QGroupBox {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    margin-top: 1.5em;
    padding: 6px 6px 8px 6px;
    font-weight: bold;
    color: {C['fg']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: {C['fg']};
}}
QPushButton {{
    background: {C['card']};
    color: {C['fg']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: bold;
}}
QPushButton:hover  {{ background: #343454; border-color: {C['accent']}; }}
QPushButton:pressed {{ background: {C['entry']}; }}
QPushButton:disabled {{
    background: {C['entry']}; color: {C['muted']};
    border-color: {C['border']};
}}
QPushButton[role="accent"] {{
    background: {C['accent2']}; color: #101020; border: none;
}}
QPushButton[role="accent"]:hover {{ background: #A6C8FF; }}
QPushButton[role="accent"]:disabled {{
    background: {C['entry']}; color: {C['muted']}; border: 1px solid {C['border']};
}}
QPushButton[role="green"] {{
    background: {C['green']}; color: #101020; border: none;
}}
QPushButton[role="green"]:hover {{ background: #BCFFB7; }}
QPushButton[role="green"]:disabled {{
    background: {C['entry']}; color: {C['muted']}; border: 1px solid {C['border']};
}}
QPushButton[role="toggle"] {{
    background: {C['entry']}; color: {C['muted']};
    border: 1px solid {C['border']};
    padding: 4px 10px; border-radius: 4px;
    font-size: 8pt; font-weight: normal;
}}
QPushButton[role="toggle"]:checked {{
    background: {C['accent']}; color: #11111B;
    border-color: {C['accent']};
}}
QPushButton[role="toggle"]:hover {{
    border-color: {C['accent2']};
}}
QComboBox {{
    background: {C['entry']}; color: {C['fg']};
    border: 1px solid {C['border']}; border-radius: 4px;
    padding: 4px 8px;
    font-family: Consolas, monospace;
}}
QComboBox QAbstractItemView {{
    background: {C['entry']}; color: {C['fg']};
    selection-background-color: {C['accent']}; selection-color: #11111B;
    border: 1px solid {C['border']};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QLineEdit {{
    background: {C['entry']}; color: {C['fg']};
    border: 1px solid {C['border']}; border-radius: 4px;
    padding: 4px 8px;
    font-family: Consolas, monospace; font-size: 8pt;
}}
QLineEdit:focus {{ border-color: {C['accent2']}; }}
QSlider::groove:horizontal {{
    background: {C['entry']}; height: 4px; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['accent']}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {C['accent']}; border-radius: 2px;
}}
QCheckBox {{ color: {C['fg2']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {C['border']}; border-radius: 3px;
    background: {C['entry']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']}; border-color: {C['accent']};
}}
QScrollArea {{ background: {C['panel']}; border: none; }}
QScrollBar:vertical {{
    background: {C['entry']}; width: 5px;
    border-radius: 2px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C['border']}; border-radius: 2px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {C['entry']}; height: 5px; border-radius: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {C['border']}; border-radius: 2px; min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QStatusBar {{
    background: #0D0D1A; color: {C['fg2']};
    font-family: Consolas, monospace; font-size: 8pt;
    border-top: 1px solid {C['border']};
}}
QProgressBar {{
    background: {C['entry']}; border: none;
    border-radius: 2px; max-height: 4px; text-align: center;
}}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 2px; }}
QSplitter::handle {{ background: {C['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
"""

_FONT_TITLE = QFont("Microsoft YaHei", 16, QFont.Weight.Bold)
_FONT_MONO  = QFont("Consolas", 8)
_FONT_BOLD  = QFont("Microsoft YaHei", 9, QFont.Weight.Bold)
_FONT_SMALL = QFont("Microsoft YaHei", 8)

# 直接样式表：避免 QPushButton[role=] 属性选择器在 Qt 中的不稳定问题
_SS_ACCENT = (
    f"QPushButton {{ background: {C['accent2']}; color: #101020; border: none;"
    f" border-radius: 5px; padding: 7px 16px; font-weight: bold; font-size: 9pt; }}"
    f"QPushButton:hover {{ background: #A6C8FF; }}"
    f"QPushButton:pressed {{ background: #6B9FE4; color: #101020; }}"
    f"QPushButton:disabled {{ background: {C['entry']}; color: {C['muted']};"
    f" border: 1px solid {C['border']}; }}"
)
_SS_GREEN = (
    f"QPushButton {{ background: {C['green']}; color: #101020; border: none;"
    f" border-radius: 5px; padding: 7px 16px; font-weight: bold; font-size: 9pt; }}"
    f"QPushButton:hover {{ background: #BCFFB7; }}"
    f"QPushButton:pressed {{ background: #88C484; color: #101020; }}"
    f"QPushButton:disabled {{ background: {C['entry']}; color: {C['muted']};"
    f" border: 1px solid {C['border']}; }}"
)
# 视图切换按钮：选中态用蓝色填充，未选中态用暗色，禁用态更暗
_SS_TOGGLE_ON = (
    "QPushButton { background: #89B4FA; color: #101020; border: none;"
    " border-radius: 4px; padding: 4px 12px; font-weight: bold; font-size: 8pt; }"
    "QPushButton:hover { background: #A6C8FF; }"
)
_SS_TOGGLE_OFF = (
    f"QPushButton {{ background: {C['entry']}; color: {C['fg2']};"
    f" border: 1px solid {C['border']}; border-radius: 4px;"
    f" padding: 4px 12px; font-size: 8pt; }}"
    f"QPushButton:hover {{ border-color: #89B4FA; color: #EEF2FF; }}"
)
_SS_TOGGLE_DIS = (
    f"QPushButton {{ background: {C['entry']}; color: {C['muted']};"
    f" border: 1px solid #222238; border-radius: 4px;"
    f" padding: 4px 12px; font-size: 8pt; }}"
)


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def _numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """(H, W, 3) float32 [0,1] → QPixmap"""
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    h, w = u8.shape[:2]
    qimg = QImage(u8.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _gray_to_jet_qpixmap(
    gray: np.ndarray,
    rgb_img: np.ndarray | None = None,
    alpha: float = 0.80,
) -> QPixmap:
    """(H, W) float32 [0,1] → colormap QPixmap。

    使用 turbo colormap（替代饱和度过高的 jet）。
    若传入 rgb_img，则将热力图与原图灰度以 alpha 权重混合，
    保留结构轮廓、降低视觉刺激感。
    """
    try:
        import matplotlib.cm as cm
        colored = cm.turbo(np.clip(gray, 0, 1))[:, :, :3].astype(np.float32)
    except Exception:
        u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        c  = np.stack([u8, np.zeros_like(u8), (255 - u8)], axis=-1)
        colored = c.astype(np.float32) / 255.0

    if rgb_img is not None:
        # 将原图转为灰度后扩展为 3 通道，以保持解剖结构可见
        gray_bg = np.mean(rgb_img, axis=2, keepdims=True)  # (H, W, 1)
        gray_bg = np.repeat(gray_bg, 3, axis=2)
        colored = alpha * colored + (1.0 - alpha) * gray_bg

    return _numpy_to_qpixmap(np.clip(colored, 0, 1))


def _build_overlay_qpixmap(rgb_img: np.ndarray, gray_cam: np.ndarray) -> QPixmap | None:
    """生成 Grad-CAM 叠加图 QPixmap"""
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        overlay = show_cam_on_image(rgb_img, gray_cam, use_rgb=True)
        return _numpy_to_qpixmap(overlay.astype(np.float32) / 255.0)
    except Exception:
        return None


def _detect_modules_from_ckpt(ckpt_path: str) -> tuple[bool, bool]:
    """扫描 checkpoint 键名，自动判断是否包含 WTConv / EMA 模块。"""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        return False, False
    keys = set(state.keys())
    has_wtconv = any("adapter" in k for k in keys)
    has_ema    = any("attention.ema" in k for k in keys)
    return has_wtconv, has_ema


def _load_model_ext(
    model_name: str,
    ckpt_path: str,
    device: torch.device,
    use_wtconv: bool = False,
    use_ema: bool = False,
) -> nn.Module:
    """扩展版 load_model：支持 use_wtconv / use_ema 参数。"""
    model = get_model(model_name, pretrained=False, use_wtconv=use_wtconv, use_ema=use_ema)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ══════════════════════════════════════════════════════════════
#  后台工作线程
# ══════════════════════════════════════════════════════════════

class ModelLoadWorker(QThread):
    """后台线程：加载模型权重。"""
    finished = Signal(object, object, object, str)   # model, device, transform, name
    error    = Signal(str)

    def __init__(self, model_name, ckpt, device_str, use_wtconv=False, use_ema=False):
        super().__init__()
        self.model_name  = model_name
        self.ckpt        = ckpt
        self.device_str  = device_str
        self.use_wtconv  = use_wtconv
        self.use_ema     = use_ema

    def run(self):
        try:
            dev   = resolve_device(self.device_str)
            model = _load_model_ext(
                self.model_name, self.ckpt, dev,
                self.use_wtconv, self.use_ema,
            )
            trans = get_eval_transform(224)
            self.finished.emit(model, dev, trans, self.model_name)
        except Exception:
            self.error.emit(traceback.format_exc())


class InferenceWorker(QThread):
    """后台线程：推理 + Grad-CAM 生成。"""
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, model, transform, device, img_path,
                 threshold, use_gradcam, target_class, model_name):
        super().__init__()
        self.model        = model
        self.transform    = transform
        self.device       = device
        self.img_path     = img_path
        self.threshold    = threshold
        self.use_gradcam  = use_gradcam
        self.target_class = target_class
        self.model_name   = model_name

    def run(self):
        try:
            inp, logit, prob = infer_single(
                self.model, self.transform, self.img_path, self.device
            )
            pred_label = int(prob >= self.threshold)
            true_label = infer_true_label(self.img_path)

            gray_cam = None
            rgb_img  = denormalize_image(inp)

            if self.use_gradcam:
                tc  = self.target_class
                tgt = (pred_label if tc == "predicted" else
                       (1 if tc == "pneumonia" else 0))
                gray_cam, rgb_img = get_gradcam(
                    self.model, self.model_name, inp, tgt
                )

            self.finished.emit({
                "prob":       prob,
                "pred_label": pred_label,
                "true_label": true_label,
                "rgb_img":    rgb_img,
                "gray_cam":   gray_cam,
                "img_path":   self.img_path,
            })
        except Exception:
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════
#  自定义控件
# ══════════════════════════════════════════════════════════════

class ProbabilityGauge(QWidget):
    """半圆仪表盘 —— 显示 P(PNEUMONIA)，带动画过渡。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value     = 0.0   # 当前显示值（动画中）
        self._target    = 0.0   # 目标值
        self._threshold = 0.5
        self._has_data  = False
        self.setMinimumHeight(175)
        self.setMaximumHeight(200)

        self._timer = QTimer(self)
        self._timer.setInterval(16)   # ~60 fps
        self._timer.timeout.connect(self._tick)

    # ── 公共接口 ──────────────────────────────────────────────

    def set_value(self, v: float, threshold: float = 0.5):
        self._target    = float(v)
        self._threshold = float(threshold)
        self._has_data  = True
        if not self._timer.isActive():
            self._timer.start()

    def reset(self):
        self._target = 0.0
        self._value  = 0.0
        self._has_data = False
        self._timer.stop()
        self.update()

    # ── 内部动画 ──────────────────────────────────────────────

    def _tick(self):
        diff = self._target - self._value
        self._value += diff * 0.14
        if abs(diff) < 5e-4:
            self._value = self._target
            self._timer.stop()
        self.update()

    # ── 绘制 ──────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h  = self.width(), self.height()
        margin = 20
        cx    = w // 2
        cy    = int(h * 0.77)
        r     = min(cx - margin, cy - margin // 2)

        if not self._has_data:
            self._draw_idle(p, cx, cy, r)
            p.end()
            return

        val = max(0.0, min(1.0, self._value))

        # 颜色插值：绿(0) → 黄(0.5) → 红(1)
        if val <= 0.5:
            t = val * 2.0
            r_c = int(166 + (249 - 166) * t)
            g_c = int(227 + (226 - 227) * t)
            b_c = int(161 + (175 - 161) * t)
        else:
            t = (val - 0.5) * 2.0
            r_c = int(249 + (243 - 249) * t)
            g_c = int(226 + (139 - 226) * t)
            b_c = int(175 + (168 - 175) * t)
        val_color = QColor(r_c, g_c, b_c)

        arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)

        # 背景弧
        bg_pen = QPen(QColor("#2B2B44"), 11, Qt.PenStyle.SolidLine)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(bg_pen)
        p.drawArc(arc_rect, 0 * 16, 180 * 16)

        # 数值弧
        span = int(val * 180)
        if span > 0:
            val_pen = QPen(val_color, 11, Qt.PenStyle.SolidLine)
            val_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(val_pen)
            p.drawArc(arc_rect, 180 * 16, -span * 16)

        # 阈值刻度线
        thr_rad = math.radians(180 - self._threshold * 180)
        tx = cx + (r + 4)  * math.cos(thr_rad)
        ty = cy - (r + 4)  * math.sin(thr_rad)
        tx2 = cx + (r - 16) * math.cos(thr_rad)
        ty2 = cy - (r - 16) * math.sin(thr_rad)
        p.setPen(QPen(QColor(C['yellow']), 1.5, Qt.PenStyle.DashLine))
        p.drawLine(int(tx), int(ty), int(tx2), int(ty2))

        # 指针
        needle_rad = math.radians(180 - val * 180)
        nx = cx + (r - 12) * math.cos(needle_rad)
        ny = cy - (r - 12) * math.sin(needle_rad)
        p.setPen(QPen(QColor(C['accent']), 2.5, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(cx, cy, int(nx), int(ny))

        # 中心圆点
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(C['accent'])))
        p.drawEllipse(cx - 5, cy - 5, 10, 10)

        # 端点标签
        p.setFont(QFont("Consolas", 8))
        p.setPen(QPen(QColor(C['muted'])))
        p.drawText(cx - r - 4, cy + 16, "0.0")
        p.drawText(cx + r - 20, cy + 16, "1.0")

        # 概率数值
        prob_txt = f"{val:.3f}"
        f1 = QFont("Consolas", 18, QFont.Weight.Bold)
        p.setFont(f1)
        p.setPen(QPen(val_color))
        fm = QFontMetrics(f1)
        tw = fm.horizontalAdvance(prob_txt)
        p.drawText(cx - tw // 2, cy - r // 3, prob_txt)

        # 预测标签
        label = "PNEUMONIA" if val >= self._threshold else "NORMAL"
        label_col = QColor(C['red']) if val >= self._threshold else QColor(C['green'])
        f2 = QFont("Microsoft YaHei", 9, QFont.Weight.Bold)
        p.setFont(f2)
        p.setPen(QPen(label_col))
        fm2 = QFontMetrics(f2)
        tw2 = fm2.horizontalAdvance(label)
        p.drawText(cx - tw2 // 2, cy + 30, label)

        p.end()

    def _draw_idle(self, p: QPainter, cx: int, cy: int, r: int):
        arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        pen = QPen(QColor("#2B2B44"), 11, Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(arc_rect, 0 * 16, 180 * 16)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(C['border'])))
        p.drawEllipse(cx - 4, cy - 4, 8, 8)

        f = QFont("Microsoft YaHei", 9)
        p.setFont(f)
        p.setPen(QPen(QColor(C['muted'])))
        hint = "等待推理…"
        fm = QFontMetrics(f)
        tw = fm.horizontalAdvance(hint)
        p.drawText(cx - tw // 2, cy + 28, hint)


# ─────────────────────────────────────────────────────────────

class ProbBarChart(pg.PlotWidget):
    """pyqtgraph 概率柱状图：NORMAL / PNEUMONIA 双柱 + 阈值线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(C['panel'])
        self.setMinimumHeight(155)
        self.setMaximumHeight(185)

        pi = self.getPlotItem()
        pi.getAxis("left").setStyle(tickFont=_FONT_MONO)
        pi.getAxis("bottom").setStyle(tickFont=_FONT_MONO)
        pi.getAxis("bottom").setTicks([[(0, "NORMAL"), (1, "PNEUMONIA")]])
        pi.setXRange(-0.5, 1.5, padding=0.1)
        pi.setYRange(0, 1.08, padding=0)
        pi.hideButtons()
        self.showGrid(x=False, y=True, alpha=0.18)
        self.setMouseEnabled(x=False, y=False)

        self._bars = pg.BarGraphItem(
            x=[0, 1],
            height=[0, 0],
            width=0.55,
            brushes=[pg.mkBrush(C["green"]), pg.mkBrush(C["red"])],
        )
        self.addItem(self._bars)

        self._thr_line = pg.InfiniteLine(
            pos=0.5,
            angle=0,
            pen=pg.mkPen(color=C["yellow"], width=1.5,
                         style=Qt.PenStyle.DashLine),
        )
        self.addItem(self._thr_line)

        # 概率文字标注
        self._txt_n = pg.TextItem(text="", color=C["green"], anchor=(0.5, 1.0))
        self._txt_p = pg.TextItem(text="", color=C["red"],   anchor=(0.5, 1.0))
        self._txt_n.setFont(_FONT_MONO)
        self._txt_p.setFont(_FONT_MONO)
        self.addItem(self._txt_n)
        self.addItem(self._txt_p)

    def update_data(self, prob_pneumonia: float, threshold: float = 0.5):
        p_normal = 1.0 - prob_pneumonia
        predicted = prob_pneumonia >= threshold
        self._bars.setOpts(
            height=[p_normal, prob_pneumonia],
            brushes=[
                pg.mkBrush(C["green"] if not predicted else "#454559"),
                pg.mkBrush(C["red"]   if predicted     else "#454559"),
            ],
        )
        self._thr_line.setPos(threshold)
        self._txt_n.setText(f"{p_normal:.3f}")
        self._txt_n.setPos(0, p_normal + 0.02)
        self._txt_p.setText(f"{prob_pneumonia:.3f}")
        self._txt_p.setPos(1, prob_pneumonia + 0.02)

    def reset(self):
        self._bars.setOpts(height=[0, 0])
        self._txt_n.setText("")
        self._txt_p.setText("")


# ─────────────────────────────────────────────────────────────

class InferenceHistoryChart(pg.PlotWidget):
    """pyqtgraph 推理历史折线：记录最近 20 次 P(PNEUMONIA)。"""

    MAX_N = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(C['panel'])
        self.setMinimumHeight(120)
        self.setMaximumHeight(155)

        pi = self.getPlotItem()
        pi.getAxis("left").setStyle(tickFont=_FONT_MONO)
        pi.getAxis("bottom").setStyle(tickFont=_FONT_MONO)
        pi.setYRange(0, 1.05, padding=0)
        pi.hideButtons()
        self.showGrid(x=False, y=True, alpha=0.18)
        self.setMouseEnabled(x=False, y=False)

        self._thr_line = pg.InfiniteLine(
            pos=0.5, angle=0,
            pen=pg.mkPen(C["yellow"], width=1, style=Qt.PenStyle.DashLine),
        )
        self.addItem(self._thr_line)

        self._curve = self.plot(
            pen=pg.mkPen(C["accent"], width=2),
            symbol="o",
            symbolPen=None,
            symbolBrush=C["accent"],
            symbolSize=5,
        )
        self._history: list[float] = []

    def add_point(self, prob: float, threshold: float = 0.5):
        self._history.append(prob)
        if len(self._history) > self.MAX_N:
            self._history.pop(0)

        n = len(self._history)
        brushes = [
            pg.mkBrush(C["red"] if p >= threshold else C["green"])
            for p in self._history
        ]
        self._curve.setData(
            x=list(range(n)),
            y=self._history,
            symbolBrush=brushes,
        )
        self._thr_line.setPos(threshold)
        if n > 1:
            self.getPlotItem().setXRange(0, n - 1, padding=0.1)

    def clear_history(self):
        self._history.clear()
        self._curve.setData([], [])


# ─────────────────────────────────────────────────────────────

class ImageViewer(QGraphicsView):
    """QGraphicsView 图像查看器：滚轮缩放 / 拖拽平移 / 适配窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"border: none; background: {C['entry']};")

        self._item: QGraphicsPixmapItem | None = None
        self._draw_hint()

    # ── 公共接口 ──────────────────────────────────────────────

    def set_pixmap(self, pix: QPixmap):
        self._scene.clear()
        self._item = QGraphicsPixmapItem(pix)
        self._item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self._scene.addItem(self._item)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fit_view()

    def fit_view(self):
        if self._item:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_reset(self):
        self.resetTransform()
        if self._item:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear(self):
        self._scene.clear()
        self._item = None
        self._draw_hint()

    # ── 事件 ──────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        if self._item is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._item:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    # ── 内部 ──────────────────────────────────────────────────

    def _draw_hint(self):
        self._scene.setSceneRect(0, 0, 600, 420)
        lines = [
            ("请加载模型 → 选择图像 → 运行推理", _FONT_BOLD,   C['fg2'], 162),
            ("运行后将在此显示原图与 Grad-CAM 可视化",  _FONT_SMALL, C['muted'], 196),
        ]
        for text, font, color, y in lines:
            ti = self._scene.addText(text, font)
            ti.setDefaultTextColor(QColor(color))
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(text)
            ti.setPos(300 - tw / 2, y)


# ══════════════════════════════════════════════════════════════
#  主应用窗口
# ══════════════════════════════════════════════════════════════

class PneumoniaApp(QMainWindow):

    _SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("肺炎检测系统  ·  Chest X-Ray Pneumonia Detection")
        self.resize(1450, 860)
        self.setMinimumSize(1100, 680)

        # ── 运行时状态 ────────────────────────────────────────
        self._model:      nn.Module | None = None
        self._device:     torch.device | None = None
        self._transform   = None
        self._model_name: str = ""
        self._busy:       bool = False
        self._img_path:   str  = ""
        self._spinner_idx: int  = 0

        # ── 三视图图像缓存 ─────────────────────────────────────
        self._pix_original: QPixmap | None = None
        self._pix_heatmap:  QPixmap | None = None
        self._pix_overlay:  QPixmap | None = None
        self._last_result:  dict | None = None

        # ── UI ───────────────────────────────────────────────
        self._build_ui()
        self._refresh_states()

        # 旋转动画计时器
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(80)
        self._spin_timer.timeout.connect(self._tick_spinner)

    # ══════════════════════════════════════════════════════════
    #  UI 构建
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部标题栏
        main_layout.addWidget(self._build_header())

        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([290, 860, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        main_layout.addWidget(splitter, 1)

        # 状态栏
        self._build_statusbar()

    # ── 顶部标题栏 ────────────────────────────────────────────

    def _build_header(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(52)
        bar.setStyleSheet(
            f"background: {C['surface']};"
            f"border-bottom: 1px solid {C['border']};"
        )
        lo = QHBoxLayout(bar)
        lo.setContentsMargins(18, 0, 18, 0)
        lo.setSpacing(10)

        lbl_icon = QLabel("🫁")
        lbl_icon.setFont(QFont("Segoe UI Emoji", 18))
        lbl_icon.setStyleSheet("background: transparent;")

        lbl_title = QLabel("肺炎检测系统")
        lbl_title.setFont(_FONT_TITLE)
        lbl_title.setStyleSheet(f"color: {C['fg']}; background: transparent;")

        lbl_sub = QLabel("Chest X-Ray Pneumonia Detection  |  PySide6 + pyqtgraph")
        lbl_sub.setFont(_FONT_MONO)
        lbl_sub.setStyleSheet(f"color: {C['muted']}; background: transparent;")

        self._model_status_lbl = QLabel("●  未加载模型")
        self._model_status_lbl.setFont(_FONT_MONO)
        self._model_status_lbl.setStyleSheet(
            f"color: {C['muted']}; background: transparent;"
        )

        self._spinner_lbl = QLabel("")
        self._spinner_lbl.setFont(QFont("Consolas", 13))
        self._spinner_lbl.setFixedWidth(20)
        self._spinner_lbl.setStyleSheet(
            f"color: {C['accent']}; background: transparent;"
        )

        lo.addWidget(lbl_icon)
        lo.addWidget(lbl_title)
        lo.addWidget(lbl_sub)
        lo.addStretch()
        lo.addWidget(self._model_status_lbl)
        lo.addWidget(self._spinner_lbl)
        return bar

    # ── 左侧控制面板 ──────────────────────────────────────────

    def _build_left(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(310)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner.setStyleSheet(f"background: {C['panel']};")
        lo = QVBoxLayout(inner)
        lo.setContentsMargins(10, 12, 10, 12)
        lo.setSpacing(8)

        lo.addWidget(self._build_model_group())
        lo.addWidget(self._build_infer_group())
        lo.addWidget(self._build_action_group())

        tip = QLabel("💡 Ctrl+滚轮 缩放图像  |  拖拽 平移")
        tip.setFont(_FONT_SMALL)
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {C['muted']}; background: transparent; padding: 4px;")
        lo.addWidget(tip)
        lo.addStretch()

        scroll.setWidget(inner)
        return scroll

    def _build_model_group(self) -> QGroupBox:
        grp = QGroupBox("模型配置")
        lo = QVBoxLayout(grp)
        lo.setSpacing(7)

        # 架构
        lo.addWidget(self._lbl("网络架构"))
        self._arch_combo = QComboBox()
        self._arch_combo.addItems(["densenet121", "resnet50", "efficientnet_b0"])
        lo.addWidget(self._arch_combo)

        # Checkpoint
        lo.addWidget(self._lbl("Checkpoint (.pth)"))
        ckpt_row = QHBoxLayout()
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("选择 .pth 文件…")
        self._ckpt_edit.setReadOnly(True)
        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(52)
        browse_btn.clicked.connect(self._browse_ckpt)
        ckpt_row.addWidget(self._ckpt_edit)
        ckpt_row.addWidget(browse_btn)
        lo.addLayout(ckpt_row)

        # 设备
        lo.addWidget(self._lbl("推理设备"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda"])
        lo.addWidget(self._device_combo)

        # 加载按钮
        self._load_btn = QPushButton("加载模型")
        self._load_btn.setStyleSheet(_SS_ACCENT)
        self._load_btn.clicked.connect(self._on_load_model)
        lo.addWidget(self._load_btn)

        return grp

    def _build_infer_group(self) -> QGroupBox:
        grp = QGroupBox("推理配置")
        lo = QVBoxLayout(grp)
        lo.setSpacing(7)

        # 阈值
        thr_row = QHBoxLayout()
        thr_row.addWidget(self._lbl("决策阈值"))
        self._thr_val_lbl = QLabel("0.50")
        self._thr_val_lbl.setFont(_FONT_MONO)
        self._thr_val_lbl.setStyleSheet(
            f"color: {C['accent']}; background: transparent;"
        )
        thr_row.addStretch()
        thr_row.addWidget(self._thr_val_lbl)
        lo.addLayout(thr_row)

        self._thr_slider = QSlider(Qt.Orientation.Horizontal)
        self._thr_slider.setRange(5, 95)
        self._thr_slider.setValue(50)
        self._thr_slider.setTickInterval(10)
        self._thr_slider.valueChanged.connect(self._on_threshold_changed)
        lo.addWidget(self._thr_slider)

        # Grad-CAM
        self._gradcam_cb = QCheckBox("启用 Grad-CAM 热力图")
        self._gradcam_cb.setChecked(True)
        self._gradcam_cb.toggled.connect(self._on_gradcam_toggled)
        lo.addWidget(self._gradcam_cb)

        self._target_lbl = self._lbl("目标类别")
        lo.addWidget(self._target_lbl)
        self._target_combo = QComboBox()
        self._target_combo.addItems(["predicted", "pneumonia", "normal"])
        lo.addWidget(self._target_combo)

        return grp

    def _build_action_group(self) -> QGroupBox:
        grp = QGroupBox("操作")
        lo = QVBoxLayout(grp)
        lo.setSpacing(7)

        self._select_btn = QPushButton("选择 X-Ray 图像…")
        self._select_btn.clicked.connect(self._on_browse_image)
        lo.addWidget(self._select_btn)

        self._img_lbl = QLabel("尚未选择图像")
        self._img_lbl.setFont(_FONT_MONO)
        self._img_lbl.setWordWrap(True)
        self._img_lbl.setStyleSheet(
            f"color: {C['muted']}; background: transparent;"
        )
        lo.addWidget(self._img_lbl)

        self._run_btn = QPushButton("▶  运行推理")
        self._run_btn.setStyleSheet(_SS_ACCENT)
        self._run_btn.clicked.connect(self._on_run_inference)
        lo.addWidget(self._run_btn)

        self._save_btn = QPushButton("💾  保存当前视图")
        self._save_btn.setStyleSheet(_SS_GREEN)
        self._save_btn.clicked.connect(self._on_save)
        lo.addWidget(self._save_btn)

        return grp

    # ── 中间图像区 ────────────────────────────────────────────

    def _build_center(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {C['bg']};")
        lo = QVBoxLayout(w)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)

        # 先创建 viewer，再构建工具栏（工具栏内引用 self._viewer）
        self._viewer = ImageViewer()

        lo.addWidget(self._build_view_toolbar())
        lo.addWidget(self._viewer, 1)
        lo.addWidget(self._build_result_bar())
        return w

    def _build_view_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(40)
        bar.setStyleSheet(
            f"background: {C['surface']};"
            f"border-bottom: 1px solid {C['border']};"
        )
        lo = QHBoxLayout(bar)
        lo.setContentsMargins(12, 0, 12, 0)
        lo.setSpacing(6)

        lbl = QLabel("视图:")
        lbl.setFont(_FONT_SMALL)
        lbl.setStyleSheet(f"color: {C['muted']}; background: transparent;")
        lo.addWidget(lbl)

        self._btn_orig = self._toggle_btn("原始图像", True)
        self._btn_heat = self._toggle_btn("热力图", False)
        self._btn_over = self._toggle_btn("叠加视图", False)

        self._btn_orig.clicked.connect(lambda: self._switch_view(0))
        self._btn_heat.clicked.connect(lambda: self._switch_view(1))
        self._btn_over.clicked.connect(lambda: self._switch_view(2))

        lo.addWidget(self._btn_orig)
        lo.addWidget(self._btn_heat)
        lo.addWidget(self._btn_over)
        lo.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {C['border']};")
        lo.addWidget(sep)

        fit_btn = QPushButton("适配窗口")
        fit_btn.clicked.connect(self._viewer.fit_view)
        fit_btn.setFixedHeight(26)
        reset_btn = QPushButton("1:1")
        reset_btn.clicked.connect(self._viewer.zoom_reset)
        reset_btn.setFixedWidth(42)
        reset_btn.setFixedHeight(26)
        lo.addWidget(fit_btn)
        lo.addWidget(reset_btn)

        self._view_mode = 0   # 0=orig 1=heat 2=overlay
        return bar

    def _build_result_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(50)
        bar.setStyleSheet(
            f"background: {C['surface']};"
            f"border-top: 1px solid {C['border']};"
        )
        lo = QHBoxLayout(bar)
        lo.setContentsMargins(14, 0, 14, 0)
        lo.setSpacing(12)

        self._result_badge = QLabel("WAITING")
        self._result_badge.setFont(_FONT_MONO)
        self._result_badge.setStyleSheet(
            f"color: {C['muted']}; background: {C['card']};"
            f"border-radius: 3px; padding: 3px 8px;"
        )

        self._result_lbl = QLabel("尚未运行推理")
        self._result_lbl.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self._result_lbl.setStyleSheet(f"color: {C['fg2']}; background: transparent;")

        lo.addWidget(self._result_badge)
        lo.addWidget(self._result_lbl, 1)
        return bar

    # ── 右侧分析面板 ──────────────────────────────────────────

    def _build_right(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(270)
        w.setMaximumWidth(330)
        w.setStyleSheet(
            f"background: {C['panel']};"
            f"border-left: 1px solid {C['border']};"
        )

        lo = QVBoxLayout(w)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setSpacing(6)

        lo.addWidget(self._sec_lbl("概率仪表"))
        self._gauge = ProbabilityGauge()
        lo.addWidget(self._gauge)

        lo.addWidget(self._divider())
        lo.addWidget(self._sec_lbl("概率分布"))
        self._bar_chart = ProbBarChart()
        lo.addWidget(self._bar_chart)

        lo.addWidget(self._divider())
        lo.addWidget(self._sec_lbl("推理历史"))
        self._history_chart = InferenceHistoryChart()
        lo.addWidget(self._history_chart)

        lo.addStretch()
        return w

    # ── 状态栏 ────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()

        self._status_badge = QLabel("READY")
        self._status_badge.setFont(_FONT_MONO)
        self._status_badge.setStyleSheet(
            f"color: {C['green']}; background: {C['card']};"
            f"border-radius: 3px; padding: 2px 8px; margin: 2px 4px;"
        )
        sb.addWidget(self._status_badge)

        self._status_lbl = QLabel("就绪")
        self._status_lbl.setFont(_FONT_MONO)
        sb.addWidget(self._status_lbl, 1)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(120)
        self._progress.setRange(0, 0)
        self._progress.hide()
        sb.addPermanentWidget(self._progress)

    # ── 小组件工厂 ────────────────────────────────────────────

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(_FONT_SMALL)
        lbl.setStyleSheet(f"color: {C['fg2']}; background: transparent;")
        return lbl

    @staticmethod
    def _sec_lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {C['fg']}; background: transparent;")
        return lbl

    @staticmethod
    def _divider() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(f"background: {C['border']}; margin: 2px 0;")
        return f

    @staticmethod
    def _toggle_btn(text: str, checked: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setFixedHeight(26)
        btn.setStyleSheet(_SS_TOGGLE_ON if checked else _SS_TOGGLE_OFF)
        return btn

    # ══════════════════════════════════════════════════════════
    #  事件处理
    # ══════════════════════════════════════════════════════════

    def _browse_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Checkpoint",
            filter="PyTorch 模型 (*.pth *.pt);;所有文件 (*.*)",
        )
        if path:
            self._ckpt_edit.setText(path)

    def _on_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 X-Ray 图像",
            filter="图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;所有文件 (*.*)",
        )
        if not path:
            return

        self._img_path = path
        self._img_lbl.setText(os.path.basename(path))
        self._img_lbl.setStyleSheet(f"color: {C['fg2']}; background: transparent;")

        # 清除旧结果
        self._pix_original = self._pix_heatmap = self._pix_overlay = None
        self._last_result  = None
        self._result_badge.setText("SELECTED")
        self._result_badge.setStyleSheet(
            f"color: {C['cyan']}; background: {C['card']};"
            f"border-radius: 3px; padding: 3px 8px;"
        )
        self._result_lbl.setText("已选择图像，点击「运行推理」开始分析")
        self._result_lbl.setStyleSheet(f"color: {C['fg2']}; background: transparent;")

        # 预览原图
        try:
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            arr = np.array(pil).astype(np.float32) / 255.0
            self._pix_original = _numpy_to_qpixmap(arr)
            self._view_mode = 0
            self._update_view_toggle_state()
            self._viewer.set_pixmap(self._pix_original)
        except Exception as e:
            self._set_status(f"[Error] 加载图像失败: {e}", "error")

        self._set_status(f"已选择: {os.path.basename(path)}")
        self._refresh_states()

    def _on_threshold_changed(self, v: int):
        thr = v / 100.0
        self._thr_val_lbl.setText(f"{thr:.2f}")
        # 若已有结果，实时更新图表
        if self._last_result:
            prob = self._last_result["prob"]
            self._gauge.set_value(prob, thr)
            self._bar_chart.update_data(prob, thr)

    def _on_gradcam_toggled(self, checked: bool):
        self._target_lbl.setVisible(checked)
        self._target_combo.setVisible(checked)

    # ── 模型加载 ──────────────────────────────────────────────

    def _on_load_model(self):
        if self._busy:
            return
        ckpt = self._ckpt_edit.text().strip()
        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(self, "提示", "请先选择有效的 Checkpoint 文件")
            return

        # 自动检测 checkpoint 包含的模块
        use_wtconv, use_ema = False, False
        try:
            use_wtconv, use_ema = _detect_modules_from_ckpt(ckpt)
            mods = []
            if use_wtconv: mods.append("WTConv")
            if use_ema:    mods.append("EMA")
            if mods:
                self._set_status(f"检测到模块: {', '.join(mods)}", "warn")
        except Exception:
            pass  # 检测失败时静默忽略，加载失败会给出完整错误

        self._set_busy(True, "加载模型中…", "LOADING")
        worker = ModelLoadWorker(
            model_name  = self._arch_combo.currentText(),
            ckpt        = ckpt,
            device_str  = self._device_combo.currentText(),
            use_wtconv  = use_wtconv,
            use_ema     = use_ema,
        )
        worker.finished.connect(self._on_model_loaded)
        worker.error.connect(self._on_model_error)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._worker = worker   # hold reference
        worker.start()

    def _on_model_loaded(self, model, device, transform, name):
        self._model      = model
        self._device     = device
        self._transform  = transform
        self._model_name = name
        self._model_status_lbl.setText(f"●  {name}  on  {device}")
        self._model_status_lbl.setStyleSheet(
            f"color: {C['green']}; background: transparent;"
        )
        self._set_busy(False, f"模型已就绪: {name} → {device}", "READY", "ok")
        self._refresh_states()

    def _on_model_error(self, tb: str):
        self._model_status_lbl.setText("●  加载失败")
        self._model_status_lbl.setStyleSheet(
            f"color: {C['red']}; background: transparent;"
        )
        self._set_busy(False, "模型加载失败", "ERROR", "error")
        self._refresh_states()
        QMessageBox.critical(self, "加载失败", tb)

    # ── 推理 ──────────────────────────────────────────────────

    def _on_run_inference(self):
        if self._busy or self._model is None:
            return
        if not self._img_path or not os.path.isfile(self._img_path):
            QMessageBox.warning(self, "提示", "请先选择有效的图像文件")
            return

        thr = self._thr_slider.value() / 100.0
        self._set_busy(True, "推理中…", "RUNNING")
        self._result_badge.setText("RUNNING")
        self._result_badge.setStyleSheet(
            f"color: {C['yellow']}; background: {C['card']};"
            f"border-radius: 3px; padding: 3px 8px;"
        )
        self._result_lbl.setText("正在分析图像，请稍候…")

        worker = InferenceWorker(
            model        = self._model,
            transform    = self._transform,
            device       = self._device,
            img_path     = self._img_path,
            threshold    = thr,
            use_gradcam  = self._gradcam_cb.isChecked(),
            target_class = self._target_combo.currentText(),
            model_name   = self._model_name,
        )
        worker.finished.connect(self._on_inference_done)
        worker.error.connect(self._on_inference_error)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._worker2 = worker
        worker.start()

    def _on_inference_done(self, result: dict):
        self._last_result = result
        prob       = result["prob"]
        pred_label = result["pred_label"]
        true_label = result["true_label"]
        rgb_img    = result["rgb_img"]
        gray_cam   = result["gray_cam"]
        img_path   = result["img_path"]
        thr        = self._thr_slider.value() / 100.0

        # 构建三视图 QPixmap
        self._pix_original = _numpy_to_qpixmap(rgb_img)
        self._pix_heatmap  = (
            _gray_to_jet_qpixmap(gray_cam, rgb_img=rgb_img)
            if gray_cam is not None else None
        )
        self._pix_overlay  = (
            _build_overlay_qpixmap(rgb_img, gray_cam)
            if gray_cam is not None else None
        )

        # 显示视图（优先叠加图，其次原图）
        if self._pix_overlay is not None:
            self._view_mode = 2
        elif self._pix_heatmap is not None:
            self._view_mode = 1
        else:
            self._view_mode = 0
        self._update_view_toggle_state()
        self._show_current_view()

        # 更新右侧图表（动画）
        self._gauge.set_value(prob, thr)
        self._bar_chart.update_data(prob, thr)
        self._history_chart.add_point(prob, thr)

        # 结果文字
        pred_str  = LABEL_NAME[pred_label]
        conf_pct  = max(prob, 1 - prob) * 100
        res_txt   = (
            f"预测: {pred_str}    "
            f"P(PNEUMONIA) = {prob:.4f}    "
            f"置信度: {conf_pct:.1f}%"
        )
        if true_label is not None:
            correct = pred_label == true_label
            res_txt += ("    ✓ 正确" if correct else "    ✗ 错误")

        res_color = C['red'] if pred_label == 1 else C['green']
        badge_txt = "PNEUMONIA" if pred_label == 1 else "NORMAL"

        self._result_badge.setText(badge_txt)
        self._result_badge.setStyleSheet(
            f"color: {res_color}; background: {C['card']};"
            f"border-radius: 3px; padding: 3px 8px;"
        )
        self._result_lbl.setText(res_txt)
        self._result_lbl.setStyleSheet(
            f"color: {res_color}; background: transparent;"
        )

        self._set_busy(
            False,
            f"完成  |  {os.path.basename(img_path)}  →  {pred_str}  (P = {prob:.4f})",
            "READY", "ok",
        )
        self._refresh_states()

    def _on_inference_error(self, tb: str):
        self._set_busy(False, "推理失败", "ERROR", "error")
        self._result_badge.setText("ERROR")
        self._result_badge.setStyleSheet(
            f"color: {C['red']}; background: {C['card']};"
            f"border-radius: 3px; padding: 3px 8px;"
        )
        self._result_lbl.setText("推理失败，请查看错误详情")
        self._result_lbl.setStyleSheet(
            f"color: {C['red']}; background: transparent;"
        )
        self._refresh_states()
        QMessageBox.critical(self, "推理失败", tb)

    # ── 视图切换 ──────────────────────────────────────────────

    def _switch_view(self, mode: int):
        self._view_mode = mode
        self._update_view_toggle_state()
        self._show_current_view()

    def _show_current_view(self):
        pix_map = [self._pix_original, self._pix_heatmap, self._pix_overlay]
        pix = pix_map[self._view_mode]
        if pix is None:
            pix = self._pix_original
        if pix is not None:
            self._viewer.set_pixmap(pix)

    def _update_view_toggle_state(self):
        has_cam = self._pix_heatmap is not None
        for i, btn in enumerate([self._btn_orig, self._btn_heat, self._btn_over]):
            active = (i == self._view_mode)
            enabled = (i == 0) or has_cam
            btn.setChecked(active)
            btn.setEnabled(enabled)
            if not enabled:
                btn.setStyleSheet(_SS_TOGGLE_DIS)
            elif active:
                btn.setStyleSheet(_SS_TOGGLE_ON)
            else:
                btn.setStyleSheet(_SS_TOGGLE_OFF)

    # ── 保存 ──────────────────────────────────────────────────

    def _on_save(self):
        if self._last_result is None:
            QMessageBox.information(self, "提示", "尚无推理结果，请先运行推理")
            return

        pix_map = [self._pix_original, self._pix_heatmap, self._pix_overlay]
        pix = pix_map[self._view_mode]
        if pix is None:
            pix = self._pix_original
        if pix is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前视图",
            "result.png",
            "PNG 图像 (*.png);;JPEG (*.jpg)",
        )
        if path:
            pix.save(path)
            self._set_status(f"已保存: {path}", "ok")

    # ── 状态辅助 ──────────────────────────────────────────────

    def _set_status(self, msg: str, level: str = "ok", badge: str | None = None):
        prefix = "✓  " if level == "ok" else "⚠  " if level == "warn" else "✕  "
        self._status_lbl.setText(prefix + msg)
        if badge:
            self._status_badge.setText(badge)
        color = (C['green'] if level == "ok"
                 else C['yellow'] if level == "warn"
                 else C['red'])
        if badge:
            self._status_badge.setStyleSheet(
                f"color: {color}; background: {C['card']};"
                f"border-radius: 3px; padding: 2px 8px; margin: 2px 4px;"
            )

    def _set_busy(self, busy: bool, msg: str = "", badge: str = "", level: str = "ok"):
        self._busy = busy
        if busy:
            self._progress.show()
            self._spin_timer.start()
            self._status_lbl.setText("⏳  " + msg)
            self._status_badge.setText(badge)
            self._status_badge.setStyleSheet(
                f"color: {C['yellow']}; background: {C['card']};"
                f"border-radius: 3px; padding: 2px 8px; margin: 2px 4px;"
            )
        else:
            self._progress.hide()
            self._spin_timer.stop()
            self._spinner_lbl.setText("")
            self._set_status(msg, level, badge)
        self._refresh_states()

    def _refresh_states(self):
        has_model  = self._model is not None
        has_image  = bool(self._img_path) and os.path.isfile(self._img_path)
        has_result = self._last_result is not None

        self._load_btn.setEnabled(not self._busy)
        self._select_btn.setEnabled(not self._busy)
        self._run_btn.setEnabled(has_model and has_image and not self._busy)
        self._save_btn.setEnabled(has_result and not self._busy)

        if self._busy:
            self._run_btn.setText("处理中…")
        else:
            self._run_btn.setText("▶  运行推理")

    def _tick_spinner(self):
        self._spinner_lbl.setText(self._SPINNER[self._spinner_idx])
        self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER)


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(_QSS)

    win = PneumoniaApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
