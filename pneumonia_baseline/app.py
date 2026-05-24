#!/usr/bin/env python
"""
app_gui_optimized.py — 胸部 X-Ray 肺炎检测  交互式桌面图形界面（GUI 优化版）

运行：
    python app_gui_optimized.py

说明：
    - 不改变原有推理逻辑与核心功能区：模型加载、图像选择、阈值、Grad-CAM、运行推理、保存结果均保持一致。
    - 主要优化 GUI 观感、分区层次、状态反馈、图像预览体验和窗口自适应。
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test import (
    LABEL_NAME,
    denormalize_image,
    get_eval_transform,
    get_gradcam,
    infer_single,
    infer_true_label,
    load_model,
    resolve_device,
    visualize_and_save,
)


# ══════════════════════════════════════════════════════════════
#  主题色（深色医学影像风）
# ══════════════════════════════════════════════════════════════

_BG          = "#141421"   # 主背景
_SURFACE     = "#1B1B2B"   # 内容背景
_PANEL       = "#202034"   # 左侧面板
_CARD        = "#272742"   # 卡片
_CARD2       = "#222238"   # 次级卡片
_ENTRY       = "#11111D"   # 输入框 / 下拉框
_BORDER      = "#343454"   # 边框 / 分割线

_FG          = "#EEF2FF"   # 主文字
_FG2         = "#B8C0E0"   # 次文字
_MUTED       = "#7F849C"   # 弱文字

_ACCENT      = "#CBA6F7"   # 紫色强调
_ACCENT2     = "#89B4FA"   # 蓝色强调
_CYAN        = "#89DCEB"   # 青色
_GREEN       = "#A6E3A1"   # NORMAL / 正确 / 保存
_RED         = "#F38BA8"   # PNEUMONIA / 错误
_YELLOW      = "#F9E2AF"   # 提示

_STATUS_BG   = "#0F0F1A"
_CANVAS_BG   = "#10101B"

_FONT_CN     = ("Microsoft YaHei", 9)
_FONT_CN_B   = ("Microsoft YaHei", 9, "bold")
_FONT_TITLE  = ("Microsoft YaHei", 16, "bold")
_FONT_SUB    = ("Microsoft YaHei", 8)
_FONT_MONO   = ("Consolas", 9)
_FONT_MONO_S = ("Consolas", 8)


# ══════════════════════════════════════════════════════════════
#  应用主类
# ══════════════════════════════════════════════════════════════

class PneumoniaApp:
    """肺炎检测 GUI。

    核心推理逻辑仍复用 test.py：
    load_model / infer_single / get_gradcam / visualize_and_save
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("胸部 X-Ray 肺炎检测系统")
        self.root.geometry("1380x820")
        self.root.minsize(1080, 680)
        self.root.configure(bg=_BG)

        # ── 运行时状态 ────────────────────────────────────────
        self._model      = None
        self._device     = None
        self._transform  = None
        self._model_name = None

        self._last_img    = None     # 最近一次推理结果 PIL Image（用于保存）
        self._current_pil = None     # 当前 Canvas 显示的 PIL Image（用于缩放/重绘）
        self._photo_ref   = None     # 防止 PhotoImage 被 GC

        self._zoom = 1.0
        self._resize_after_id = None
        self._busy = False

        # ── Tk 变量 ───────────────────────────────────────────
        self.v_model      = tk.StringVar(value="densenet121")
        self.v_ckpt       = tk.StringVar(value=r"results_split\checkpoints\densenet121_best.pth")
        self.v_device     = tk.StringVar(value="auto")
        self.v_threshold  = tk.DoubleVar(value=0.5)
        self.v_gradcam    = tk.BooleanVar(value=True)
        self.v_target     = tk.StringVar(value="predicted")
        self.v_img_path   = tk.StringVar(value="")
        self.v_status     = tk.StringVar(value="就绪  |  请先加载模型，再选择图像")
        self.v_zoom_text  = tk.StringVar(value="100%")

        self._build_ui()
        self._refresh_action_states()

    # ──────────────────────────────────────────────────────────
    #  UI 构建
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        shell = tk.Frame(self.root, bg=_BG)
        shell.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # 左侧控制面板（固定宽度）
        left = tk.Frame(shell, bg=_PANEL, width=318)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        self._build_left(left)

        # 右侧结果区域
        right = tk.Frame(shell, bg=_BG)
        right.pack(side="right", fill="both", expand=True)
        self._build_right(right)

        # 底部状态栏
        status = tk.Frame(self.root, bg=_STATUS_BG, height=30)
        status.pack(side="bottom", fill="x")
        status.pack_propagate(False)

        self._status_badge = tk.Label(
            status,
            text="READY",
            bg=_CARD,
            fg=_GREEN,
            font=("Consolas", 8, "bold"),
            padx=8,
            pady=2,
        )
        self._status_badge.pack(side="left", padx=(12, 8), pady=5)

        tk.Label(
            status,
            textvariable=self.v_status,
            bg=_STATUS_BG,
            fg=_FG2,
            anchor="w",
            font=_FONT_MONO_S,
        ).pack(side="left", fill="x", expand=True)

    # ── 左侧面板 ───────────────────────────────────────────────

    def _build_left(self, parent):
        # 标题区
        header = tk.Frame(parent, bg=_PANEL)
        header.pack(fill="x", padx=16, pady=(18, 12))

        tk.Label(
            header,
            text="肺炎检测系统",
            bg=_PANEL,
            fg=_FG,
            font=_FONT_TITLE,
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            header,
            text="Chest X-Ray Pneumonia Detection",
            bg=_PANEL,
            fg=_FG2,
            font=_FONT_MONO_S,
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        self._tip_label = tk.Label(
            header,
            text="流程：加载模型 → 选择图像 → 运行推理",
            bg=_PANEL,
            fg=_MUTED,
            font=_FONT_SUB,
            anchor="w",
        )
        self._tip_label.pack(fill="x", pady=(7, 0))

        # 模型设置
        model_card = self._card(parent)
        self._sec(model_card, "模型设置", "Model")

        self._lbl(model_card, "模型架构").pack(**self._pack_line())
        self._ddl(model_card, self.v_model, "densenet121", "resnet50", "efficientnet_b0").pack(**self._pack_line())

        self._lbl(model_card, "Checkpoint 路径").pack(**self._pack_line())
        row = tk.Frame(model_card, bg=_CARD)
        row.pack(**self._pack_line())

        tk.Entry(
            row,
            textvariable=self.v_ckpt,
            bg=_ENTRY,
            fg=_FG,
            insertbackground=_FG,
            relief="flat",
            font=_FONT_MONO_S,
            highlightthickness=1,
            highlightbackground=_BORDER,
            highlightcolor=_ACCENT2,
        ).pack(side="left", fill="x", expand=True, ipady=5)

        self._btn(row, "…", self._browse_ckpt, _CARD2, width=3).pack(side="right", padx=(6, 0))

        self._lbl(model_card, "推理设备").pack(**self._pack_line())
        self._ddl(model_card, self.v_device, "auto", "cuda", "cpu").pack(**self._pack_line())

        self._load_btn = self._btn(
            model_card,
            "⚙  加载模型",
            self._on_load_model,
            _CYAN,
            fg="#10101A",
        )
        self._load_btn.pack(**self._pack_line(pady=(10, 3)))

        self._model_dot = tk.Label(
            model_card,
            text="●  未加载",
            bg=_CARD,
            fg=_RED,
            font=_FONT_MONO_S,
            anchor="w",
        )
        self._model_dot.pack(fill="x", padx=12, pady=(3, 6))

        # 推理设置
        infer_card = self._card(parent)
        self._sec(infer_card, "推理设置", "Inference")

        thr_row = tk.Frame(infer_card, bg=_CARD)
        thr_row.pack(**self._pack_line())
        self._lbl(thr_row, "决策阈值", bg=_CARD).pack(side="left")
        self._thr_val = tk.Label(
            thr_row,
            text="0.50",
            bg=_CARD,
            fg=_ACCENT,
            font=("Consolas", 10, "bold"),
        )
        self._thr_val.pack(side="right")

        tk.Scale(
            infer_card,
            variable=self.v_threshold,
            from_=0.05,
            to=0.95,
            resolution=0.01,
            orient="horizontal",
            showvalue=False,
            bg=_CARD,
            fg=_FG,
            troughcolor=_ENTRY,
            activebackground=_ACCENT,
            highlightthickness=0,
            bd=0,
            command=lambda v: self._thr_val.configure(text=f"{float(v):.2f}"),
        ).pack(**self._pack_line())

        # Grad-CAM
        grad_card = self._card(parent)
        self._sec(grad_card, "Grad-CAM 设置", "Explainability")

        tk.Checkbutton(
            grad_card,
            text="  启用 Grad-CAM 热力图",
            variable=self.v_gradcam,
            bg=_CARD,
            fg=_FG,
            selectcolor=_ENTRY,
            activebackground=_CARD,
            activeforeground=_FG,
            font=_FONT_CN,
        ).pack(**self._pack_line(pady=(5, 3)))

        self._lbl(grad_card, "目标类别").pack(**self._pack_line())
        self._ddl(grad_card, self.v_target, "predicted", "pneumonia", "normal").pack(**self._pack_line())

        # 图像操作
        action_card = self._card(parent)
        self._sec(action_card, "图像操作", "Actions")

        self._select_btn = self._btn(action_card, "📂  选择图像", self._on_browse_image, "#45475A")
        self._select_btn.pack(**self._pack_line(pady=(7, 4)))

        img_box = tk.Frame(action_card, bg=_CARD2)
        img_box.pack(fill="x", padx=12, pady=(2, 7))

        tk.Label(
            img_box,
            text="当前图像",
            bg=_CARD2,
            fg=_MUTED,
            font=_FONT_SUB,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(6, 1))

        self._img_name = tk.Label(
            img_box,
            text="未选择",
            bg=_CARD2,
            fg=_FG2,
            font=_FONT_MONO_S,
            wraplength=250,
            justify="left",
            anchor="w",
        )
        self._img_name.pack(fill="x", padx=8, pady=(0, 7))

        self._run_btn = self._btn(action_card, "▶  运行推理", self._on_run_inference, _ACCENT2, fg="#10101A")
        self._run_btn.pack(**self._pack_line(pady=(3, 4)))

        self._save_btn = self._btn(action_card, "💾  保存结果", self._on_save, _GREEN, fg="#10101A")
        self._save_btn.pack(**self._pack_line(pady=(0, 10)))

        # 小提示
        tk.Label(
            parent,
            text="提示：右侧图像支持鼠标滚轮缩放、拖拽平移。",
            bg=_PANEL,
            fg=_MUTED,
            font=_FONT_SUB,
            wraplength=282,
            justify="left",
        ).pack(side="bottom", fill="x", padx=16, pady=(8, 14))

    # ── 右侧面板 ───────────────────────────────────────────────

    def _build_right(self, parent):
        # 顶部标题栏
        top = tk.Frame(parent, bg=_SURFACE, height=62)
        top.pack(fill="x", pady=(0, 8))
        top.pack_propagate(False)

        tk.Label(
            top,
            text="检测结果预览",
            bg=_SURFACE,
            fg=_FG,
            font=("Microsoft YaHei", 14, "bold"),
            anchor="w",
        ).pack(side="left", padx=18, fill="y")

        toolbar = tk.Frame(top, bg=_SURFACE)
        toolbar.pack(side="right", padx=12)

        self._zoom_label = tk.Label(
            toolbar,
            textvariable=self.v_zoom_text,
            bg=_CARD,
            fg=_FG2,
            font=_FONT_MONO_S,
            padx=10,
            pady=5,
        )
        self._zoom_label.pack(side="left", padx=(0, 8))

        self._btn(toolbar, "适配窗口", self._fit_to_window, _CARD, fg=_FG2).pack(side="left", padx=(0, 6))
        self._btn(toolbar, "重置缩放", self._reset_zoom, _CARD, fg=_FG2).pack(side="left")

        # 图像显示 Canvas（带滚动条）
        canvas_outer = tk.Frame(parent, bg=_BORDER, bd=0)
        canvas_outer.pack(fill="both", expand=True)

        inner = tk.Frame(canvas_outer, bg=_CANVAS_BG)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self._canvas = tk.Canvas(inner, bg=_CANVAS_BG, highlightthickness=0)
        vbar = tk.Scrollbar(inner, orient="vertical", command=self._canvas.yview)
        hbar = tk.Scrollbar(inner, orient="horizontal", command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        self._canvas.pack(fill="both", expand=True)

        # 占位内容
        self._hint_items = []
        self._draw_empty_state()

        # 鼠标滚轮缩放 + 拖拽平移 + 自适应重绘
        self._canvas.bind("<MouseWheel>", self._on_wheel)      # Windows / macOS
        self._canvas.bind("<Button-4>", self._on_wheel)        # Linux up
        self._canvas.bind("<Button-5>", self._on_wheel)        # Linux down
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # 结果信息栏
        result_bar = tk.Frame(parent, bg=_SURFACE, height=74)
        result_bar.pack(fill="x", pady=(8, 0))
        result_bar.pack_propagate(False)

        self._result_badge = tk.Label(
            result_bar,
            text="WAITING",
            bg=_CARD,
            fg=_MUTED,
            font=("Consolas", 8, "bold"),
            padx=10,
            pady=3,
        )
        self._result_badge.pack(side="left", padx=(14, 10))

        self._result_lbl = tk.Label(
            result_bar,
            text="尚未运行推理",
            bg=_SURFACE,
            fg=_FG2,
            font=("Microsoft YaHei", 12, "bold"),
            anchor="w",
        )
        self._result_lbl.pack(side="left", fill="both", expand=True, padx=(0, 14))

    # ── 小组件工厂 ─────────────────────────────────────────────

    def _pack_line(self, padx=12, pady=3, fill="x"):
        return dict(padx=padx, pady=pady, fill=fill)

    def _card(self, parent):
        frame = tk.Frame(parent, bg=_CARD, bd=0, highlightthickness=1, highlightbackground=_BORDER)
        frame.pack(fill="x", padx=14, pady=(0, 10))
        return frame

    def _sec(self, parent, title, subtitle=None):
        row = tk.Frame(parent, bg=_CARD)
        row.pack(fill="x", padx=12, pady=(10, 5))

        tk.Label(
            row,
            text=title,
            bg=_CARD,
            fg=_FG,
            font=_FONT_CN_B,
            anchor="w",
        ).pack(side="left")

        if subtitle:
            tk.Label(
                row,
                text=subtitle,
                bg=_CARD,
                fg=_MUTED,
                font=_FONT_MONO_S,
                anchor="e",
            ).pack(side="right")

    def _lbl(self, parent, text, bg=_CARD):
        return tk.Label(parent, text=text, bg=bg, fg=_FG2, font=_FONT_CN, anchor="w")

    def _btn(self, parent, text, cmd, color=_CARD2, fg=_FG, **kw):
        btn = tk.Button(
            parent,
            text=text,
            command=cmd,
            bg=color,
            fg=fg,
            relief="flat",
            activebackground=color,
            activeforeground=fg,
            font=_FONT_CN_B,
            cursor="hand2",
            padx=8,
            pady=6,
            bd=0,
            **kw,
        )
        self._bind_button_hover(btn, color)
        return btn

    def _ddl(self, parent, var, *options):
        menu = tk.OptionMenu(parent, var, *options)
        menu.configure(
            bg=_ENTRY,
            fg=_FG,
            activebackground=_ACCENT,
            activeforeground="#11111B",
            relief="flat",
            font=_FONT_MONO,
            highlightthickness=1,
            highlightbackground=_BORDER,
            indicatoron=True,
        )
        menu["menu"].configure(
            bg=_ENTRY,
            fg=_FG,
            activebackground=_ACCENT,
            activeforeground="#11111B",
            font=_FONT_MONO,
        )
        return menu

    def _bind_button_hover(self, btn, normal_color):
        hover = self._lighten(normal_color, 0.10)

        def enter(_):
            if str(btn.cget("state")) != "disabled":
                btn.configure(bg=hover, activebackground=hover)

        def leave(_):
            btn.configure(bg=normal_color, activebackground=normal_color)

        btn.bind("<Enter>", enter)
        btn.bind("<Leave>", leave)

    @staticmethod
    def _lighten(hex_color, factor=0.12):
        """简单提亮颜色，用于按钮 hover。"""
        hex_color = hex_color.lstrip("#")
        try:
            r, g, b = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
        except Exception:
            return "#3A3A55"
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02X}{g:02X}{b:02X}"

    # ──────────────────────────────────────────────────────────
    #  事件处理
    # ──────────────────────────────────────────────────────────

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="选择 Checkpoint 文件",
            filetypes=[("PyTorch 模型", "*.pth *.pt"), ("所有文件", "*.*")],
        )
        if path:
            self.v_ckpt.set(path)

    def _on_browse_image(self):
        path = filedialog.askopenfilename(
            title="选择 X-Ray 图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        self.v_img_path.set(path)
        self._img_name.configure(text=os.path.basename(path), fg=_FG)
        self._result_lbl.configure(text="已选择图像，点击「运行推理」开始分析", fg=_FG2)
        self._result_badge.configure(text="SELECTED", fg=_CYAN)
        self._last_img = None
        self._zoom = 1.0
        self._refresh_zoom_label()

        # 预览原图
        try:
            img = Image.open(path).convert("RGB")
            self._show_image(img)
        except Exception as e:
            self._set_status(f"[Error] 加载图像失败: {e}", level="error")
            messagebox.showerror("图像加载失败", str(e))
            return

        self._set_status(f"已选择: {os.path.basename(path)}  |  点击「运行推理」开始分析")
        self._refresh_action_states()

    def _on_load_model(self):
        """后台线程加载模型。"""
        if self._busy:
            return

        def _worker():
            try:
                self.root.after(0, lambda: self._set_busy(True, "加载模型中…", badge="LOADING"))

                name = self.v_model.get()
                ckpt = self.v_ckpt.get()
                dev = resolve_device(self.v_device.get())
                model = load_model(name, ckpt, dev)
                trans = get_eval_transform(224)

                self._model, self._device = model, dev
                self._transform, self._model_name = trans, name

                self.root.after(0, lambda: self._finish_load_model(name, dev))
            except Exception as e:
                self.root.after(0, lambda: self._fail_load_model(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_load_model(self, name, dev):
        self._model_dot.configure(text=f"●  {name}  on  {dev}", fg=_GREEN)
        self._set_busy(False, f"模型已就绪: {name} → {dev}", badge="READY")
        self._refresh_action_states()

    def _fail_load_model(self, err):
        self._model_dot.configure(text="●  加载失败", fg=_RED)
        self._set_busy(False, f"[Error] {err}", level="error", badge="ERROR")
        self._refresh_action_states()
        messagebox.showerror("加载失败", str(err))

    def _on_run_inference(self):
        """后台线程推理 + 可视化。"""
        if self._busy:
            return

        if self._model is None:
            messagebox.showwarning("提示", "请先点击「加载模型」")
            return

        img_path = self.v_img_path.get()
        if not img_path or not os.path.isfile(img_path):
            messagebox.showwarning("提示", "请先选择有效的图像文件")
            return

        def _worker():
            try:
                self.root.after(0, lambda: self._set_busy(True, "推理中…", badge="RUNNING"))
                self.root.after(0, lambda: self._result_badge.configure(text="RUNNING", fg=_YELLOW))
                self.root.after(0, lambda: self._result_lbl.configure(text="正在分析图像，请稍候…", fg=_FG2))

                # 推理
                inp, logit, prob = infer_single(self._model, self._transform, img_path, self._device)
                thr = self.v_threshold.get()
                pred_label = int(prob >= thr)
                true_label = infer_true_label(img_path)

                # Grad-CAM
                gray_cam = None
                rgb_img = denormalize_image(inp)
                if self.v_gradcam.get():
                    tc = self.v_target.get()
                    tgt = pred_label if tc == "predicted" else (1 if tc == "pneumonia" else 0)
                    gray_cam, rgb_img = get_gradcam(self._model, self._model_name, inp, tgt)

                # 生成可视化面板（返回 PIL Image）
                pil_result = visualize_and_save(
                    image_path=img_path,
                    input_tensor=inp,
                    prob=prob,
                    pred_label=pred_label,
                    true_label=true_label,
                    threshold=thr,
                    model_name=self._model_name,
                    grayscale_cam=gray_cam,
                    rgb_img=rgb_img,
                    save_path=None,
                )
                self._last_img = pil_result

                # 结果文字
                pred_str = LABEL_NAME[pred_label]
                conf_pct = max(prob, 1 - prob) * 100
                res_txt = (
                    f"预测: {pred_str}    "
                    f"P(PNEUMONIA) = {prob:.4f}    "
                    f"置信度: {conf_pct:.1f}%"
                )
                if true_label is not None:
                    correct = pred_label == true_label
                    res_txt += "    ✓ 预测正确" if correct else "    ✗ 预测错误"

                res_fg = _RED if pred_label == 1 else _GREEN

                self.root.after(
                    0,
                    lambda: self._finish_inference(pil_result, res_txt, res_fg, img_path, pred_str, prob, pred_label),
                )

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.root.after(0, lambda: self._fail_inference(e, tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_inference(self, pil_img, res_txt, res_fg, img_path, pred_str, prob, pred_label):
        self._zoom = 1.0
        self._show_image(pil_img)

        self._result_lbl.configure(text=res_txt, fg=res_fg)
        badge_text = "PNEUMONIA" if pred_label == 1 else "NORMAL"
        self._result_badge.configure(text=badge_text, fg=res_fg)

        self._set_busy(False, f"完成  |  {os.path.basename(img_path)}  →  {pred_str}  (P = {prob:.4f})", badge="READY")
        self._refresh_action_states()

    def _fail_inference(self, err, traceback_text):
        self._set_busy(False, f"[Error] {err}", level="error", badge="ERROR")
        self._result_badge.configure(text="ERROR", fg=_RED)
        self._result_lbl.configure(text="推理失败，请查看错误信息", fg=_RED)
        self._refresh_action_states()
        messagebox.showerror("推理失败", traceback_text)

    def _on_save(self):
        if self._last_img is None:
            messagebox.showinfo("提示", "尚无推理结果，请先运行推理")
            return

        path = filedialog.asksaveasfilename(
            title="保存可视化结果",
            defaultextension=".png",
            filetypes=[("PNG 图像", "*.png"), ("JPEG", "*.jpg")],
            initialfile="result.png",
        )
        if path:
            self._last_img.save(path, dpi=(150, 150))
            self._set_status(f"已保存: {path}")
            self._refresh_action_states()

    # ── 图像显示 / 交互 ───────────────────────────────────────

    def _draw_empty_state(self):
        self._canvas.delete("all")
        self._hint_items.clear()

        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(), 700)
        ch = max(self._canvas.winfo_height(), 440)
        cx, cy = cw // 2, ch // 2

        self._hint_items.append(
            self._canvas.create_text(
                cx,
                cy - 28,
                text="请加载模型 → 选择图像 → 运行推理",
                fill=_FG2,
                font=("Microsoft YaHei", 15, "bold"),
            )
        )
        self._hint_items.append(
            self._canvas.create_text(
                cx,
                cy + 8,
                text="运行后将在此显示原图、预测结果与 Grad-CAM 可视化面板",
                fill=_MUTED,
                font=_FONT_CN,
            )
        )
        self._canvas.configure(scrollregion=(0, 0, cw, ch))

    def _show_image(self, pil_img: Image.Image):
        """将 PIL Image 缩放后渲染到 Canvas 中央。"""
        self._current_pil = pil_img
        self._canvas.update_idletasks()

        cw = max(self._canvas.winfo_width(), 200)
        ch = max(self._canvas.winfo_height(), 200)

        iw, ih = pil_img.size

        # 初始按窗口适配；滚轮缩放基于这个适配比例。
        base_scale = min(cw / iw, ch / ih, 1.0)
        scale = max(0.08, min(base_scale * self._zoom, 6.0))

        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(resized)

        self._canvas.delete("all")

        scroll_w = max(cw, nw + 40)
        scroll_h = max(ch, nh + 40)
        x = max(cw // 2, scroll_w // 2)
        y = max(ch // 2, scroll_h // 2)

        self._canvas.create_image(x, y, anchor="center", image=self._photo_ref)
        self._canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))

        # 尽量让图像居中显示
        if scroll_w > cw:
            self._canvas.xview_moveto(max(0, (x - cw / 2) / scroll_w))
        if scroll_h > ch:
            self._canvas.yview_moveto(max(0, (y - ch / 2) / scroll_h))

        self._refresh_zoom_label(scale=scale, base_scale=base_scale)

    def _on_wheel(self, event):
        """滚轮缩放图像。"""
        if self._current_pil is None:
            return

        # Windows/macOS: event.delta；Linux: event.num
        if event.num == 4 or event.delta > 0:
            self._zoom *= 1.12
        elif event.num == 5 or event.delta < 0:
            self._zoom /= 1.12

        self._zoom = max(0.2, min(self._zoom, 8.0))
        self._show_image(self._current_pil)

    def _on_drag_start(self, event):
        self._canvas.scan_mark(event.x, event.y)

    def _on_drag_move(self, event):
        self._canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_canvas_configure(self, _event):
        """窗口变化时延迟重绘，避免频繁 resize 卡顿。"""
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(120, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_after_id = None
        if self._current_pil is None:
            self._draw_empty_state()
        else:
            self._show_image(self._current_pil)

    def _reset_zoom(self):
        self._zoom = 1.0
        if self._current_pil is not None:
            self._show_image(self._current_pil)
        else:
            self._refresh_zoom_label()

    def _fit_to_window(self):
        self._zoom = 1.0
        if self._current_pil is not None:
            self._show_image(self._current_pil)
        else:
            self._draw_empty_state()
        self._refresh_zoom_label()

    def _refresh_zoom_label(self, scale=None, base_scale=None):
        """显示相对于适配窗口的缩放比例。"""
        if self._current_pil is None:
            self.v_zoom_text.set("100%")
            return

        # 对用户来说显示 _zoom 更直观：100% = 适配窗口。
        self.v_zoom_text.set(f"{int(self._zoom * 100)}%")

    # ── 状态栏 / 控件状态 ──────────────────────────────────────

    def _set_status(self, msg: str, level: str = "ok", badge=None):
        prefix = "✓  " if level == "ok" else "⚠  " if level == "warn" else "✕  "
        self.v_status.set(prefix + msg)

        if badge:
            self._status_badge.configure(text=badge)

        if level == "error":
            self._status_badge.configure(fg=_RED)
        elif level == "warn":
            self._status_badge.configure(fg=_YELLOW)
        else:
            self._status_badge.configure(fg=_GREEN)

    def _set_busy(self, busy: bool, msg=None, level: str = "ok", badge=None):
        self._busy = busy

        if msg is not None:
            prefix = "⏳  " if busy else ("✓  " if level == "ok" else "✕  ")
            self.v_status.set(prefix + msg)

        if badge:
            self._status_badge.configure(text=badge, fg=_YELLOW if busy else (_RED if level == "error" else _GREEN))

        self._refresh_action_states()

    def _refresh_action_states(self):
        """根据模型/图像/忙碌状态更新按钮可用性。"""
        has_model = self._model is not None
        has_image = bool(self.v_img_path.get()) and os.path.isfile(self.v_img_path.get())
        has_result = self._last_img is not None

        self._load_btn.configure(state="disabled" if self._busy else "normal")
        self._select_btn.configure(state="disabled" if self._busy else "normal")
        self._run_btn.configure(state="normal" if (has_model and has_image and not self._busy) else "disabled")
        self._save_btn.configure(state="normal" if (has_result and not self._busy) else "disabled")

        if self._busy:
            self._run_btn.configure(text="处理中…")
        else:
            self._run_btn.configure(text="▶  运行推理")


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()

    # DPI 感知（Windows 高分辨率屏幕）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    PneumoniaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
