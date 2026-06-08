# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.checks import check_font, check_version, is_ascii
from ultralytics.utils.files import increment_path


class Colors:
    """Ultralytics 可视化和绘图调色板。

    此类提供使用 Ultralytics 调色板的方法，包括将十六进制颜色代码转换为 RGB 值，以及访问用于目标检测和姿态估计的预定义配色方案。

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand).
        Please use the official Ultralytics colors for all marketing materials.

    Attributes:
        palette (list[tuple]): 通用 RGB 颜色元组列表。
        n (int): 调色板中的颜色数量。
        pose_palette (np.ndarray): 用于姿态估计的特定调色板数组，dtype np.uint8。

    Examples:
        >>> from ultralytics.utils.plotting import Colors
        >>> colors = Colors()
        >>> colors(5, True)  # 返回 BGR 格式：(221, 111, 255)
        >>> colors(5, False)  # 返回 RGB 格式：(255, 111, 221)
    """

    def __init__(self):
        """初始化颜色，hex = matplotlib.colors.TABLEAU_COLORS.values()。"""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: int | torch.Tensor, bgr: bool = False) -> tuple:
        """按索引从调色板返回颜色。

        Args:
            i (int | torch.Tensor): 颜色索引。
            bgr (bool, optional): 是否返回 BGR 格式而非 RGB。

        Returns:
            (tuple): RGB 或 BGR 颜色元组。
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """将十六进制颜色代码转换为 RGB 值（即默认 PIL 顺序）。"""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 创建实例，用于 'from utils.plots import colors'


class Annotator:
    """Ultralytics 标注器，用于训练/验证马赛克图、JPG 和预测标注。

    Attributes:
        im (Image.Image | np.ndarray): 要标注的图像。
        pil (bool): 是否使用 PIL 或 cv2 绘制标注。
        font (ImageFont.truetype | ImageFont.load_default): 用于文本标注的字体。
        lw (int): 绘制线宽。
        skeleton (list[list[int]]): 关键点骨架结构。
        limb_color (np.ndarray): 肢体调色板。
        kpt_color (np.ndarray): 关键点调色板。
        dark_colors (set): 用于文本对比度的深色集合。
        light_colors (set): 用于文本对比度的浅色集合。

    Examples:
        >>> from ultralytics.utils.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """使用图像、线宽以及关键点和肢体的调色板初始化 Annotator 类。"""
        non_ascii = not is_ascii(example)  # 非拉丁标签，如中文、阿拉伯语、西里尔字母
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if not input_is_pil:
            if im.shape[2] == 1:  # 处理灰度图
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 2:  # 处理双通道图像
                im = np.ascontiguousarray(np.dstack((im, np.zeros_like(im[..., :1]))))
            elif im.shape[2] > 3:  # 多光谱
                im = np.ascontiguousarray(im[..., :3])
        if self.pil:  # 使用 PIL
            self.im = im if input_is_pil else Image.fromarray(im)  # 保持在 BGR，因为调色板是 BGR
            if self.im.mode not in {"RGB", "RGBA"}:  # 多光谱
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # 弃用修复: w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # 文本宽度、高度
        else:  # 使用 cv2
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # 字体粗细
            self.sf = self.lw / 3  # 字体缩放
        # 姿态
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """根据背景色分配文本颜色。

        Args:
            color (tuple, optional): 文本矩形的背景色。
            txt_color (tuple, optional): 文本的回退颜色。

        Returns:
            (tuple): 标签的文本颜色。

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # return (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(self, box, label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """在图像上绘制带有给定标签的边界框。

        Args:
            box (tuple): 边界框坐标 (x1, y1, x2, y2)。
            label (str, optional): 要显示的文本标签。
            color (tuple, optional): 矩形的背景色。
            txt_color (tuple, optional): 文本颜色。

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        multi_points = isinstance(box[0], list)  # 多点，形状 (n, 2)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))
        if self.pil:
            self.draw.polygon(
                [tuple(b) for b in box], width=self.lw, outline=color
            ) if multi_points else self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)  # 文本宽度、高度
                outside = p1[1] >= h  # 标签适合放在框外
                if p1[0] > self.im.size[0] - w:  # 尺寸为 (w, h)，检查标签是否超出图像右侧
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text([box[0], box[1]], label, fill=txt_color, font=self.font, anchor='ls')  # 用于 PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            cv2.polylines(
                self.im, [np.asarray(box, dtype=int)], True, color, self.lw
            ) if multi_points else cv2.rectangle(
                self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # 文本宽度、高度
                h += 3  # 添加像素以填充文本
                outside = p1[1] >= h  # 标签适合放在框外
                if p1[0] > self.im.shape[1] - w:  # 形状为 (h, w)，检查标签是否超出图像右侧
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # 填充
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu: torch.Tensor = None, alpha: float = 0.5, retina_masks: bool = False):
        """在图像上绘制掩码。

        Args:
            masks (torch.Tensor | np.ndarray): 预测掩码，形状 [n, h, w]。
            colors (list[list[int]]): 预测掩码的颜色，[[r, g, b] * n]。
            im_gpu (torch.Tensor | None): GPU 上的图像，形状 [3, h, w]，范围 [0, 1]。
            alpha (float, optional): 掩码透明度：0.0 完全透明，1.0 不透明。
            retina_masks (bool, optional): 是否使用高分辨率掩码。
        """
        if self.pil:
            # 先转为 numpy
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            assert isinstance(masks, np.ndarray), "`masks` must be a np.ndarray if `im_gpu` is not provided."
            overlay = self.im.copy()
            for i, mask in enumerate(masks):
                overlay[mask.astype(bool)] = colors[i]
            self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        else:
            assert isinstance(masks, torch.Tensor), "'masks' must be a torch.Tensor if 'im_gpu' is provided."
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
                return
            if im_gpu.device != masks.device:
                im_gpu = im_gpu.to(masks.device)

            ih, iw = self.im.shape[:2]
            if not retina_masks:
                # 使用 scale_masks 正确移除填充并上采样，先将 bool 转为 float
                masks = ops.scale_masks(masks[None].float(), (ih, iw))[0] > 0.5
                # 将原始 BGR 图像转为 RGB 张量
                im_gpu = (
                    torch.from_numpy(self.im).to(masks.device).permute(2, 0, 1).flip(0).contiguous().float() / 255.0
                )

            colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # 形状(n,3)
            colors = colors[:, None, None]  # 形状(n,1,1,3)
            masks = masks.unsqueeze(3)  # 形状(n,h,w,1)
            masks_color = masks * (colors * alpha)  # 形状(n,h,w,3)
            inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # 形状(n,h,w,1)
            mcs = masks_color.max(dim=0).values  # 形状(h,w,3)

            im_gpu = im_gpu.flip(dims=[0]).permute(1, 2, 0).contiguous()  # 形状(h,w,3)
            im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
            self.im[:] = (im_gpu * 255).byte().cpu().numpy()
        if self.pil:
            # 将 im 转回 PIL 并更新 draw
            self.fromarray(self.im)

    def kpts(
        self,
        kpts,
        shape: tuple = (640, 640),
        radius: int | None = None,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
        kpt_color: tuple | None = None,
    ):
        """在图像上绘制关键点。

        Args:
            kpts (torch.Tensor): 关键点，形状 [17, 3] (x, y, confidence)。
            shape (tuple, optional): 图像形状 (h, w)。
            radius (int, optional): 关键点半径。
            kpt_line (bool, optional): 在关键点之间画线。
            conf_thres (float, optional): 置信度阈值。
            kpt_color (tuple, optional): 关键点颜色。

        Notes:
            - `kpt_line=True` 目前仅支持人体姿态绘制。
            - 原地修改 self.im。
            - 如果 self.pil 为 True，将图像转为 numpy 数组再转回 PIL。
        """
        radius = radius if radius is not None else self.lw
        if self.pil:
            # 先转为 numpy
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` 目前仅支持人体姿态绘制
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )
        if self.pil:
            # 将 im 转回 PIL 并更新 draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1):
        """向图像添加矩形（仅 PIL）。"""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text: str, txt_color: tuple = (255, 255, 255), anchor: str = "top", box_color: tuple = ()):
        """使用 PIL 或 cv2 向图像添加文本。

        Args:
            xy (list[int]): 文本放置的左上角坐标。
            text (str): 要绘制的文本。
            txt_color (tuple, optional): 文本颜色。
            anchor (str, optional): 文本锚点位置（'top' 或 'bottom'）。
            box_color (tuple, optional): 框背景色，可带 alpha。
        """
        if self.pil:
            w, h = self.font.getsize(text)
            if anchor == "bottom":  # 从字体底部开始 y
                xy[1] += 1 - h
            for line in text.split("\n"):
                if box_color:
                    # 为每行绘制矩形
                    w, h = self.font.getsize(line)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=box_color)
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            if box_color:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3  # 添加像素以填充文本
                outside = xy[1] >= h  # 标签适合放在框外
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, box_color, -1, cv2.LINE_AA)  # 填充
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """从 NumPy 数组或 PIL 图像更新 `self.im`。"""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self, pil=False):
        """返回标注后的图像（数组或 PIL 图像）。"""
        im = np.asarray(self.im)  # self.im 为 BGR
        return Image.fromarray(im[..., ::-1]) if pil else im

    def show(self, title: str | None = None):
        """显示标注后的图像。"""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # 将 BGR NumPy 数组转为 RGB PIL 图像
        if IS_COLAB or IS_KAGGLE:  # 不能使用 IS_JUPYTER，因为它对所有 IPython 环境都会运行
            try:
                display(im)  # noqa - display() function only available in ipython environments
            except ImportError as e:
                LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        else:
            im.show(title=title)

    def save(self, filename: str = "image.jpg"):
        """将标注后的图像保存到 'filename'。"""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox: tuple | list):
        """计算边界框的尺寸和面积。

        Args:
            bbox (tuple | list): 边界框坐标，格式 (x_min, y_min, x_max, y_max)。

        Returns:
            width (float): 边界框宽度。
            height (float): 边界框高度。
            area (float): 边界框包围的面积。

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_bbox_dimension(bbox=[10, 20, 30, 40])
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height


@TryExcept()
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """绘制训练标签，包括类别直方图和框统计。

    Args:
        boxes (np.ndarray): 边界框坐标，格式 [x, y, width, height]。
        cls (np.ndarray): 类别索引。
        names (dict, optional): 类别索引到类别名称的映射字典。
        save_dir (Path, optional): 保存图表的目录。
        on_plot (Callable, optional): 图表保存后调用的函数。
    """
    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'
    import polars
    from matplotlib.colors import LinearSegmentedColormap

    # 绘制数据集标签
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # 类别数
    boxes = boxes[:1000000]  # 限制为 100 万个框
    x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])

    # Matplotlib 标签
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        ax[0].bar_label(y[2])
    else:
        ax[0].set_xlabel("classes")
    boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for class_id, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(class_id))  # 绘制
    ax[1].imshow(img)
    ax[1].axis("off")

    ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax[3].set_xlabel("width")
    ax[3].set_ylabel("height")
    for a in {0, 1, 2, 3}:
        for s in {"top", "right", "left", "bottom"}:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(
    xyxy,
    im,
    file: Path = Path("im.jpg"),
    gain: float = 1.02,
    pad: int = 10,
    square: bool = False,
    BGR: bool = False,
    save: bool = True,
):
    """将图像裁剪保存为 {file}，裁剪尺寸乘以 {gain} 并加 {pad} 像素。保存和/或返回裁剪结果。

    此函数接收一个边界框和一张图像，然后根据边界框保存图像的裁剪部分。裁剪可以可选地变为正方形，
    函数允许对边界框进行增益和填充调整。

    Args:
        xyxy (torch.Tensor | list): 表示 xyxy 格式边界框的张量或列表。
        im (np.ndarray): 输入图像。
        file (Path, optional): 裁剪图像的保存路径。
        gain (float, optional): 增大边界框尺寸的乘法因子。
        pad (int, optional): 添加到边界框宽度和高度的像素数。
        square (bool, optional): 若为 True，边界框将转换为正方形。
        BGR (bool, optional): 若为 True，图像将以 BGR 格式返回，否则为 RGB。
        save (bool, optional): 若为 True，裁剪图像将保存到磁盘。

    Returns:
        (np.ndarray): 裁剪后的图像。

    Examples:
        >>> from ultralytics.utils.plotting import save_one_box
        >>> xyxy = [50, 50, 150, 150]
        >>> im = cv2.imread("image.jpg")
        >>> cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
    """
    if not isinstance(xyxy, torch.Tensor):  # 可能是列表
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # 框
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 尝试将矩形变为正方形
    b[:, 2:] = b[:, 2:] * gain + pad  # 框宽高 * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    grayscale = im.shape[2] == 1  # 灰度图像
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR or grayscale else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # 保存 BGR，https://github.com/ultralytics/yolov5/issues/7007 色度子采样问题
        crop = crop.squeeze(-1) if grayscale else crop[..., ::-1] if BGR else crop
        Image.fromarray(crop).save(f, quality=95, subsampling=0)  # 保存 RGB
    return crop


@threaded
def plot_images(
    labels: dict[str, Any],
    images: torch.Tensor | np.ndarray = np.zeros((0, 3, 640, 640), dtype=np.float32),
    paths: list[str] | None = None,
    fname: str = "images.jpg",
    names: dict[int, str] | None = None,
    on_plot: Callable | None = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> np.ndarray | None:
    """绘制带标签、边界框、掩码和关键点的图像网格。

    Args:
        labels (dict[str, Any]): 包含检测数据的字典，键如 'cls'、'bboxes'、'conf'、'masks'、
            'keypoints'、'batch_idx'、'img'。
        images (torch.Tensor | np.ndarray): 要绘制的图像批次。形状: (batch_size, channels, height, width)。
        paths (list[str] | None): 批次中每张图像的文件路径列表。
        fname (str): 绘制的图像网格的输出文件名。
        names (dict[int, str] | None): 类别索引到类别名称的映射字典。
        on_plot (Callable | None): 保存图表后调用的回调函数。
        max_size (int): 输出图像网格的最大尺寸。
        max_subplots (int): 图像网格中子图的最大数量。
        save (bool): 是否将绘制的图像网格保存到文件。
        conf_thres (float): 显示检测的置信度阈值。

    Returns:
        (np.ndarray | None): 如果 save 为 False 则返回 numpy 数组，否则返回 None。

    Notes:
        此函数同时支持张量和 numpy 数组输入。它会自动将张量输入转换为 numpy 数组进行处理。

        Channel Support:
        - 1 channel: Grayscale
        - 2 channels: Third channel added as zeros
        - 3 channels: Used as-is (standard RGB)
        - 4+ channels: Cropped to first 3 channels
    """
    for k in {"cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images"}:
        if k not in labels:
            continue
        if k == "cls" and labels[k].ndim == 2:
            labels[k] = labels[k].squeeze(1)  # 如果形状为 (n, 1) 则压缩
        if isinstance(labels[k], torch.Tensor):
            labels[k] = labels[k].cpu().numpy()

    cls = labels.get("cls", np.zeros(0, dtype=np.int64))
    batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
    bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
    confs = labels.get("conf", None)
    masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
    kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
    images = labels.get("img", images)  # 默认使用输入图像

    if len(images) and isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # 处理双通道和多通道图像
    c = images.shape[1]
    if c == 2:
        zero = np.zeros_like(images[:, :1])
        images = np.concatenate((images, zero), axis=1)  # 用黑色通道填充双通道
    elif c > 3:
        images = images[:, :3]  # 将多光谱图像裁剪到前 3 个通道

    bs, _, h, w = images.shape  # 批次大小, _, 高度, 宽度
    bs = min(bs, max_subplots)  # 限制绘图图像数
    ns = np.ceil(bs**0.5)  # 子图数量（正方形）
    if np.max(images[0]) <= 1:
        images *= 255  # 反归一化（可选）

    # 构建图像
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # 初始化
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 块起点
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # 调整大小（可选）
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # 标注
    fs = int((h + w) * ns * 0.01)  # 字体大小
    fs = max(fs, 18)  # 确保字体大小足够大以便阅读。
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 块起点
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # 边框
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # 文件名
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None
            conf = confs[idx] if confs is not None else None  # 检查是否有置信度（标签 vs 预测）

            if len(bboxes):
                boxes = bboxes[idx]
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # 如果已归一化，容差 0.1
                        boxes[..., [0, 2]] *= w  # 缩放到像素
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # 如果图像缩放，绝对坐标需要缩放
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    label = f"{c}" if labels else f"{c} {conf[0]:.1f}"
                    annotator.text([x, y], label, txt_color=color, box_color=(64, 64, 64, 128))

            # 绘制关键点
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # 如果已归一化，容差 0.01
                        kpts_[..., 0] *= w  # 缩放到像素
                        kpts_[..., 1] *= h
                    elif scale < 1:  # 如果图像缩放，绝对坐标需要缩放
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # 绘制掩码
            if len(masks):
                if idx.shape[0] == masks.shape[0] and masks.max() <= 1:  # overlap_mask=False
                    image_masks = masks[idx]
                else:  # overlap_mask=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(1, nl + 1).reshape((nl, 1, 1))
                    image_masks = (image_masks == index).astype(np.float32)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # 保存
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None):
    """从结果 CSV 文件绘制训练结果。此函数支持多种数据类型，包括分割、姿态估计和分类。图表保存为 CSV 所在目录下的 'results.png'。

    Args:
        file (str, optional): 包含训练结果的 CSV 文件路径。
        dir (str, optional): 如果未提供 'file'，CSV 文件所在的目录。
        on_plot (Callable, optional): 绘图后执行的回调函数。以文件名为参数。

    Examples:
        >>> from ultralytics.utils.plotting import plot_results
        >>> plot_results("path/to/results.csv")
    """
    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    fig, ax = None, None
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = (
                    loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                )
                fig, ax = plt.subplots(2, len(columns) // 2, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel()
            x = data.select(data.columns[0]).to_numpy().flatten()
            for i, j in enumerate(columns):
                y = data.select(j).to_numpy().flatten().astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # 实际结果
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # 平滑线
                ax[i].set_title(j, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")
    if ax is not None:
        ax[1].legend()
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()
        if on_plot:
            on_plot(fname)


def plt_color_scatter(v, f, bins: int = 20, cmap: str = "viridis", alpha: float = 0.8, edgecolors: str = "none"):
    """绘制基于二维直方图着色的散点图。

    Args:
        v (array-like): x 轴的值。
        f (array-like): y 轴的值。
        bins (int, optional): 直方图的分箱数。
        cmap (str, optional): 散点图的颜色映射。
        alpha (float, optional): 散点图的透明度。
        edgecolors (str, optional): 散点图的边缘颜色。

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'

    # 计算二维直方图及对应颜色
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # 散点图
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


@plt_settings()
def plot_tune_results(results_file: str = "tune_results.ndjson", exclude_zero_fitness_points: bool = True):
    """绘制存储在调优 NDJSON 文件中的进化结果。

    Args:
        results_file (str, optional): 包含调优结果的 NDJSON 文件路径。
        exclude_zero_fitness_points (bool, optional): 不在调优图中包含适应度为零的点。

    Examples:
        >>> plot_tune_results("path/to/tune_results.ndjson")
    """
    import json

    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """将一个 matplotlib 图表保存到 'file'。"""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"Saved {file}")

    results_file = Path(results_file)
    with open(results_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return

    keys = list(records[0].get("hyperparameters", {}))
    x = np.array(
        [[r.get("fitness", 0.0)] + [r.get("hyperparameters", {}).get(k, np.nan) for k in keys] for r in records],
        dtype=float,
    )
    len(x)
    all_fitness = x[:, 0]  # 适应度
    zero_mask = slice(None)
    if exclude_zero_fitness_points:
        zero_mask = all_fitness > 0  # 排除适应度为零的点
        x, all_fitness = x[zero_mask], all_fitness[zero_mask]
    if len(all_fitness) == 0:
        LOGGER.warning("No valid fitness values to plot (all iterations may have failed)")
        return
    fitness = all_fitness.copy()
    # 仅对下界进行迭代 sigma 拒绝
    for _ in range(3):  # 最多 3 次迭代
        mean, std = fitness.mean(), fitness.std()
        lower_bound = mean - 3 * std
        mask = fitness >= lower_bound
        if mask.all():  # 没有更多异常值
            break
        x, fitness = x[mask], fitness[mask]
    j = np.argmax(fitness)  # 最大适应度索引
    n = math.ceil(len(keys) ** 0.5)  # 图中的列数和行数
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + 1]
        mu = v[j]  # 最佳单次结果
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # 限制为 40 个字符
        plt.tick_params(axis="both", labelsize=8)  # 设置轴标签大小为 8
        if i % n != 0:
            plt.yticks([])
    _save_one_file(results_file.with_name("tune_scatter_plots.png"))

    # 适应度 vs 迭代
    x = range(1, len(all_fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    for dataset in sorted({k for r in records for k in r.get("datasets", {})}):
        y = np.array([r.get("datasets", {}).get(dataset, {}).get("fitness", np.nan) for r in records], dtype=float)
        if exclude_zero_fitness_points and not isinstance(zero_mask, slice):
            y = y[zero_mask]
        plt.plot(x, y, "o", markersize=5, alpha=0.8, label=dataset)
    plt.plot(x, gaussian_filter1d(all_fitness, sigma=3), ":", color="0.35", label="smoothed mean", linewidth=2)
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    _save_one_file(results_file.with_name("tune_fitness.png"))


@plt_settings()
def feature_visualization(x, module_type: str, stage: int, n: int = 32, save_dir: Path = Path("runs/detect/exp")):
    """在推理过程中可视化给定模型模块的特征图。

    Args:
        x (torch.Tensor): 要可视化的特征。
        module_type (str): 模块类型。
        stage (int): 模型中的模块阶段。
        n (int, optional): 要绘制的最大特征图数量。
        save_dir (Path, optional): 保存结果的目录。
    """
    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'

    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # 所有模型头
        if m in module_type:
            return
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # 批次, 通道, 高度, 宽度
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.rsplit('.', 1)[-1]}_features.png"  # 文件名

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # 选择批次索引 0，按通道分块
            n = min(n, channels)  # 绘图数量
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 行 x n/8 列
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy 保存
