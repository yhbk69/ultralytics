# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    Pose26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, SETTINGS, WINDOWS, YAML, colorstr, emojis
from ultralytics.utils.checks import REMOTE_FILE_PREFIXES, check_file, check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2ELoss,
    PoseLoss26,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Ultralytics 系列中所有 YOLO 模型的基类。

    该类为 YOLO 模型提供通用功能，包括前向传播处理、模型融合、信息
    显示和权重加载能力。

    Attributes:
        model (torch.nn.Sequential): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        stride (torch.Tensor): 模型步长值。

    Methods:
        forward: 执行训练或推理的前向传播。
        predict: 对输入张量执行推理。
        fuse: 融合 Conv/BatchNorm 层并进行重参数化以优化。
        info: 打印模型信息。
        load: 将权重加载到模型中。
        loss: 计算训练损失。

    Examples:
        创建 BaseModel 实例
        >>> model = BaseModel()
        >>> model.info()  # 显示模型信息
    """

    def forward(self, x, *args, **kwargs):
        """执行模型的前向传播，用于训练或推理。

        如果 x 是字典，则计算并返回训练损失；否则返回推理预测结果。

        Args:
            x (torch.Tensor | dict): 推理时的输入张量，或训练时包含图像张量和标签的字典。
            *args (Any): 可变长度参数列表。
            **kwargs (Any): 任意关键字参数。

        Returns:
            (torch.Tensor): 如果 x 是字典则为损失（训练），否则为网络预测（推理）。
        """
        if isinstance(x, dict):  # 用于训练和训练过程中的验证场景。
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """通过网络执行一次前向传播。

        Args:
            x (torch.Tensor): 输入到模型的张量。
            profile (bool): 如果为 True，打印每一层的计算时间。
            visualize (bool): 如果为 True，保存模型的特征图。
            augment (bool): 预测时对图像进行增强。
            embed (list, optional): 返回嵌入的层索引列表。

        Returns:
            (torch.Tensor): 模型的最后输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """通过网络执行一次前向传播。

        Args:
            x (torch.Tensor): 输入到模型的张量。
            profile (bool): 如果为 True，打印每一层的计算时间。
            visualize (bool): 如果为 True，保存模型的特征图。
            embed (list, optional): 返回嵌入的层索引列表。

        Returns:
            (torch.Tensor): 模型的最后输出。
        """
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """对输入图像 x 执行增强并返回增强推理结果。"""
        LOGGER.warning(
            f"{self.__class__.__name__} 不支持 'augment=True' 预测。"
            f"回退到单尺度预测。"
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """分析模型单层在给定输入上的计算时间和 FLOPs。

        Args:
            m (torch.nn.Module): 要分析的层。
            x (torch.Tensor): 输入到该层的数据。
            dt (list): 存储该层计算时间的列表。
        """
        try:
            import thop
        except ImportError:
            thop = None  # 未安装 ultralytics-thop 时的 conda 支持

        c = m == self.model[-1] and isinstance(x, list)  # 是否为最后一层列表，复制输入作为原地操作修复
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """融合 Conv/ConvTranspose 和 BatchNorm 层，并对 RepConv/RepVGGDW 进行重参数化以提高效率。

        Args:
            verbose (bool): 融合后是否打印模型信息。

        Returns:
            (torch.nn.Module): 返回融合后的模型。
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积
                    delattr(m, "bn")  # 移除批归一化
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # 移除批归一化
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()  # 移除 one2many 头
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """检查模型中的归一化层数量是否低于给定阈值。

        Args:
            thresh (int, optional): 归一化层数量的阈值。

        Returns:
            (bool): 如果模型中归一化层数量小于阈值则为 True，否则为 False。
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # 归一化层，例如 BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # 如果模型中 BatchNorm 层数量小于 'thresh' 则为 True

    def info(self, detailed=False, verbose=True, imgsz=640):
        """打印模型信息。

        Args:
            detailed (bool): 如果为 True，打印模型的详细信息。
            verbose (bool): 如果为 True，打印模型信息。
            imgsz (int): 用于计算模型信息的图像尺寸。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """将函数应用到模型中的所有张量，包括 Detect 头属性如 stride 和 anchors。

        Args:
            fn (function): 要应用到模型的函数。

        Returns:
            (BaseModel): 更新后的 BaseModel 对象。
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # 包含所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect、YOLOEDetect、YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """将权重加载到模型中。

        Args:
            weights (dict | torch.nn.Module): 要加载的预训练权重。
            verbose (bool, optional): 是否记录传输进度。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision 模型不是字典
        csd = model.float().state_dict()  # 检查点 state_dict 转为 FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # 取交集
        self.load_state_dict(updated_csd, strict=False)  # 加载
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # 目前针对 yolo 模型硬编码
        # 主要用于加速多通道训练
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"从预训练权重中传输了 {len_updated_csd}/{len(self.model.state_dict())} 项")

    def loss(self, batch, preds=None):
        """计算损失。

        Args:
            batch (dict): 要计算损失的批次。
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """为 BaseModel 初始化损失准则。"""
        raise NotImplementedError("compute_loss() 需要由任务头实现")


class DetectionModel(BaseModel):
    """YOLO 检测模型。

    该类实现了 YOLO 检测架构，处理模型初始化、前向传播、增强
    推理和目标检测任务的损失计算。

    Attributes:
        yaml (dict): 模型配置字典。
        model (torch.nn.Sequential): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        names (dict): 类别名称字典。
        inplace (bool): 是否使用原地操作。
        end2end (bool): 模型是否使用端到端检测。
        stride (torch.Tensor): 模型步长值。

    Methods:
        __init__: 初始化 YOLO 检测模型。
        _predict_augment: 执行增强推理。
        _descale_pred: 增强推理后对预测结果进行反缩放。
        _clip_augmented: 裁剪 YOLO 增强推理的尾部。
        init_criterion: 初始化损失准则。

    Examples:
        初始化检测模型
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLO 检测模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` 模块已弃用，请使用 torch.nn.Identity。"
                "请删除本地 *.pt 文件并重新下载最新的模型检查点。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # 定义模型
        self.yaml["channels"] = ch  # 保存通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"使用 nc={nc} 覆盖 model.yaml 中的 nc={self.yaml['nc']}")
            self.yaml["nc"] = nc  # 覆盖 YAML 值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型，保存列表
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认名称字典
        self.inplace = self.yaml.get("inplace", True)

        # 构建步长
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包含所有 Detect 子类，如 Segment、Pose、OBB、YOLOEDetect、YOLOESegment
            s = 256  # 2x 最小步长
            m.inplace = self.inplace

            def _forward(x):
                """通过模型执行前向传播，根据不同 Detect 子类类型进行相应处理。"""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()  # 避免在训练开始前改变批次统计量
            m.training = True  # 设置为 True 以正确返回步长
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向传播
            self.stride = m.stride
            self.model.train()  # 将模型设置回训练（默认）模式
            m.bias_init()  # 只运行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步长，例如 RTDETR

        # 初始化权重和偏置
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """返回模型是否使用端到端无 NMS 检测。"""
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """覆盖端到端检测模式。"""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """设置模型头部（最后一层）的属性。

        Args:
            **kwargs (Any): 表示要设置的属性的任意关键字参数。
        """
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"头部没有属性 '{k}'。")
                continue
            setattr(head, k, v)

    def _predict_augment(self, x):
        """对输入图像 x 执行增强并返回增强推理和训练输出。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            (tuple[torch.Tensor, None]): 增强推理输出和训练输出（None）。
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("模型不支持 'augment=True'，回退到单尺度预测。")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # 高度，宽度
        s = [1, 0.83, 0.67]  # 尺度
        f = [None, 3, None]  # 翻转（2-上下翻转，3-左右翻转）
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # 前向传播
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强尾部
        return torch.cat(y, -1), None  # 增强推理，训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """增强推理后对预测结果进行反缩放（逆操作）。

        Args:
            p (torch.Tensor): 预测张量。
            flips (int | None): 翻转类型（None=无，2=上下翻转，3=左右翻转）。
            scale (float): 缩放因子。
            img_size (tuple): 原始图像尺寸（高度，宽度）。
            dim (int): 分割维度。

        Returns:
            (torch.Tensor): 反缩放后的预测结果。
        """
        p[:, :4] /= scale  # 反缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 反上下翻转
        elif flips == 3:
            x = img_size[1] - x  # 反左右翻转
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """裁剪 YOLO 增强推理的尾部。

        Args:
            y (list[torch.Tensor]): 检测张量列表。

        Returns:
            (list[torch.Tensor]): 裁剪后的检测张量。
        """
        nl = self.model[-1].nl  # 检测层数量（P3-P5）
        g = sum(4**x for x in range(nl))  # 网格点
        e = 1  # 排除层数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大目标
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小目标
        return y

    def init_criterion(self):
        """为 DetectionModel 初始化损失准则。"""
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO 方向边界框（OBB）模型。

    该类扩展了 DetectionModel，用于处理方向边界框检测任务，为旋转目标检测提供专门的损失
    计算。

    Methods:
        __init__: 初始化 YOLO OBB 模型。
        init_criterion: 初始化 OBB 检测的损失准则。

    Examples:
        初始化 OBB 模型
        >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLO OBB 模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """为模型初始化损失准则。"""
        return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO 分割模型。

    该类扩展了 DetectionModel，用于处理实例分割任务，为像素级目标检测和分割提供专门的损失
    计算。

    Methods:
        __init__: 初始化 YOLO 分割模型。
        init_criterion: 初始化分割的损失准则。

    Examples:
        初始化分割模型
        >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 Ultralytics YOLO 分割模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """为 SegmentationModel 初始化损失准则。"""
        return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO 姿态模型。

    该类扩展了 DetectionModel，用于处理人体姿态估计任务，为关键点检测和姿态估计提供专门的损失
    计算。

    Attributes:
        kpt_shape (tuple): 关键点数据的形状（关键点数量，维度数量）。

    Methods:
        __init__: 初始化 YOLO 姿态模型。
        init_criterion: 初始化姿态估计的损失准则。

    Examples:
        初始化姿态模型
        >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """初始化 Ultralytics YOLO 姿态模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            data_kpt_shape (tuple): 关键点数据的形状。
            verbose (bool): 是否显示模型信息。
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # 加载模型 YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"使用 kpt_shape={data_kpt_shape} 覆盖 model.yaml 中的 kpt_shape={cfg['kpt_shape']}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """为 PoseModel 初始化损失准则。"""
        return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO 分类模型。

    该类实现了用于图像分类任务的 YOLO 分类架构，提供模型
    初始化、配置和输出重塑功能。

    Attributes:
        yaml (dict): 模型配置字典。
        model (torch.nn.Sequential): 神经网络模型。
        stride (torch.Tensor): 模型步长值。
        names (dict): 类别名称字典。

    Methods:
        __init__: 初始化 ClassificationModel。
        _from_yaml: 设置模型配置并定义架构。
        reshape_outputs: 将模型更新为指定的类别数量。
        init_criterion: 初始化损失准则。

    Examples:
        初始化分类模型
        >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):
        """使用 YAML、通道数、类别数量和 verbose 标志初始化 ClassificationModel。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """设置 Ultralytics YOLO 模型配置并定义模型架构。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典

        # 定义模型
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # 输入通道
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"使用 nc={nc} 覆盖 model.yaml 中的 nc={self.yaml['nc']}")
            self.yaml["nc"] = nc  # 覆盖 YAML 值
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("未指定 nc。必须在 model.yaml 或函数参数中指定 nc。")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型，保存列表
        self.stride = torch.Tensor([1])  # 无步长约束
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认名称字典
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """如果需要，将 TorchVision 分类模型更新为类别数量 'nc'。

        Args:
            model (torch.nn.Module): 要更新的模型。
            nc (int): 新的类别数量。
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # 最后一个模块
        if isinstance(m, Classify):  # YOLO Classify() 头部
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # 最后一个 torch.nn.Linear 索引
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # 最后一个 torch.nn.Conv2d 索引
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """为 ClassificationModel 初始化损失准则。"""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR（使用 Transformer 的实时检测与跟踪）检测模型类。

    该类负责构建 RTDETR 架构，定义损失函数，并促进
    训练和推理过程。RTDETR 是一个从
    DetectionModel 基类扩展而来的目标检测和跟踪模型。

    Attributes:
        nc (int): 检测的类别数量。
        criterion (RTDETRDetectionLoss): 训练的损失函数。

    Methods:
        __init__: 初始化 RTDETRDetectionModel。
        init_criterion: 初始化损失准则。
        loss: 计算训练损失。
        predict: 执行模型的前向传播。

    Examples:
        初始化 RTDETR 模型
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """初始化 RTDETRDetectionModel。

        Args:
            cfg (str | dict): 配置文件名或路径。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 初始化期间打印附加信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _apply(self, fn):
        """将函数应用到模型中的所有张量，包括解码器的锚点和有效掩码。

        Args:
            fn (function): 要应用到模型的函数。

        Returns:
            (RTDETRDetectionModel): 更新后的 RTDETRDetectionModel 对象。
        """
        self = super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """为 RTDETRDetectionModel 初始化损失准则。"""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """计算给定批次数据的损失。

        Args:
            batch (dict): 包含图像和标签数据的字典。
            preds (tuple, optional): 预先计算的模型预测。

        Returns:
            (torch.Tensor): 总损失值。
            (torch.Tensor): 三个主要损失的张量。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: 将 gt_bbox 和 gt_labels 预处理为列表。
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: RTDETR 中大约有 12 个损失，反向传播使用所有损失但只显示三个主要损失。
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """执行模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            profile (bool): 如果为 True，分析每一层的计算时间。
            visualize (bool): 如果为 True，保存特征图用于可视化。
            batch (dict, optional): 评估用的真实数据。
            augment (bool): 如果为 True，推理期间执行数据增强。
            embed (list, optional): 返回嵌入的层索引列表。

        Returns:
            (torch.Tensor): 模型的输出张量。
        """
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # 除头部外
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # 头部推理
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World 模型。

    该类实现了用于开放词汇目标检测的 YOLOv8 World 模型，支持基于文本的类别
    规范和 CLIP 模型集成，以实现零样本检测能力。

    Attributes:
        txt_feats (torch.Tensor): 类别的文本特征嵌入。
        clip_model (torch.nn.Module): 用于文本编码的 CLIP 模型。

    Methods:
        __init__: 初始化 YOLOv8 world 模型。
        set_classes: 设置离线推理的类别。
        get_text_pe: 获取文本位置嵌入。
        predict: 使用文本特征执行前向传播。
        loss: 使用文本特征计算损失。

    Examples:
        初始化 world 模型
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOv8 world 模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # 特征占位符
        self.clip_model = None  # CLIP 模型占位符
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """提前设置类别，使模型可以在没有 clip 模型的情况下进行离线推理。

        Args:
            text (list[str]): 类别名称列表。
            batch (int): 处理文本标记的批次大小。
            cache_clip_model (bool): 是否缓存 CLIP 模型。
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """使用 CLIP 模型获取文本位置嵌入。

        Args:
            text (list[str]): 类别名称列表。
            batch (int): 处理文本标记的批次大小。
            cache_clip_model (bool): 是否缓存 CLIP 模型。

        Returns:
            (torch.Tensor): 文本位置嵌入。
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # 为了向后兼容缺少 clip_model 属性的模型
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """执行模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            profile (bool): 如果为 True，分析每一层的计算时间。
            visualize (bool): 如果为 True，保存特征图用于可视化。
            txt_feats (torch.Tensor, optional): 文本特征，如果给定则使用它。
            augment (bool): 如果为 True，推理期间执行数据增强。
            embed (list, optional): 返回嵌入的层索引列表。

        Returns:
            (torch.Tensor): 模型的输出张量。
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # 除头部外
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # 运行

            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """计算损失。

        Args:
            batch (dict): 要计算损失的批次。
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE 检测模型。

    该类实现了 YOLOE 架构，用于高效的文本和视觉提示目标检测，支持
    基于提示和无提示两种推理模式。

    Attributes:
        pe (torch.Tensor): 类别的提示嵌入。
        clip_model (torch.nn.Module): 用于文本编码的 CLIP 模型。

    Methods:
        __init__: 初始化 YOLOE 模型。
        get_text_pe: 获取文本位置嵌入。
        get_visual_pe: 获取视觉嵌入。
        set_vocab: 为无提示模型设置词汇表。
        get_vocab: 获取融合的词汇层。
        set_classes: 设置离线推理的类别。
        get_cls_pe: 获取类别位置嵌入。
        predict: 使用提示执行前向传播。
        loss: 使用提示计算损失。

    Examples:
        初始化 YOLOE 模型
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOE 模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """使用 CLIP 模型获取文本位置嵌入。

        Args:
            text (list[str]): 类别名称列表。
            batch (int): 处理文本标记的批次大小。
            cache_clip_model (bool): 是否缓存 CLIP 模型。
            without_reprta (bool): 是否返回未经 reprta 模块处理的文本嵌入。

        Returns:
            (torch.Tensor): 文本位置嵌入。
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # 为了向后兼容缺少 clip_model 属性的模型
            self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)

        model = (
            self.clip_model
            if cache_clip_model
            else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
        )
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # 运行辅助文本头

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """获取视觉位置嵌入。

        Args:
            img (torch.Tensor): 输入图像张量。
            visual (torch.Tensor): 视觉特征。

        Returns:
            (torch.Tensor): 视觉位置嵌入。
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """为无提示模型设置词汇表。

        Args:
            vocab (nn.ModuleList): 词汇项列表。
            names (list[str]): 类别名称列表。
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # 为头部缓存锚点
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # 预热

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        cv2 = getattr(head, "one2one_cv2", head.cv2)

        # 无提示模型的重参数化
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """从模型中获取融合的词汇层。

        Args:
            names (list[str]): 类别名称列表。

        Returns:
            (nn.ModuleList): 词汇模块列表。
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # 将提示嵌入融合到分类头

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        vocab = nn.ModuleList()
        for cls_head in cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """提前设置类别，使模型可以在没有 clip 模型的情况下进行离线推理。

        Args:
            names (list[str]): 类别名称列表。
            embeddings (torch.Tensor): 嵌入张量。
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "无提示模型不支持设置类别。请尝试使用文本/视觉提示模型。"
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """获取类别位置嵌入。

        Args:
            tpe (torch.Tensor | None): 文本位置嵌入。
            vpe (torch.Tensor | None): 视觉位置嵌入。

        Returns:
            (torch.Tensor): 类别位置嵌入。
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """执行模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            profile (bool): 如果为 True，分析每一层的计算时间。
            visualize (bool): 如果为 True，保存特征图用于可视化。
            tpe (torch.Tensor, optional): 文本位置嵌入。
            augment (bool): 如果为 True，推理期间执行数据增强。
            embed (list, optional): 返回嵌入的层索引列表。
            vpe (torch.Tensor, optional): 视觉位置嵌入。
            return_vpe (bool): 如果为 True，返回视觉位置嵌入。

        Returns:
            (torch.Tensor): 模型的输出张量。
        """
        y, dt, embeddings = [], [], []  # 输出
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # 除头部外
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x.append(cls_pe)  # 添加类别嵌入
            x = m(x)  # 运行

            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """计算损失。

        Args:
            batch (dict): 要计算损失的批次。
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果。
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))
                if visual_prompt
                else self.init_criterion()
            )
        if preds is None:
            preds = self.forward(
                batch["img"],
                tpe=None if "visuals" in batch else batch.get("txt_feats", None),
                vpe=batch.get("visuals", None),
            )
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE 分割模型。

    该类扩展了 YOLOEModel，用于处理带有文本和视觉提示的实例分割任务，为像素级目标检测和分割提供专门的损失计算。

    Methods:
        __init__: 初始化 YOLOE 分割模型。
        loss: 使用提示计算分割损失。

    Examples:
        初始化 YOLOE 分割模型
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOE 分割模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数量。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """计算损失。

        Args:
            batch (dict): 要计算损失的批次。
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果。
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))
                if visual_prompt
                else self.init_criterion()
            )

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """模型集成。

    该类允许将多个 YOLO 模型组合成集成，通过模型平均
    或其他集成技术来提高性能。

    Methods:
        __init__: 初始化模型集成。
        forward: 从集成中的所有模型生成预测。

    Examples:
        创建模型集成
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """初始化模型集成。"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """运行集成前向传播并连接所有模型的预测。

        Args:
            x (torch.Tensor): 输入张量。
            augment (bool): 是否对输入进行增强。
            profile (bool): 是否分析模型。
            visualize (bool): 是否可视化特征。

        Returns:
            (torch.Tensor): 所有模型连接后的预测。
            (None): 集成推理始终返回 None。
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # 最大集成
        # y = torch.stack(y).mean(0)  # 平均集成
        y = torch.cat(y, 2)  # nms 集成，y 形状(B, HW, C*num_models)
        return y, None  # 推理，训练输出


# 函数 ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """用于临时添加或修改 Python 模块缓存 (`sys.modules`) 中模块的上下文管理器。

    该函数可用于在运行时更改模块路径。在重构代码时很有用，当你
    将模块从一个位置移动到另一个位置，但仍希望支持旧的导入路径以进行向后
    兼容。

    Args:
        modules (dict, optional): 将旧模块路径映射到新模块路径的字典。
        attributes (dict, optional): 将旧模块属性映射到新模块属性的字典。

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # 现在将导入 new.module
        >>> from old.module import attribute  # 现在将导入 new.module.attribute

    Notes:
        更改仅在上下文管理器内部生效，并在上下文管理器退出后撤销。
        请注意，直接操作 `sys.modules` 可能导致不可预测的结果，尤其是在较大的
        应用程序或库中。请谨慎使用此函数。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # 在 sys.modules 中以旧名称设置属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # 在 sys.modules 中以旧名称设置模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # 移除临时模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """用于在反序列化期间替换未知类的占位类。"""

    def __init__(self, *args, **kwargs):
        """初始化 SafeClass 实例，忽略所有参数。"""
        pass

    def __call__(self, *args, **kwargs):
        """运行 SafeClass 实例，忽略所有参数。"""
        pass


class SafeUnpickler(pickle.Unpickler):
    """自定义反序列化器，将未知类替换为 SafeClass。"""

    def find_class(self, module, name):
        """尝试查找类，如果不在安全模块中则返回 SafeClass。

        Args:
            module (str): 模块名称。
            name (str): 类名称。

        Returns:
            (type): 找到的类或 SafeClass。
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # 添加其他被认为安全的模块
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """尝试使用 torch.load() 函数加载 PyTorch 模型。如果引发 ModuleNotFoundError，则捕获
    错误，记录警告消息，并尝试通过 check_requirements()
    函数安装缺失的模块。安装后，该函数再次尝试使用 torch.load() 加载模型。

    Args:
        weight (str | Path): PyTorch 模型的文件路径。
        safe_only (bool): 如果为 True，加载期间将未知类替换为 SafeClass。

    Returns:
        (dict): 加载的模型检查点。
        (str): 加载的文件名。

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # 如果本地缺失则在线搜索

    def _load():
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
                # 解决跨平台 pathlib pickle 不兼容问题
                **(
                    {"pathlib.PosixPath": "pathlib.WindowsPath"}
                    if WINDOWS
                    else {"pathlib.WindowsPath": "pathlib.PosixPath"}
                ),
            },
        ):
            if safe_only:
                # 通过自定义 pickle 模块加载
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    return torch_load(f, pickle_module=safe_pickle)
            return torch_load(file, map_location="cpu")

    try:
        ckpt = _load()

    except RuntimeError as e:
        # 下载的权重损坏（例如截断）；跳过用户提供的本地路径以避免破坏性删除。
        if "PytorchStreamReader" not in str(e) or Path(str(weight)).exists():
            raise
        LOGGER.warning(f"缓存损坏 {file}，重新下载 {weight}...")
        Path(file).unlink(missing_ok=True)
        file = attempt_download_asset(weight)
        ckpt = _load()

    except ModuleNotFoundError as e:  # e.name 是缺失的模块名称
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"错误 ❌️ {weight} 似乎是一个最初使用 "
                    f"https://github.com/ultralytics/yolov5 训练的 Ultralytics YOLOv5 模型。\n该模型与 "
                    f"https://github.com/ultralytics/ultralytics 上的 YOLOv8 不向前兼容。"
                    f"\n建议的修复方法是使用最新的 'ultralytics' 包训练新模型，或"
                    f"运行带有官方 Ultralytics 模型的命令，例如 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"错误 ❌️ {weight} 需要 numpy>=1.26.1，但当前安装的是 numpy=={__import__('numpy').__version__}。"
                )
            ) from e
        elif e.name and e.name.startswith("ultralytics."):
            raise ModuleNotFoundError(
                emojis(
                    f"错误 ❌️ {weight} 需要缺失的 Ultralytics 模块 '{e.name}'。"
                    "使用最新的 'ultralytics' 包训练新模型，或运行带有官方 "
                    "Ultralytics 模型的命令，例如 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"{weight} 似乎需要 '{e.name}'，但该模块不在 Ultralytics 要求中。"
            f"\n现在将为 '{e.name}' 运行自动安装，但此功能将在未来移除。"
            f"\n建议的修复方法是使用最新的 'ultralytics' 包训练新模型，或"
            f"运行带有官方 Ultralytics 模型的命令，例如 'yolo predict model=yolo26n.pt'"
        )
        check_requirements(e.name)  # 安装缺失的模块
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # 文件可能是使用 torch.save(model, "saved_model.pt") 保存的 YOLO 实例
        LOGGER.warning(
            f"文件 '{weight}' 似乎保存或格式不正确。"
            f"为了获得最佳结果，请使用 model.save('filename.pt') 正确保存 YOLO 模型。"
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """加载单个模型权重。

    Args:
        weight (str | Path): 模型权重路径。
        device (torch.device, optional): 加载模型的设备。
        inplace (bool): 是否执行原地操作。
        fuse (bool): 是否融合模型。

    Returns:
        (torch.nn.Module): 加载的模型。
        (dict): 模型检查点字典。
    """
    if str(weight).lower().startswith(REMOTE_FILE_PREFIXES):
        weight = check_file(weight, download_dir=SETTINGS["weights_dir"])
    ckpt, weight = torch_safe_load(weight)  # 加载检查点
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # 合并模型和默认参数，优先使用模型参数
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 模型

    # 模型兼容性更新
    model.args = args  # 将参数附加到模型
    model.pt_path = str(weight)  # 将 *.pt 文件路径作为字符串附加到模型（避免 WindowsPath pickle 问题）
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # 模型设置为评估模式

    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 兼容性

    # 返回模型和检查点
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """将 YOLO model.yaml 字典解析为 PyTorch 模型。

    Args:
        d (dict): 模型字典。
        ch (int): 输入通道数。
        verbose (bool): 是否打印模型详细信息。

    Returns:
        (torch.nn.Sequential): PyTorch 模型。
        (list): 需要保存输出的层索引的有序列表。
    """
    import ast

    # 参数
    legacy = True  # v3/v5/v8/v9 模型的向后兼容性
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"未传递模型尺度。假设 scale='{scale}'。")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数，例如 Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # 层，保存列表，输出通道
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # 带有 'repeat' 参数的模块
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # 获取模块
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度增益
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # 如果 c2 != nc（例如，Classify() 输出）
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # 设置 1) 嵌入通道数 和 2) 头数
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # 重复次数
                n = 1
            if m is C3k2:  # 用于 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # 用于 L/X 尺寸
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # 重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])  # 通道作为第二个参数
        elif m is RTDETRDecoder:  # 特殊情况，通道参数必须在索引 1 处传递
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引、'from' 索引、类型
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 追加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 YAML 文件加载 YOLO 模型。

    Args:
        path (str | Path): YAML 文件路径。

    Returns:
        (dict): 模型字典。
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 模型现在使用 -p6 后缀。将 {path.stem} 重命名为 {new_stem}。")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # 例如 yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # 模型字典
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """从模型路径中提取模型尺度的尺寸字符 n、s、m、l 或 x。

    Args:
        model_path (str | Path): YOLO 模型 YAML 文件的路径。

    Returns:
        (str): 模型尺度的尺寸字符（n、s、m、l 或 x），如果未找到则返回空字符串。
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """从 PyTorch 模型的架构或配置中猜测任务。

    Args:
        model (torch.nn.Module | dict | str | Path): PyTorch 模型、模型配置字典或模型文件路径。

    Returns:
        (str): 模型的任务（'detect'、'segment'、'classify'、'pose'、'obb'）。
    """

    def cfg2task(cfg):
        """从 YAML 字典猜测。"""
        m = cfg["head"][-1][-2].lower()  # 输出模块名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    # 从模型配置猜测
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # 从 PyTorch 模型猜测
    if isinstance(model, torch.nn.Module):  # PyTorch 模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # nosec B307: 安全地评估已知属性路径
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307: 安全地评估已知属性路径
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # 从模型文件名猜测
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # 无法从模型确定任务
    LOGGER.warning(
        "无法自动猜测模型任务，假设 'task=detect'。"
        "请显式定义模型的任务，例如 'task=detect'、'segment'、'classify'、'pose' 或 'obb'。"
    )
    return "detect"  # 假设为检测
