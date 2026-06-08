# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
对图像、视频、目录、glob、YouTube、网络摄像头、流等进行预测。

Usage - sources:
    $ yolo mode=predict model=yolo26n.pt source=0                               # 网络摄像头
                                                img.jpg                         # 图像
                                                vid.mp4                         # 视频
                                                screen                          # 截屏
                                                path/                           # 目录
                                                list.txt                        # 图像列表
                                                list.streams                    # 流列表
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP 流

Usage - formats:
    $ yolo mode=predict model=yolo26n.pt                 # PyTorch
                              yolo26n.torchscript        # TorchScript
                              yolo26n.onnx               # ONNX Runtime 或带 dnn=True 的 OpenCV DNN
                              yolo26n_openvino_model     # OpenVINO
                              yolo26n.engine             # TensorRT
                              yolo26n.mlpackage          # CoreML (仅限 macOS)
                              yolo26n_saved_model        # TensorFlow SavedModel
                              yolo26n.pb                 # TensorFlow GraphDef
                              yolo26n.tflite             # TensorFlow Lite
                              yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo26n_paddle_model       # PaddlePaddle
                              yolo26n.mnn                # MNN
                              yolo26n_ncnn_model         # NCNN
                              yolo26n_imx_model          # Sony IMX
                              yolo26n_rknn_model         # Rockchip RKNN
                              yolo26n_executorch_model   # PyTorch Executorch
                              yolo26n_axelera_model      # Axelera AI
                              yolo26n_deepx_model        # DeepX
"""

from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode

STREAM_WARNING = """
除非传入 `stream=True`，否则推理结果会累积在 RAM 中，这可能导致大型数据源或长时间运行的流和视频出现内存不足错误。
参见 https://docs.ultralytics.com/modes/predict/ 获取帮助。

Example:
    results = model(source=..., stream=True)  # Results 对象的生成器
    for r in results:
        boxes = r.boxes  # 用于边界框输出的 Boxes 对象
        masks = r.masks  # 用于分割掩码输出的 Masks 对象
        probs = r.probs  # 用于分类输出的类别概率
"""


class BasePredictor:
    """创建预测器的基类。

    此类为预测功能提供基础，处理模型设置、推理和跨各种输入源的结果处理。

    Attributes:
        args (SimpleNamespace): 预测器的配置。
        save_dir (Path): 保存结果的目录。
        done_warmup (bool): 预测器是否已完成设置。
        model (torch.nn.Module): 用于预测的模型。
        data (str): 数据配置。
        device (torch.device): 用于预测的设备。
        dataset (Dataset): 用于预测的数据集。
        vid_writer (dict[Path, cv2.VideoWriter]): {保存路径: 视频写入器} 字典，用于保存视频输出。
        plotted_img (np.ndarray): 最后绘制的图像。
        source_type (SimpleNamespace): 输入源的类型。
        seen (int): 已处理的图像数量。
        windows (list[str]): 可视化窗口名称列表。
        batch (tuple): 当前批次数据。
        results (list[Any]): 当前批次结果。
        transforms (Callable): 分类的图像变换。
        callbacks (dict[str, list[Callable]]): 不同事件的回调函数。
        txt_path (Path): 保存文本结果的路径。
        _lock (threading.Lock): 线程安全推理的锁。

    Methods:
        preprocess: 推理前准备输入图像。
        inference: 对给定图像运行推理。
        postprocess: 将原始预测处理为结构化结果。
        predict_cli: 为命令行界面运行预测。
        setup_source: 设置输入源和推理模式。
        stream_inference: 对输入源进行流式推理。
        setup_model: 初始化并配置模型。
        write_results: 将推理结果写入文件。
        save_predicted_images: 保存预测可视化图像。
        show: 在窗口中显示结果。
        run_callbacks: 执行事件注册的回调。
        add_callback: 注册新的回调函数。
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        """初始化 BasePredictor 类。

        Args:
            cfg (str | Path | dict | SimpleNamespace): 配置文件路径或配置字典。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # 默认 conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # 设置完成后方可使用
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # {save_path: video_writer, ...} 字典
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # 用于自动线程安全推理
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """推理前准备输入图像。

        Args:
            im (torch.Tensor | list[np.ndarray]): 张量格式为 (N, 3, H, W) 的图像，列表格式为 [(H, W, 3) x N]。

        Returns:
            (torch.Tensor): 形状为 (N, 3, H, W) 的预处理图像张量。
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR 转 RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC 转 BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # 连续
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 转 fp16/32
        if not_tensor:
            im /= 255  # 0-255 转 0.0-1.0
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """使用指定模型和参数对给定图像运行推理。"""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """推理前预变换输入图像。

        Args:
            im (list[np.ndarray]): 形状为 [(H, W, 3) x N] 的图像列表。

        Returns:
            (list[np.ndarray]): 变换后的图像列表。
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """对图像的后处理预测并返回。"""
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """对图像或流执行推理。

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                推理的源数据。
            model (str | Path | torch.nn.Module, optional): 用于推理的模型。
            stream (bool): 是否流式传输推理结果。如果为 True，返回生成器。
            *args (Any): 推理方法的额外参数。
            **kwargs (Any): 推理方法的额外关键字参数。

        Returns:
            (list[ultralytics.engine.results.Results] | generator): Results 对象或 Results 对象的生成器。
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # 将 Results 列表合并为一个

    def predict_cli(self, source=None, model=None):
        """用于命令行界面（CLI）预测的方法。

        此函数旨在使用 CLI 运行预测。它设置源和模型，然后以流式方式处理输入。
        此方法通过消费生成器而不存储结果来确保没有输出累积在内存中。

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                推理的源数据。
            model (str | Path | torch.nn.Module, optional): 用于推理的模型。

        Notes:
            不要修改此函数或移除生成器。生成器确保没有输出累积在内存中，
            这对于防止长时间运行预测期间的内存问题至关重要。
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source, stride: int | None = None):
        """设置源和推理模式。

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): 推理的源数据。
            stride (int, optional): 用于图像尺寸检查的模型步长。
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)  # 检查图像尺寸
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # 大量图像
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # 长序列
            import torchvision  # noqa (在此导入以触发 torchvision NMS 在 nms.py 中的使用)

            if not getattr(self, "stream", True):  # 视频
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """对输入源进行流式推理并将结果保存到文件。

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                推理的源数据。
            model (str | Path | torch.nn.Module, optional): 用于推理的模型。
            *args (Any): 推理方法的额外参数。
            **kwargs (Any): 推理方法的额外关键字参数。

        Yields:
            (ultralytics.engine.results.Results): Results 对象。
        """
        if self.args.verbose:
            LOGGER.info("")

        # 设置模型
        if not self.model:
            self.setup_model(model)

        with self._lock:  # 用于线程安全推理
            # 每次调用 predict 时设置源
            self.setup_source(source if source is not None else self.args.source)

            # 检查 save_dir/labels 文件是否存在
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 预热模型
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.format in {"pt", "triton"} else self.dataset.bs,
                        self.model.channels,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # 预处理
                with profilers[0]:
                    im = self.preprocess(im0s)

                # 推理
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # 产出嵌入张量
                        continue

                # 后处理
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # 可视化、保存、写入结果
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # 打印批次结果
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # 释放资源
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # 关闭所有打开的窗口

        # 打印最终结果
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # 每张图像的速度
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # 标签数量
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """使用给定参数初始化 YOLO 模型并将其设置为评估模式。

        Args:
            model (str | Path | torch.nn.Module): 要加载或使用的模型。
            verbose (bool): 是否打印详细输出。
        """
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if model.end2end:
                model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
        self.model = AutoBackend(
            model=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # 更新设备
        self.args.half = self.model.fp16  # 更新 half
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # 重用导出元数据中的 imgsz
        self.model.eval()
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """将推理结果写入文件或目录。

        Args:
            i (int): 批次中当前图像的索引。
            p (Path): 当前图像的路径。
            im (torch.Tensor): 预处理后的图像张量。
            s (list[str]): 结果字符串列表。

        Returns:
            (str): 包含结果信息的字符串。
        """
        string = ""  # 打印字符串
        if len(im.shape) == 3:
            im = im[None]  # 为批次维度扩展
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 如果帧未确定则为 0

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # 在其他位置使用
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # 将预测添加到图像
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # 保存结果
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """将视频预测保存为 mp4/avi 或将图像保存为 jpg 到指定路径。

        Args:
            save_path (Path): 保存结果的路径。
            frame (int): 视频模式的帧号。
        """
        im = self.plotted_img

        # 保存视频和流
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"  # 将帧保存到单独的目录
            if save_path not in self.vid_writer:  # 新视频
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # 需要整数，浮点数在 MP4 编解码器中会产生错误
                    frameSize=(im.shape[1], im.shape[0]),  # (宽, 高)
                )

            # 保存视频
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # 保存图像
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # 保存为 JPG 以获得最佳支持

    def show(self, p: str = ""):
        """在窗口中显示图像。"""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小 (Linux)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (宽, 高)
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):  # 图像 300ms；否则 1ms
            raise StopIteration

    def run_callbacks(self, event: str):
        """运行特定事件的所有已注册回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable):
        """为特定事件添加回调函数。"""
        self.callbacks[event].append(func)
