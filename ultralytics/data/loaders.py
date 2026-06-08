# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import time
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread


@dataclass
class SourceTypes:
    """用于表示各种预测输入源类型的类。

    This class uses dataclass to define boolean flags for different types of input sources that can be used for making
    predictions with YOLO models.

    属性：
        stream (bool): Flag indicating if the input source is a video stream.
        screenshot (bool): Flag indicating if the input source is a screenshot.
        from_img (bool): Flag indicating if the input source is an image file.
        tensor (bool): Flag indicating if the input source is a tensor.

    示例：
        >>> source_types = SourceTypes(stream=True, screenshot=False, from_img=False)
        >>> print(source_types.stream)
        True
        >>> print(source_types.from_img)
        False
    """

    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """用于各种类型视频流的流加载器。

    Supports RTSP, RTMP, HTTP, and TCP streams. This class handles the loading and processing of multiple video streams
    simultaneously, making it suitable for real-time video analysis tasks.

    属性：
        sources (list[str]): The source input paths or URLs for the video streams.
        vid_stride (int): Video frame-rate stride.
        buffer (bool): Whether to buffer input streams.
        running (bool): Flag to indicate if the streaming thread is running.
        mode (str): Set to 'stream' indicating real-time capture.
        imgs (list[list[np.ndarray]]): List of image frames for each stream.
        fps (list[float]): List of FPS for each stream.
        frames (list[int]): List of total frames for each stream.
        threads (list[Thread]): List of threads for each stream.
        shape (list[tuple[int, int, int]]): List of shapes for each stream.
        caps (list[cv2.VideoCapture]): List of cv2.VideoCapture objects for each stream.
        bs (int): Batch size for processing.
        cv2_flag (int): OpenCV flag for image reading (grayscale or color/BGR).

    方法：
        update: Read stream frames in daemon thread.
        close: Close stream loader and release resources.
        __iter__: Returns an iterator object for the class.
        __next__: Returns source paths, transformed, and original images for processing.
        __len__: Return the length of the sources object.

    示例：
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # 处理图像
        ...     pass
        >>> stream_loader.close()

    注意：
        - The class uses threading to efficiently load frames from multiple streams simultaneously.
        - It automatically handles YouTube links, converting them to the best available stream URL.
        - The class implements a buffer system to manage frame storage and retrieval.
    """

    def __init__(self, sources: str = "file.streams", vid_stride: int = 1, buffer: bool = False, channels: int = 3):
        """初始化多视频源的流加载器，支持各种流类型。

        参数：
            sources (str): Path to streams file or single stream URL.
            vid_stride (int): Video frame-rate stride.
            buffer (bool): Whether to buffer input streams.
            channels (int): Number of image channels (1 for grayscale, 3 for color).
        """
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # grayscale or color (BGR)

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # video capture objects
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [ops.clean_str(x).replace(os.sep, "_") for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # 启动线程从视频流读取帧
            st = f"{i + 1}/{n}: {s}... "
            if urllib.parse.urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:  # YouTube 视频
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Jsn8D3aC840' or 'https://youtu.be/Jsn8D3aC840'
                s = get_best_youtube_url(s)
            s = int(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "Try running 'source=0' in a local environment."
                )
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, im = self.caps[i].read()  # guarantee first frame
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # newline

    def update(self, i: int, cap: cv2.VideoCapture, stream: str):
        """在守护线程中读取流帧并更新图像缓冲区。"""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    im = (
                        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
                    )
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """终止流加载器，停止线程并释放视频捕获资源。"""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # 添加超时
        for cap in self.caps:  # 遍历存储的 VideoCapture 对象
            try:
                cap.release()  # release video capture
            except Exception as e:
                LOGGER.warning(f"Could not release VideoCapture object: {e}")

    def __iter__(self):
        """返回迭代器对象并重置帧计数器。"""
        self.count = -1
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回来自多个视频流的下一批帧用于处理。"""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # 等待每个缓冲区中有可用帧
            while not x:
                if not self.threads[i].is_alive():
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"Waiting for stream {i}")

            # 获取并移除 imgs 缓冲区中的第一帧
            if self.buffer:
                images.append(x.pop(0))

            # 获取最后一帧，并清空 imgs 缓冲区中的其余帧
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self) -> int:
        """返回 LoadStreams 对象中视频流的数量。"""
        return self.bs  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    """用于捕获和处理屏幕图像的 Ultralytics 截图 DataLoader。

    This class manages the loading of screenshot images for processing with YOLO. It is suitable for use with `yolo
    predict source=screen`.

    属性：
        screen (int): The screen number to capture.
        left (int): The left coordinate for screen capture area.
        top (int): The top coordinate for screen capture area.
        width (int): The width of the screen capture area.
        height (int): The height of the screen capture area.
        mode (str): Set to 'stream' indicating real-time capture.
        frame (int): Counter for captured frames.
        sct (mss.mss): Screen capture object from `mss` library.
        bs (int): Batch size, set to 1.
        fps (int): Frames per second, set to 30.
        monitor (dict[str, int]): Monitor configuration details.
        cv2_flag (int): OpenCV flag for image reading (grayscale or color/BGR).

    方法：
        __iter__: Returns an iterator object.
        __next__: Captures the next screenshot and returns it.

    示例：
        >>> loader = LoadScreenshots("0 100 100 640 480")  # screen 0, top-left (100,100), 640x480
        >>> for sources, imgs, info in loader:
        ...     print(f"Captured frame: {imgs[0].shape}")
    """

    def __init__(self, source: str, channels: int = 3):
        """使用指定的屏幕和区域参数初始化截图捕获。

        参数：
            source (str): Screen capture source string in format "screen_num left top width height".
            channels (int): Number of image channels (1 for grayscale, 3 for color).
        """
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # grayscale or color (BGR)

        # 解析显示器形状
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """返回截图捕获的迭代器对象。"""
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """使用 mss 库捕获并返回下一张截图作为 numpy 数组。"""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # BGRA 转 BGR
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im0
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        self.frame += 1
        return [str(self.screen)], [im0], [s]  # screen, img, string


class LoadImagesAndVideos:
    """用于加载和处理 YOLO 目标检测中图像和视频的类。

    This class manages the loading and pre-processing of image and video data from various sources, including single
    image files, video files, and lists of image and video paths.

    属性：
        files (list[str]): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (list[bool]): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate.
        bs (int): Batch size.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during __iter__().
        ni (int): Number of images.
        cv2_flag (int): OpenCV flag for image reading (grayscale or color/BGR).

    方法：
        __init__: Initialize the LoadImagesAndVideos object.
        __iter__: Returns an iterator object for VideoStream or ImageFolder.
        __next__: Returns the next batch of images or video frames along with their paths and metadata.
        _new_video: Creates a new video capture object for the given path.
        __len__: Returns the number of batches in the object.

    示例：
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # 处理图像或视频帧批次
        ...     pass

    注意：
        - Supports various image formats including HEIC.
        - Handles both local files and directories.
        - Can read from a text file containing paths to images and videos.
    """

    def __init__(self, path: str | Path | list, batch: int = 1, vid_stride: int = 1, channels: int = 3):
        """初始化图像和视频的 DataLoader，支持各种输入格式。

        参数：
            path (str | Path | list): Path to images/videos, directory, or list of paths.
            batch (int): Batch size for processing.
            vid_stride (int): Video frame-rate stride.
            channels (int): Number of image channels (1 for grayscale, 3 for color).
        """
        parent = None
        if isinstance(path, str) and Path(path).suffix in {".txt", ".csv"}:  # txt/csv file with source paths
            parent, content = Path(path).parent, Path(path).read_text()
            path = content.splitlines() if Path(path).suffix == ".txt" else content.split(",")  # list of sources
            path = [p.strip() for p in path]
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(glob.escape(a), "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # 将文件定义为图像或视频
        images, videos = [], []
        for f in files:
            suffix = f.rpartition(".")[-1].lower()  # 获取不带点的小写文件扩展名
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"  # default to video if no images
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # grayscale or color (BGR)
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """遍历图像/视频文件，产出源路径、图像和元数据。"""
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回下一批图像或视频帧及其路径和元数据。"""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if imgs:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # end of video or failure

                if success:
                    success, im0 = self.cap.retrieve()
                    im0 = (
                        cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None]
                        if self.cv2_flag == cv2.IMREAD_GRAYSCALE
                        else im0
                    )
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # end of video
                            self.count += 1
                            self.cap.release()
                else:
                    # 如果当前视频结束或无法打开，则移到下一个文件
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                # 处理图像文件
                self.mode = "image"
                im0 = imread(path, flags=self.cv2_flag)  # BGR
                if im0 is None:
                    LOGGER.warning(f"Image Read Error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni:  # end of image list
                    break

        return paths, imgs, info

    def _new_video(self, path: str):
        """为给定路径创建新的视频捕获对象并初始化视频相关属性。"""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self) -> int:
        """返回数据集中的批次数量。"""
        return math.ceil(self.nf / self.bs)  # number of batches


class LoadPilAndNumpy:
    """加载 PIL 和 Numpy 数组中的图像用于批处理。

    This class manages loading and pre-processing of image data from both PIL and Numpy formats. It performs basic
    validation and format conversion to ensure that the images are in the required format for downstream processing.

    属性：
        paths (list[str]): List of image paths or autogenerated filenames.
        im0 (list[np.ndarray]): List of images stored as Numpy arrays.
        mode (str): Type of data being processed, set to 'image'.
        bs (int): Batch size, equivalent to the length of `im0`.

    方法：
        _single_check: Validate and format a single image to a Numpy array.

    示例：
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"Loaded {len(images)} images")
        Loaded 2 images
    """

    def __init__(self, im0: Image.Image | np.ndarray | list, channels: int = 3):
        """初始化 PIL 和 Numpy 图像加载器，将输入转换为标准化格式。

        参数：
            im0 (PIL.Image.Image | np.ndarray | list): Single image or list of images in PIL or numpy format.
            channels (int): Number of image channels (1 for grayscale, 3 for color).
        """
        if not isinstance(im0, list):
            im0 = [im0]
        # use `image{i}.jpg` when Image.filename returns an empty path.
        self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
        pil_flag = "L" if channels == 1 else "RGB"  # grayscale or RGB
        self.im0 = [self._single_check(im, pil_flag) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)
        self.count = 0

    @staticmethod
    def _single_check(im: Image.Image | np.ndarray, flag: str = "RGB") -> np.ndarray:
        """验证并将图像格式化为 NumPy 数组。

        注意：
            - PIL inputs are converted to NumPy and returned in OpenCV-compatible BGR order for color images.
            - NumPy inputs are returned as-is (no channel-order conversion is applied).
        """
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im.convert(flag))
            # 如果是灰度图则添加新轴；将 RGB 转换为 BGR 以兼容 OpenCV。
            im = im[..., None] if flag == "L" else im[..., ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        elif im.ndim == 2:  # grayscale in numpy form
            im = im[..., None]
        return im

    def __len__(self) -> int:
        """返回 'im0' 属性的长度，表示已加载的图像数量。"""
        return len(self.im0)

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回下一批图像、路径和元数据用于处理。"""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        """遍历 PIL/numpy 图像，产出路径、原始图像和元数据用于处理。"""
        self.count = 0
        return self


class LoadTensor:
    """用于加载和处理目标检测任务中张量数据的类。

    This class handles the loading and pre-processing of image data from PyTorch tensors, preparing them for further
    processing in object detection pipelines.

    属性：
        im0 (torch.Tensor): The input tensor containing the image(s) with shape (B, C, H, W).
        bs (int): Batch size, inferred from the shape of `im0`.
        mode (str): Current processing mode, set to 'image'.
        paths (list[str]): List of image paths or auto-generated filenames.

    方法：
        _single_check: Validates and formats an input tensor.

    示例：
        >>> import torch
        >>> tensor = torch.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"Processed {len(images)} images")
    """

    def __init__(self, im0: torch.Tensor) -> None:
        """初始化 LoadTensor 对象，用于处理 torch.Tensor 图像数据。

        参数：
            im0 (torch.Tensor): Input tensor with shape (B, C, H, W).
        """
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [f"image{i}.jpg" for i in range(self.bs)]
        self.count = 0

    @staticmethod
    def _single_check(im: torch.Tensor, stride: int = 32) -> torch.Tensor:
        """验证并格式化单个图像张量，确保形状和归一化正确。"""
        s = (
            f"torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            LOGGER.warning(
                f"torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. Dividing input by 255."
            )
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """产出用于遍历张量图像数据的迭代器对象。"""
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], torch.Tensor, list[str]]:
        """产出下一批张量图像和元数据用于处理。"""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self) -> int:
        """返回张量输入的批次大小。"""
        return self.bs


def autocast_list(source: list[Any]) -> list[Image.Image | np.ndarray]:
    """将源列表转换为 numpy 数组或 PIL 图像列表用于 Ultralytics 预测。"""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(
                ImageOps.exif_transpose(Image.open(urllib.request.urlopen(im) if str(im).startswith("http") else im))
            )
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL 或 numpy 图像
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \n"
                f"See https://docs.ultralytics.com/modes/predict for supported source types."
            )

    return files


def get_best_youtube_url(url: str, method: str = "pytube") -> str | None:
    """从给定的 YouTube 视频中获取最佳质量 MP4 视频流的 URL。

    参数：
        url (str): The URL of the YouTube video.
        method (str): The method to use for extracting video info. Options are "pytube", "pafy", and "yt-dlp".

    返回：
        (str | None): The URL of the best quality MP4 video stream, or None if no suitable stream is found.

    示例：
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    注意：
        - Requires additional libraries based on the chosen method: pytubefix, pafy, or yt-dlp.
        - The function prioritizes streams with at least 1080p resolution when available.
        - For the "yt-dlp" method, it looks for formats with video codec, no audio, and *.mp4 extension.
    """
    if method == "pytube":
        # Switched from pytube to pytubefix to resolve https://github.com/pytube/pytube/issues/1954
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)  # sort streams by resolution
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:  # check if resolution is at least 1080p
                return stream.url

    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy

        return pafy.new(url).getbestvideo(preftype="mp4").url

    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # extract info
        for f in reversed(info_dict.get("formats", [])):  # reversed because best is usually last
            # 查找带视频编码、无音频、*.mp4 扩展名、至少 1920x1080 尺寸的格式
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")


# 定义常量
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)
