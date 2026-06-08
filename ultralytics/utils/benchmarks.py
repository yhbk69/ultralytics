# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO 模型基准测试，对不同格式进行速度和精度评估。

用法:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolo26n.yaml', 'yolov8s.yaml']).run()
    benchmark(model='yolo26n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Lite         | `tflite`                  | yolo26n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo26n_web_model/
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.cuda

from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.nn.modules import Segment26
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    IS_DOCKER,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    TQDM,
    WEIGHTS_DIR,
    YAML,
)
from ultralytics.utils.checks import IS_PYTHON_3_13, check_imgsz, check_requirements, check_yolo, is_rockchip
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo26n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=1e-3,
    format="",
    **kwargs,
):
    """对不同格式的 YOLO 模型进行速度和精度基准测试。

    参数:
        model (str | Path): 模型文件或目录路径。
        data (str | None): 评估数据集，未传入时从 TASK2DATA 继承。
        imgsz (int): 基准测试的图像大小。
        half (bool): 为 True 时使用半精度。
        int8 (bool): 为 True 时使用 int8 精度。
        device (str): 运行基准测试的设备，'cpu' 或 'cuda'。
        verbose (bool | float): 如果为 True 或浮点数，断言基准测试通过给定指标。
        eps (float): 防止除零的 Epsilon 值。
        format (str): 基准测试的导出格式。未提供时测试所有格式。
        **kwargs (Any): 导出器的额外关键字参数。

    返回:
        (polars.DataFrame): 包含每种格式基准测试结果的 Polars DataFrame，包括文件大小、指标和推理时间。

    示例:
        使用默认设置进行 YOLO 模型基准测试:
        >>> from ultralytics.utils.benchmarks import benchmark
        >>> benchmark(model="yolo26n.pt", imgsz=640)
    """
    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import polars as pl  # 限定作用域以加速 'import ultralytics'

    pl.Config.set_tbl_cols(-1)  # 显示所有列
    pl.Config.set_tbl_rows(-1)  # 显示所有行
    pl.Config.set_tbl_width_chars(-1)  # 无宽度限制
    pl.Config.set_tbl_hide_column_data_types(True)  # 隐藏数据类型
    pl.Config.set_tbl_hide_dataframe_shape(True)  # 隐藏形状信息
    pl.Config.set_tbl_formatting("ASCII_BORDERS_ONLY_CONDENSED")

    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    data = data or TASK2DATA[model.task]  # 任务对应数据集，如 task=detect 对应 coco8.yaml
    key = TASK2METRIC[model.task]  # 任务对应指标，如 task=detect 对应 metrics/mAP50-95(B)

    y = []
    t0 = time.time()

    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert format in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."
    for name, format, suffix, cpu, gpu, valid_args in zip(*export_formats().values()):
        emoji, filename = "❌", None  # 导出默认值
        try:
            if format_arg and format_arg != format:
                continue

            # 检查
            if format == "pb":
                assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"
            elif format == "edgetpu":
                assert LINUX and not ARM64, "Edge TPU export only supported on non-aarch64 Linux"
                assert shutil.which("edgetpu_compiler"), "Edge TPU benchmark requires edgetpu_compiler"
            elif format == "tfjs":
                assert not (LINUX and ARM64), "TF.js export not supported on ARM64 Linux"
            elif format == "coreml":
                assert MACOS or (LINUX and not ARM64), "CoreML export only supported on macOS and non-aarch64 Linux"
            if format == "coreml":
                assert not IS_PYTHON_3_13, "CoreML not supported on Python 3.13"
            if format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
                # assert not IS_PYTHON_MINIMUM_3_12, "TFLite exports not supported on Python>=3.12 yet"
            if format == "paddle":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle exports not supported yet"
                assert model.task != "obb", "Paddle OBB bug https://github.com/PaddlePaddle/Paddle/issues/72024"
                assert (LINUX and not IS_JETSON) or MACOS, "Windows and Jetson Paddle exports not supported yet"
            if format == "mnn":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 MNN exports not supported yet"
            if format == "ncnn":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN exports not supported yet"
            if format == "imx":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 IMX exports not supported"
                assert model.task in {"detect", "classify", "pose", "segment"}, (
                    "IMX export is only supported for detection, classification, pose estimation and segmentation tasks"
                )
                assert "C2f" in model.__str__(), "IMX only supported for YOLOv8n and YOLO11n"
            if format == "rknn":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 RKNN exports not supported yet"
                assert LINUX, "RKNN only supported on Linux"
                assert not is_rockchip(), "RKNN Inference only supported on Rockchip devices"
            if format == "executorch":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 ExecuTorch exports not supported yet"
            if format == "axelera":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Axelera exports not supported"
                assert LINUX and not (ARM64 and IS_DOCKER), (
                    "export is only supported on Linux and is not supported on ARM64 Docker."
                )
                assert not (model.task == "segment" and any(isinstance(m, Segment26) for m in model.model.modules())), (
                    "Axelera export does not currently support YOLO26 segmentation models"
                )
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # 导出
            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = deepcopy(model)  # PyTorch 格式
            else:
                export_data = data if "data" in valid_args else None
                filename = deepcopy(model).export(
                    imgsz=imgsz,
                    format=format,
                    half=half,
                    int8=int8,
                    data=export_data,
                    device=device,
                    verbose=False,
                    **kwargs,
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "导出失败"
            emoji = "❎"  # 表示导出成功

            # 预测
            assert model.task != "pose" or format != "pb", "GraphDef Pose inference is not supported"
            assert format not in {"edgetpu", "tfjs"}, "inference not supported"
            assert format != "coreml" or platform.system() == "Darwin", "inference only supported on macOS>=10.13"
            assert format != "axelera", "inference only supported on Axelera hardware"
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False)

            # 验证
            results = exported_model.val(
                data=data,
                batch=1,
                imgsz=imgsz,
                plots=False,
                device=device,
                half=half,
                int8=int8,
                verbose=False,
                conf=0.001,  # all the pre-set benchmark mAP values are based on conf=0.001
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # 每秒帧数
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.error(f"Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # 打印结果
    check_yolo(device=device)  # 打印系统信息
    df = pl.DataFrame(y, schema=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"], orient="row")
    df = df.with_row_index(" ", offset=1)  # 添加索引信息
    df_display = df.with_columns(pl.all().cast(pl.String).fill_null("-"))

    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df_display}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].to_numpy()  # 与底线比较的值
        floor = verbose  # 通过的最低指标底线，如 YOLOv5n 的 mAP = 0.29
        assert all(x > floor for x in metrics if not np.isnan(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df_display


class RF100Benchmark:
    """在 RF100 数据集集合上对 YOLO 模型进行基准测试。

    该类提供下载、处理和评估 YOLO 模型在 RF100 数据集上表现的功能。

    属性:
        ds_names (list[str]): 用于基准测试的数据集名称。
        ds_cfg_list (list[Path]): 数据集配置文件路径列表。
        rf (Roboflow | None): 用于访问数据集的 Roboflow 实例。
        val_metrics (list[str]): 用于验证的指标。

    方法:
        set_key: 设置 Roboflow API 密钥以访问数据集。
        parse_dataset: 解析数据集链接并下载数据集。
        fix_yaml: 修复 YAML 文件中的训练和验证路径。
        evaluate: 在验证结果上评估模型性能。
    """

    def __init__(self):
        """初始化 RF100Benchmark 类，用于在 RF100 数据集上对 YOLO 模型进行基准测试。"""
        self.ds_names = []
        self.ds_cfg_list = []
        self.rf = None
        self.val_metrics = ["class", "images", "targets", "precision", "recall", "map50", "map95"]

    def set_key(self, api_key: str):
        """设置 Roboflow API 密钥。

        参数:
            api_key (str): API 密钥。

        示例:
            设置 Roboflow API 密钥以访问数据集:
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("your_roboflow_api_key")
        """
        check_requirements("roboflow")
        from roboflow import Roboflow

        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt: str = "datasets_links.txt"):
        """解析数据集链接并下载数据集。

        参数:
            ds_link_txt (str): 包含数据集链接的文件路径。

        返回:
            (tuple[list[str], list[Path]]): 数据集名称列表和数据集配置文件路径列表。

        示例:
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("api_key")
            >>> benchmark.parse_dataset("datasets_links.txt")
        """
        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")
        os.chdir("rf-100")
        os.mkdir("ultralytics-benchmarks")
        safe_download(f"{ASSETS_URL}/datasets_links.txt")

        with open(ds_link_txt, encoding="utf-8") as file:
            for line in file:
                try:
                    _, _url, workspace, project, version = re.split("/+", line.strip())
                    self.ds_names.append(project)
                    proj_version = f"{project}-{version}"
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")
                    else:
                        LOGGER.info("Dataset already downloaded.")
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue

        return self.ds_names, self.ds_cfg_list

    @staticmethod
    def fix_yaml(path: Path):
        """修复给定 YAML 文件中的训练和验证路径。"""
        yaml_data = YAML.load(path)
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        YAML.save(path, yaml_data)

    def evaluate(self, yaml_path: str, val_log_file: str, eval_log_file: str, list_ind: int):
        """在验证结果上评估模型性能。

        参数:
            yaml_path (str): YAML 配置文件路径。
            val_log_file (str): 验证日志文件路径。
            eval_log_file (str): 评估日志文件路径。
            list_ind (int): 当前数据集在列表中的索引。

        返回:
            (float): 评估模型的平均精度（mAP）值。

        示例:
            在特定数据集上评估模型
            >>> benchmark = RF100Benchmark()
            >>> benchmark.evaluate("path/to/data.yaml", "path/to/val_log.txt", "path/to/eval_log.txt", 0)
        """
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]
        class_names = YAML.load(yaml_path)["names"]
        with open(val_log_file, encoding="utf-8") as f:
            lines = f.readlines()
            eval_lines = []
            for line in lines:
                if any(symbol in line for symbol in skip_symbols):
                    continue
                entries = line.split(" ")
                entries = list(filter(lambda val: val != "", entries))
                entries = [e.strip("\n") for e in entries]
                eval_lines.extend(
                    {
                        "class": entries[0],
                        "images": entries[1],
                        "targets": entries[2],
                        "precision": entries[3],
                        "recall": entries[4],
                        "map50": entries[5],
                        "map95": entries[6],
                    }
                    for e in entries
                    if e in class_names or (e == "all" and "(AP)" not in entries and "(AR)" not in entries)
                )
        map_val = 0.0
        if len(eval_lines) > 1:
            LOGGER.info("Multiple dicts found")
            for lst in eval_lines:
                if lst["class"] == "all":
                    map_val = lst["map50"]
        else:
            LOGGER.info("Single dict found")
            map_val = next(res["map50"] for res in eval_lines)

        with open(eval_log_file, "a", encoding="utf-8") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")

        return float(map_val)


class ProfileModels:
    """ProfileModels 类，用于在 ONNX 和 TensorRT 上分析不同模型。

    该类分析不同模型的性能，返回模型速度和 FLOPs 等结果。

    属性:
        paths (list[str]): 要分析的模型路径。
        num_timed_runs (int): 计时运行次数。
        num_warmup_runs (int): 预热运行次数。
        min_time (float): 最短分析时间（秒）。
        imgsz (int): 模型使用的图像大小。
        half (bool): 是否使用 FP16 半精度进行 TensorRT 分析的标志。
        trt (bool): 是否使用 TensorRT 进行分析的标志。
        device (torch.device): 用于分析的设备。

    方法:
        run: 对 YOLO 模型在各种格式下进行速度和精度分析。
        get_files: 获取所有相关模型文件。
        get_onnx_model_info: 从 ONNX 模型中提取元数据。
        iterative_sigma_clipping: 应用 sigma 裁剪移除异常值。
        profile_tensorrt_model: 分析 TensorRT 模型。
        profile_onnx_model: 分析 ONNX 模型。
        generate_table_row: 生成包含模型指标的表格行。
        generate_results_dict: 生成分析结果的字典。
        print_table: 打印格式化的结果表格。

    示例:
        分析模型并打印结果
        >>> from ultralytics.utils.benchmarks import ProfileModels
        >>> profiler = ProfileModels(["yolo26n.yaml", "yolov8s.yaml"], imgsz=640)
        >>> profiler.run()
    """

    def __init__(
        self,
        paths: list[str],
        num_timed_runs: int = 100,
        num_warmup_runs: int = 10,
        min_time: float = 60,
        imgsz: int = 640,
        half: bool = True,
        trt: bool = True,
        device: torch.device | str | None = None,
    ):
        """初始化 ProfileModels 类用于分析模型。

        参数:
            paths (list[str]): 要分析的模型路径列表。
            num_timed_runs (int): 计时运行次数。
            num_warmup_runs (int): 实际分析前的预热运行次数。
            min_time (float): 分析模型的最短时间（秒）。
            imgsz (int): 分析期间使用的图像大小。
            half (bool): 是否使用 FP16 半精度进行 TensorRT 分析的标志。
            trt (bool): 是否使用 TensorRT 进行分析的标志。
            device (torch.device | str | None): 用于分析的设备。如果为 None，自动确定。

        注意:
            ONNX 的 FP16 'half' 参数选项已移除，因为在 CPU 上比 FP32 更慢。
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt  # 运行 TensorRT 分析
        self.device = device if isinstance(device, torch.device) else select_device(device)

    def run(self):
        """对 YOLO 模型在各种格式（包括 ONNX 和 TensorRT）下进行速度和精度分析。

        返回:
            (list[dict]): 包含每个模型分析结果的字典列表。

        示例:
            分析模型并打印结果
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolo26n.yaml", "yolo11s.yaml"])
            >>> results = profiler.run()
        """
        files = self.get_files()

        if not files:
            LOGGER.warning("No matching *.pt or *.onnx files found.")
            return []

        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                model = YOLO(str(file))
                model.fuse()  # 以便在 model.info() 中报告正确的参数和 GFLOPs
                model_info = model.info(imgsz=self.imgsz)
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    engine_file = model.export(
                        format="engine",
                        half=self.half,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(
                    format="onnx",
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)
        return output

    def get_files(self):
        """返回用户给定的所有相关模型文件路径列表。

        返回:
            (list[Path]): 模型文件的 Path 对象列表。
        """
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {".pt", ".yaml", ".yml"}:  # 添加不存在的
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        LOGGER.info(f"Profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        """从 ONNX 模型文件中提取元数据，包括层数、参数、梯度和 FLOPs。"""
        return 0.0, 0.0, 0.0, 0.0  # 返回 (num_layers, num_params, num_gradients, num_flops)

    @staticmethod
    def iterative_sigma_clipping(data: np.ndarray, sigma: float = 2, max_iters: int = 3):
        """对数据应用迭代 sigma 裁剪以移除异常值。

        参数:
            data (np.ndarray): 输入数据数组。
            sigma (float): 用于裁剪的标准差倍数。
            max_iters (int): 裁剪过程的最大迭代次数。

        返回:
            (np.ndarray): 移除异常值后的裁剪数据数组。
        """
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """使用 TensorRT 分析 YOLO 模型性能，测量平均运行时间和标准差。

        参数:
            engine_file (str): TensorRT 引擎文件路径。
            eps (float): 防止除零的小 epsilon 值。

        返回:
            (tuple[float, float]): 推理时间的均值和标准差（毫秒）。
        """
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # 模型和输入
        model = YOLO(engine_file)
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # 使用 uint8 用于分类

        # 预热运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time

        # 计算运行次数，取 min_time 和 num_timed_runs 的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

        # 计时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])  # 转换为毫秒

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma 裁剪
        return np.mean(run_times), np.std(run_times)

    @staticmethod
    def check_dynamic(tensor_shape):
        """检查 ONNX 模型中的张量形状是否为动态。"""
        return not all(isinstance(dim, int) and dim >= 0 for dim in tensor_shape)

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        """分析 ONNX 模型，测量多次运行的平均推理时间和标准差。

        参数:
            onnx_file (str): ONNX 模型文件路径。
            eps (float): 防止除零的小 epsilon 值。

        返回:
            (tuple[float, float]): 推理时间的均值和标准差（毫秒）。
        """
        check_requirements([("onnxruntime", "onnxruntime-gpu")])  # 任一包满足要求
        import onnxruntime as ort

        # 使用 'TensorrtExecutionProvider'、'CUDAExecutionProvider' 或 'CPUExecutionProvider' 的会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8  # 限制线程数
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        input_data_dict = {}
        for input_tensor in sess.get_inputs():
            input_type = input_tensor.type
            if self.check_dynamic(input_tensor.shape):
                if len(input_tensor.shape) != 4 and self.check_dynamic(input_tensor.shape[1:]):
                    raise ValueError(f"Unsupported dynamic shape {input_tensor.shape} of {input_tensor.name}")
                input_shape = (
                    (1, 3, self.imgsz, self.imgsz) if len(input_tensor.shape) == 4 else (1, *input_tensor.shape[1:])
                )
            else:
                input_shape = input_tensor.shape

            # 映射 ONNX 数据类型到 numpy 数据类型
            if "float16" in input_type:
                input_dtype = np.float16
            elif "float" in input_type:
                input_dtype = np.float32
            elif "double" in input_type:
                input_dtype = np.float64
            elif "int64" in input_type:
                input_dtype = np.int64
            elif "int32" in input_type:
                input_dtype = np.int32
            else:
                raise ValueError(f"Unsupported ONNX datatype {input_type}")

            input_data = np.random.rand(*input_shape).astype(input_dtype)
            input_name = input_tensor.name
            input_data_dict[input_name] = input_data

        output_name = sess.get_outputs()[0].name

        # 预热运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], input_data_dict)
            elapsed = time.time() - start_time

        # 计算运行次数，取 min_time 和 num_timed_runs 的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        # 计时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], input_data_dict)
            run_times.append((time.time() - start_time) * 1000)  # 转换为毫秒

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  # sigma 裁剪
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(
        self,
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        """生成包含模型性能指标的表格行字符串。

        参数:
            model_name (str): 模型名称。
            t_onnx (tuple): ONNX 模型推理时间统计（均值，标准差）。
            t_engine (tuple): TensorRT 引擎推理时间统计（均值，标准差）。
            model_info (tuple): 模型信息（层数，参数，梯度，FLOPs）。

        返回:
            (str): 包含模型指标的格式化表格行字符串。
        """
        _layers, params, _gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±"
            f"{t_engine[1]:.1f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    def generate_results_dict(
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        """生成分析结果的字典。

        参数:
            model_name (str): 模型名称。
            t_onnx (tuple): ONNX 模型推理时间统计（均值，标准差）。
            t_engine (tuple): TensorRT 引擎推理时间统计（均值，标准差）。
            model_info (tuple): 模型信息（层数，参数，梯度，FLOPs）。

        返回:
            (dict): 包含分析结果的字典。
        """
        _layers, params, _gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows: list[str]):
        """打印格式化的模型分析结果表格。

        参数:
            table_rows (list[str]): List of formatted table row strings.
        """
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        headers = [
            "Model",
            "size<br><sup>(pixels)",
            "mAP<sup>val<br>50-95",
            f"Speed<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",
            f"Speed<br><sup>{gpu} TensorRT<br>(ms)",
            "params<br><sup>(M)",
            "FLOPs<br><sup>(B)",
        ]
        header = "|" + "|".join(f" {h} " for h in headers) + "|"
        separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"

        LOGGER.info(f"\n\n{header}")
        LOGGER.info(separator)
        for row in table_rows:
            LOGGER.info(row)
