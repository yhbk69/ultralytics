# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
from typing import Any

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

torch.classes.__path__ = []  # Torch module __path__._path issue: https://github.com/datalab-to/marker/issues/442


class Inference:
    """执行目标检测、图像分类、图像分割和姿态估计推理的类。

    此类提供加载模型、配置设置、上传视频文件以及使用 Streamlit 和 Ultralytics YOLO 模型
    进行实时推理的功能。

    属性:
        st (module): 用于创建 UI 的 Streamlit 模块。
        temp_dict (dict): 存储模型路径和其他配置的临时字典。
        model_path (str): 已加载模型的路径。
        model (YOLO): YOLO 模型实例。
        source (str): 选定的视频源（摄像头或视频文件）。
        enable_trk (bool): 启用跟踪选项。
        conf (float): 检测的置信度阈值。
        iou (float): 非极大值抑制的 IoU 阈值。
        org_frame (Any): 用于显示原始帧的容器。
        ann_frame (Any): 用于显示标注帧的容器。
        vid_file_name (str | int): 上传视频文件的名称或摄像头索引。
        selected_ind (list[int]): 用于检测的选定类别索引列表。

    方法:
        web_ui: 使用自定义 HTML 元素设置 Streamlit Web 界面。
        sidebar: 配置用于模型和推理设置的 Streamlit 侧边栏。
        source_upload: 通过 Streamlit 界面处理视频文件上传。
        configure: 配置模型并加载用于推理的选定类别。
        inference: 执行实时目标检测推理。

    示例:
        使用自定义模型创建 Inference 实例
        >>> inf = Inference(model="path/to/model.pt")
        >>> inf.inference()

        使用默认设置创建 Inference 实例
        >>> inf = Inference()
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 Inference 类，检查 Streamlit 要求并设置模型路径。

        参数:
            **kwargs (Any): 用于模型配置的额外关键字参数。
        """
        check_requirements("streamlit>=1.29.0")  # 范围导入以加快 ultralytics 包加载速度
        import streamlit as st

        self.st = st  # Streamlit 模块的引用
        self.source = None  # 视频源选择（摄像头或视频文件）
        self.img_file_names = []  # 图像文件名列表
        self.enable_trk = False  # 切换目标跟踪的标志
        self.conf = 0.25  # 检测的置信度阈值
        self.iou = 0.45  # 非极大值抑制的交并比（IoU）阈值
        self.org_frame = None  # 原始帧显示的容器
        self.ann_frame = None  # 标注帧显示的容器
        self.vid_file_name = None  # 视频文件名或摄像头索引
        self.selected_ind: list[int] = []  # 用于检测的选定类别索引列表
        self.model = None  # YOLO 模型实例

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # 模型文件路径
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self) -> None:
        """使用自定义 HTML 元素设置 Streamlit Web 界面。"""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # 隐藏主菜单样式

        # Streamlit 应用的主标题
        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit 应用</h1></div>"""

        # Streamlit 应用的副标题
        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
        margin-top:-15px; margin-bottom:50px;">使用 Ultralytics YOLO 的强大功能，在摄像头、视频和图像上体验实时目标检测！
        🚀</h5></div>"""

        # 设置 HTML 页面配置并追加自定义 HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit 应用", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """配置用于模型和推理设置的 Streamlit 侧边栏。"""
        with self.st.sidebar:  # 添加 Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("用户配置")  # 向垂直设置菜单添加元素
        self.source = self.st.sidebar.selectbox(
            "视频源",
            ("webcam", "video", "image"),
        )  # 添加视频源选择下拉框
        if self.source in ["webcam", "video"]:
            self.enable_trk = self.st.sidebar.radio("启用跟踪", ("是", "否")) == "是"  # 启用目标跟踪
        self.conf = float(
            self.st.sidebar.slider("置信度阈值", 0.0, 1.0, self.conf, 0.01)
        )  # 置信度滑块
        self.iou = float(self.st.sidebar.slider("IoU 阈值", 0.0, 1.0, self.iou, 0.01))  # NMS 阈值滑块

        if self.source != "image":  # 仅为视频/摄像头创建列
            col1, col2 = self.st.columns(2)  # 创建两列用于显示帧
            self.org_frame = col1.empty()  # 原始帧的容器
            self.ann_frame = col2.empty()  # 标注帧的容器

    def source_upload(self) -> None:
        """通过 Streamlit 界面处理视频文件上传。"""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS  # 范围导入

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("上传视频文件", type=VID_FORMATS)
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO 对象
                with open("ultralytics.mp4", "wb") as out:  # 以字节方式打开临时文件
                    out.write(g.read())  # 将字节读入文件
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # 使用摄像头索引 0
        elif self.source == "image":
            import tempfile  # 范围导入

            if imgfiles := self.st.sidebar.file_uploader(
                "上传图像文件", type=IMG_FORMATS, accept_multiple_files=True
            ):
                for imgfile in imgfiles:  # 将每个上传的图像保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def configure(self) -> None:
        """配置模型并加载用于推理的选定类别。"""
        # 添加模型选择下拉菜单
        M_ORD, T_ORD = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"], ["", "-seg", "-pose", "-obb", "-cls"]
        available_models = sorted(
            [
                x.replace("yolo", "YOLO")
                for x in GITHUB_ASSETS_STEMS
                if any(x.startswith(b) for b in M_ORD) and "grayscale" not in x
            ],
            key=lambda x: (M_ORD.index(x[:7].lower()), T_ORD.index(x[7:].lower() or "")),
        )
        if self.model_path:  # 在 available_models 中插入用户提供的自定义模型
            available_models.insert(0, self.model_path)
        selected_model = self.st.sidebar.selectbox("模型", available_models)

        with self.st.spinner("模型下载中..."):
            if selected_model.endswith((".pt", ".onnx", ".torchscript", ".mlpackage", ".engine")) or any(
                fmt in selected_model for fmt in ("openvino_model", "rknn_model")
            ):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"  # 函数调用期间未提供模型时默认使用 .pt
            self.model = YOLO(model_path)  # 加载 YOLO 模型
            class_names = list(self.model.names.values())  # 将字典转换为类别名称列表
        self.st.success("模型加载成功！")

        # 带类别名称的多选框，获取选定类别的索引
        selected_classes = self.st.sidebar.multiselect("类别", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # 确保 selected_options 是列表
            self.selected_ind = list(self.selected_ind)

    def image_inference(self) -> None:
        """对上传的图像执行推理。"""
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)  # 加载并显示原始图像
            if image is not None:
                self.st.markdown(f"#### 已处理: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="原始图像")
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                with col2:
                    self.st.image(annotated_image, channels="BGR", caption="预测图像")
                try:  # 清理临时文件
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass  # 文件不存在，忽略
            else:
                self.st.error("无法加载上传的图像。")

    def inference(self) -> None:
        """在视频或摄像头输入上执行实时目标检测推理。"""
        self.web_ui()  # 初始化 Web 界面
        self.sidebar()  # 创建侧边栏
        self.source_upload()  # 上传视频源
        self.configure()  # 配置应用

        if self.st.sidebar.button("开始"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("请上传图像文件以执行推理。")
                return

            stop_button = self.st.sidebar.button("停止")  # 停止推理的按钮
            cap = cv2.VideoCapture(self.vid_file_name)  # 捕获视频
            if not cap.isOpened():
                self.st.error("无法打开摄像头或视频源。")
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("无法从摄像头读取帧。请确认摄像头已正确连接。")
                    break

                # 使用模型处理帧
                if self.enable_trk:
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                annotated_frame = results[0].plot()  # 在帧上添加标注

                if stop_button:
                    cap.release()  # 释放捕获器
                    self.st.stop()  # 停止 Streamlit 应用

                self.org_frame.image(frame, channels="BGR", caption="原始帧")  # 显示原始帧
                self.ann_frame.image(annotated_frame, channels="BGR", caption="预测帧")  # 显示处理后的帧

            cap.release()  # 释放捕获器
        cv2.destroyAllWindows()  # 销毁所有 OpenCV 窗口


if __name__ == "__main__":
    import sys  # 导入 sys 模块以访问命令行参数

    # 检查是否提供了模型名称作为命令行参数
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # 如果提供了第一个参数则作为模型名称
    # 创建 Inference 类实例并运行推理
    Inference(model=model).inference()
