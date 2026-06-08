from pathlib import Path
from ultralytics import YOLO

model = YOLO("my/yolo11n.pt")

video_dir = Path("my/videos")
video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
videos = sorted([f for f in video_dir.iterdir() if f.suffix in video_exts])

if __name__ == "__main__":
    for v in videos:
        print(f"\n===== 处理 {v.name} =====")
        results = model.predict(
            source=str(v),           # 输入源：视频文件路径
            show=True,               # 实时显示检测画面
            save=True,               # 是否保存带标注的视频（保存到 runs/detect/predict* 目录）
            conf=0.25,               # 置信度阈值，低于此值的检测结果会被过滤
            iou=0.6,                 # 非极大值抑制（NMS）的 IoU 阈值，用于去除重叠框
            imgsz=640,               # 输入图像尺寸（像素），视频帧会被缩放到该尺寸
            half=True,               # 是否使用半精度（FP16）推理，可加速并减少显存占用
            stream=True,             # 是否使用流式处理（生成器模式），适合长视频或实时流，避免内存爆炸
            verbose=False,           # 是否打印每一帧的详细检测信息（类别、坐标、耗时等）

            # ===== 以下是其他常用参数（根据需要取消注释并调整）=====
            # device='cuda:0',        # 推理设备：'cpu'、'cuda:0'（第一块 GPU）、'mps'（Apple M1/M2）
            # project='runs/detect', # 保存结果的项目根目录，默认为 'runs/detect'
            # name='exp',            # 本次运行的子文件夹名，默认为 'exp'、'exp2' 等自动递增
            # exist_ok=False,        # 是否允许覆盖已有同名目录，设为 True 则直接覆盖而不自动递增
            # save_txt=False,        # 是否将检测结果保存为 YOLO 格式的 .txt 标签文件
            # save_crop=False,       # 是否将检测到的目标裁剪保存为单独图片
            # save_conf=False,       # 保存 .txt 时是否同时保存置信度分数（需要 save_txt=True）
            # line_width=3,          # 画检测框的线条宽度，若为 None 则根据图像尺寸自动调整
            # visualize=False,       # 是否生成特征图可视化（需要安装 `wandb` 或 `comet`）
            # augment=False,         # 是否使用测试时增强（TTA），会降低速度但可能提升精度
            # agnostic_nms=False,    # 是否进行类别无关的 NMS（不同类别的框也进行去重）
            # retina_masks=False,    # 是否使用高分辨率掩码（仅对实例分割模型有效）
            # classes=None,          # 过滤指定类别，例如 [0, 2, 5] 表示只检测 person、car、bus
            # max_det=300,           # 每张图像最多检测的目标数量
            # vid_stride=1,          # 视频帧步长，例如设为 2 则每隔一帧处理一次
            # stream_buffer=False,   # 流式处理时是否缓冲所有帧（设为 True 可能增加延迟）
        )
        for _ in results:
            pass
