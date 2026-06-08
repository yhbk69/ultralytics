# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    """通过实时视频流中的人体姿态管理健身动作计数。

    此类扩展了 BaseSolution，使用 YOLO 姿态估计模型监测健身训练。它基于
    预定义的上下位置角度阈值来跟踪和计数训练动作的重复次数。

    属性:
        states (dict[int, dict[str, float | int | str]]): 每个跟踪目标的健身监测角度、次数和阶段。
        up_angle (float): 判定动作"上"位置的角度阈值。
        down_angle (float): 判定动作"下"位置的角度阈值。
        kpts (list[int]): 用于角度计算的关键点索引。

    方法:
        process: 处理帧以检测姿态、计算角度并计数重复次数。

    示例:
        >>> gym = AIGym(model="yolo26n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> results = gym.process(image)
        >>> processed_image = results.plot_im
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 AIGym，使用姿态估计和预定义角度进行健身监测。

        参数:
            **kwargs (Any): 传递给父类构造函数的关键字参数，包括：
                - model (str): 模型名称或路径，默认为 "yolo26n-pose.pt"。
        """
        kwargs["model"] = kwargs.get("model", "yolo26n-pose.pt")
        super().__init__(**kwargs)
        self.states = defaultdict(lambda: {"angle": 0, "count": 0, "stage": "-"})  # 存储计数、角度和阶段的字典

        # 从 CFG 中一次性提取参数供后续使用
        self.up_angle = float(self.CFG["up_angle"])  # 预定义的上姿态角度阈值
        self.down_angle = float(self.CFG["down_angle"])  # 预定义的下姿态角度阈值
        self.kpts = self.CFG["kpts"]  # 用户选择的健身关键点，供后续使用

    def process(self, im0) -> SolutionResults:
        """使用 Ultralytics YOLO 姿态模型监测健身训练。

        此函数处理输入图像以跟踪和分析人体姿态，用于健身监测。它使用 YOLO
        姿态模型检测关键点、估计角度，并基于预定义的角度阈值计数重复次数。

        参数:
            im0 (np.ndarray): 待处理的输入图像。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im`、'workout_count'（已完成动作次数列表）、
                'workout_stage'（当前阶段列表）、'workout_angle'（角度列表）和 'total_tracks'
                （跟踪总人数）。

        示例:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> results = gym.process(image)
            >>> processed_image = results.plot_im
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        self.extract_tracks(im0)  # 提取跟踪数据（边界框、类别和掩码）

        if len(self.boxes):
            kpt_data = self.tracks.keypoints.data

            for i, k in enumerate(kpt_data):
                state = self.states[self.track_ids[i]]  # 获取状态详情
                # 获取关键点并估计角度
                state["angle"] = annotator.estimate_pose_angle(*[k[int(idx)] for idx in self.kpts])
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                # 根据角度阈值判断阶段和计数逻辑
                if state["angle"] < self.down_angle:
                    if state["stage"] == "up":
                        state["count"] += 1
                    state["stage"] = "down"
                elif state["angle"] > self.up_angle:
                    state["stage"] = "up"

                # 显示角度、计数和阶段文本
                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=state["angle"],  # 显示的角度文本
                        count_text=state["count"],  # 显示的计数文本
                        stage_text=state["stage"],  # 显示的阶段文本
                        center_kpt=k[int(self.kpts[1])],  # 显示的中心关键点
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)  # 显示输出图像（若环境支持）

        # 返回 SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            workout_count=[v["count"] for v in self.states.values()],
            workout_stage=[v["stage"] for v in self.states.values()],
            workout_angle=[v["angle"] for v in self.states.values()],
            total_tracks=len(self.track_ids),
        )
