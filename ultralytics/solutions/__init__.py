# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license
# Ultralytics 视觉 AI 解决方案包，提供开箱即用的计算机视觉应用模块

from .ai_gym import AIGym
from .analytics import Analytics
from .distance_calculation import DistanceCalculation
from .heatmap import Heatmap
from .instance_segmentation import InstanceSegmentation
from .object_blurrer import ObjectBlurrer
from .object_counter import ObjectCounter
from .object_cropper import ObjectCropper
from .parking_management import ParkingManagement, ParkingPtsSelection
from .queue_management import QueueManager
from .region_counter import RegionCounter
from .security_alarm import SecurityAlarm
from .similarity_search import SearchApp, VisualAISearch
from .speed_estimation import SpeedEstimator
from .streamlit_inference import Inference
from .trackzone import TrackZone
from .vision_eye import VisionEye

__all__ = (
    "AIGym",
    "Analytics",
    "DistanceCalculation",
    "Heatmap",
    "Inference",
    "InstanceSegmentation",
    "ObjectBlurrer",
    "ObjectCounter",
    "ObjectCropper",
    "ParkingManagement",
    "ParkingPtsSelection",
    "QueueManager",
    "RegionCounter",
    "SearchApp",
    "SecurityAlarm",
    "SpeedEstimator",
    "TrackZone",
    "VisionEye",
    "VisualAISearch",
)
