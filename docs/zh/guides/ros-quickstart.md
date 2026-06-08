---
comments: true
description: 学习如何将 Ultralytics YOLO 与运行 ROS Noetic 的机器人集成，利用 RGB 图像、深度图像和点云实现高效的目标检测、分割和增强的机器人感知。
keywords: Ultralytics, YOLO, 目标检测, 深度学习, 机器学习, 指南, ROS, 机器人操作系统, 机器人, ROS Noetic, Python, Ubuntu, 仿真, 可视化, 通信, 中间件, 硬件抽象, 工具, 实用程序, 生态系统, Noetic Ninjemys, 自动驾驶车辆, AMV
---

# ROS（机器人操作系统）快速入门指南

<p align="center"> <iframe src="https://player.vimeo.com/video/639236696?h=740f412ce5" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><a href="https://vimeo.com/639236696">ROS 介绍（带字幕）</a>，来自 <a href="https://vimeo.com/osrfoundation">Open Robotics</a>，发布于 <a href="https://vimeo.com/">Vimeo</a>。</p>

## 什么是 ROS？

[机器人操作系统（ROS）](https://www.ros.org/) 是一个广泛应用于机器人研究和工业的开源框架。ROS 提供了一系列 [库和工具](https://www.ros.org/blog/ecosystem/)，帮助开发者创建机器人应用程序。ROS 被设计为可以与各种 [机器人平台](https://robots.ros.org/) 配合使用，使其成为机器人开发者的灵活而强大的工具。

### ROS 的关键特性

1. **模块化架构**：ROS 具有模块化架构，允许开发者通过组合更小、可复用的组件（称为 [节点](https://wiki.ros.org/ROS/Tutorials/UnderstandingNodes)）来构建复杂系统。每个节点通常执行特定功能，节点之间通过 [话题](https://wiki.ros.org/ROS/Tutorials/UnderstandingTopics) 或 [服务](https://wiki.ros.org/ROS/Tutorials/UnderstandingServicesParams) 使用消息进行通信。

2. **通信中间件**：ROS 提供了强大的通信基础设施，支持进程间通信和分布式计算。这通过数据流（话题）的发布-订阅模型和服务调用的请求-回复模型实现。

3. **硬件抽象**：ROS 提供了硬件抽象层，使开发者能够编写与设备无关的代码。这使得相同的代码可以用于不同的硬件配置，便于集成和实验。

4. **工具和实用程序**：ROS 附带了一套丰富的工具和实用程序，用于可视化、调试和仿真。例如，RViz 用于可视化传感器数据和机器人状态信息，而 Gazebo 提供了一个强大的仿真环境，用于测试算法和机器人设计。

5. **广泛的生态系统**：ROS 生态庞大且持续增长，拥有大量适用于不同机器人应用的软件包，包括导航、操作、感知等。社区积极参与这些软件包的开发和维护。

???+ note "ROS 版本的演变"

    自 2007 年开发以来，ROS 经历了 [多个版本](https://wiki.ros.org/Distributions) 的演变，每个版本都引入了新功能和改进，以满足机器人社区日益增长的需求。ROS 的发展可以分为两个主要系列：ROS 1 和 ROS 2。本指南重点介绍 ROS 1 的长期支持（LTS）版本，即 ROS Noetic Ninjemys，代码也应该兼容更早的版本。

    ### ROS 1 与 ROS 2

    虽然 ROS 1 为机器人开发提供了坚实的基础，但 ROS 2 通过以下特性解决了其不足之处：

    - **实时性能**：改进了对实时系统和确定性行为的支持。
    - **安全性**：增强了安全特性，确保在各种环境中安全可靠地运行。
    - **可扩展性**：更好地支持多机器人系统和大规模部署。
    - **跨平台支持**：扩展了对 Linux 之外的多种操作系统的兼容性，包括 Windows 和 macOS。
    - **灵活通信**：使用 DDS 实现更灵活、更高效的进程间通信。

### ROS 消息和话题

在 ROS 中，节点之间的通信通过 [消息](https://wiki.ros.org/Messages) 和 [话题](https://wiki.ros.org/Topics) 来实现。消息是定义节点之间交换信息的数据结构，而话题是用于发送和接收消息的命名通道。节点可以向话题发布消息或从话题订阅消息，从而实现彼此之间的通信。这种发布-订阅模型允许异步通信和节点之间的解耦。机器人系统中的每个传感器或执行器通常将数据发布到一个话题，然后由其他节点用于处理或控制。在本指南中，我们将重点关注图像（Image）、深度（Depth）和点云（PointCloud）消息以及相机话题。

## 在 ROS 中设置 Ultralytics YOLO

本指南已在 [这个 ROS 环境](https://github.com/ambitious-octopus/rosbot_ros/tree/noetic) 中测试通过，该环境是 [ROSbot ROS 仓库](https://github.com/husarion/rosbot_ros) 的一个分支。此环境包含 Ultralytics YOLO 软件包、便于设置的 Docker 容器、全面的 ROS 软件包以及用于快速测试的 Gazebo 世界。它被设计为与 [Husarion ROSbot 2 PRO](https://husarion.com/manuals/rosbot/) 配合使用。提供的代码示例适用于任何 ROS Noetic/Melodic 环境，包括仿真和真实环境。

<p align="center">
  <img width="50%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/husarion-rosbot-2-pro.avif" alt="Husarion ROSbot 2 PRO 自主机器人平台">
</p>

### 依赖安装

除了 ROS 环境外，还需要安装以下依赖：

- **[ROS NumPy 软件包](https://github.com/eric-wieser/ros_numpy)**：用于在 ROS Image 消息和 NumPy 数组之间快速转换。

    ```bash
    pip install ros_numpy
    ```

- **Ultralytics 软件包**：

    ```bash
    pip install ultralytics
    ```

## 将 Ultralytics 与 ROS `sensor_msgs/Image` 结合使用

`sensor_msgs/Image` [消息类型](https://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html) 在 ROS 中常用于表示图像数据。它包含编码、高度、宽度和像素数据字段，适用于传输由相机或其他传感器捕获的图像。图像消息在机器人应用中广泛用于视觉感知、[目标检测](https://www.ultralytics.com/glossary/object-detection) 和导航等任务。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/detection-segmentation-ros-gazebo.avif" alt="ROS Gazebo 中的检测和分割">
</p>

### 图像逐步使用说明

以下代码片段演示了如何将 Ultralytics YOLO 软件包与 ROS 结合使用。在此示例中，我们订阅一个相机话题，使用 YOLO 处理传入的图像，并将检测到的对象发布到新的话题中，分别用于 [检测](../tasks/detect.md) 和 [分割](../tasks/segment.md)。

首先，导入必要的库并实例化两个模型：一个用于 [分割](../tasks/segment.md)，一个用于 [检测](../tasks/detect.md)。初始化一个 ROS 节点（命名为 `ultralytics`）以启用与 ROS 主节点的通信。为确保稳定连接，我们加入短暂暂停，给节点足够的时间在继续之前建立连接。

```python
import time

import rospy

from ultralytics import YOLO

detection_model = YOLO("yolo26m.pt")
segmentation_model = YOLO("yolo26m-seg.pt")
rospy.init_node("ultralytics")
time.sleep(1)
```

初始化两个 ROS 话题：一个用于 [检测](../tasks/detect.md)，一个用于 [分割](../tasks/segment.md)。这些话题将用于发布标注后的图像，使其可供进一步处理。节点之间的通信通过 `sensor_msgs/Image` 消息来实现。

```python
from sensor_msgs.msg import Image

det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)
```

最后，创建一个订阅者，监听 `/camera/color/image_raw` 话题上的消息，并在收到每条新消息时调用回调函数。此回调函数接收 `sensor_msgs/Image` 类型的消息，使用 `ros_numpy` 将其转换为 NumPy 数组，用之前实例化的 YOLO 模型处理图像，标注图像，然后将其发布回相应的话题：检测结果发布到 `/ultralytics/detection/image`，分割结果发布到 `/ultralytics/segmentation/image`。

```python
import ros_numpy


def callback(data):
    """处理图像并发布标注图像的回调函数。"""
    array = ros_numpy.numpify(data)
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

    if seg_image_pub.get_num_connections():
        seg_result = segmentation_model(array)
        seg_annotated = seg_result[0].plot(show=False)
        seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)

while True:
    rospy.spin()
```

??? example "完整代码"

    ```python
    import time

    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image

    from ultralytics import YOLO

    detection_model = YOLO("yolo26m.pt")
    segmentation_model = YOLO("yolo26m-seg.pt")
    rospy.init_node("ultralytics")
    time.sleep(1)

    det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
    seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


    def callback(data):
        """处理图像并发布标注图像的回调函数。"""
        array = ros_numpy.numpify(data)
        if det_image_pub.get_num_connections():
            det_result = detection_model(array)
            det_annotated = det_result[0].plot(show=False)
            det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        if seg_image_pub.get_num_connections():
            seg_result = segmentation_model(array)
            seg_annotated = seg_result[0].plot(show=False)
            seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


    rospy.Subscriber("/camera/color/image_raw", Image, callback)

    while True:
        rospy.spin()
    ```

???+ tip "调试"

    由于系统的分布式特性，调试 ROS（机器人操作系统）节点可能具有挑战性。以下工具可以帮助完成此过程：

    1. `rostopic echo <TOPIC-NAME>`：此命令允许你查看在特定话题上发布的消息，帮助你检查数据流。
    2. `rostopic list`：使用此命令列出 ROS 系统中所有可用的话题，让你了解活跃的数据流。
    3. `rqt_graph`：此可视化工具显示节点之间的通信图，提供节点如何互联以及如何交互的洞察。
    4. 对于更复杂的可视化，如 3D 表示，可以使用 [RViz](https://wiki.ros.org/rviz)。RViz（ROS 可视化）是一个强大的 ROS 3D 可视化工具。它允许你实时可视化机器人的状态及其环境。通过 RViz，你可以查看传感器数据（如 `sensor_msgs/Image`）、机器人模型状态以及各种其他类型的信息，从而更容易调试和理解机器人系统的行为。

### 使用 `std_msgs/String` 发布检测到的类别

标准 ROS 消息还包括 `std_msgs/String` 消息。在许多应用中，不需要重新发布整个标注图像，只需要机器人视野中存在的类别。以下示例演示了如何使用 `std_msgs/String` [消息](https://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html) 在 `/ultralytics/detection/classes` 话题上重新发布检测到的类别。这些消息更轻量，提供基本信息，在各种应用中非常有价值。

#### 示例用例

设想一个配备相机和目标 [检测模型](../tasks/detect.md) 的仓库机器人。该机器人不是通过网络发送大型标注图像，而是可以将检测到的类别列表作为 `std_msgs/String` 消息发布。例如，当机器人检测到 "box"（箱子）、"pallet"（托盘）和 "forklift"（叉车）等物体时，它会将这些类别发布到 `/ultralytics/detection/classes` 话题。然后，中央监控系统可以利用这些信息实时跟踪库存，优化机器人的路径规划以避开障碍物，或触发特定动作，如拾取检测到的箱子。这种方法减少了通信所需的带宽，并专注于传输关键数据。

### 字符串逐步使用说明

此示例演示了如何将 Ultralytics YOLO 软件包与 ROS 结合使用。在此示例中，我们订阅一个相机话题，使用 YOLO 处理传入的图像，并使用 `std_msgs/String` 消息将检测到的对象发布到新话题 `/ultralytics/detection/classes`。`ros_numpy` 软件包用于将 ROS Image 消息转换为 NumPy 数组，以便用 YOLO 处理。

```python
import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

detection_model = YOLO("yolo26m.pt")
rospy.init_node("ultralytics")
time.sleep(1)
classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)


def callback(data):
    """处理图像并发布检测到的类别的回调函数。"""
    array = ros_numpy.numpify(data)
    if classes_pub.get_num_connections():
        det_result = detection_model(array)
        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        classes_pub.publish(String(data=str(names)))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
while True:
    rospy.spin()
```

## 将 Ultralytics 与 ROS 深度图像结合使用

除了 RGB 图像，ROS 还支持 [深度图像](https://en.wikipedia.org/wiki/Depth_map)，它提供物体与相机之间距离的信息。深度图像对于机器人应用至关重要，如避障、3D 建图和定位。

深度图像是一种每个像素表示相机到物体距离的图像。与捕获颜色的 RGB 图像不同，深度图像捕获空间信息，使机器人能够感知其环境的 3D 结构。

!!! tip "获取深度图像"

    深度图像可以使用各种传感器获取：

    1. [立体相机](https://en.wikipedia.org/wiki/Stereo_camera)：使用两个相机根据图像视差计算深度。
    2. [飞行时间（ToF）相机](https://en.wikipedia.org/wiki/Time-of-flight_camera)：测量光线从物体返回所需的时间。
    3. [结构光传感器](https://en.wikipedia.org/wiki/Structured-light_3D_scanner)：投射图案并测量其在表面上的形变。

### 将 YOLO 与深度图像结合使用

在 ROS 中，深度图像由 `sensor_msgs/Image` 消息类型表示，包含编码、高度、宽度和像素数据字段。深度图像的编码字段通常使用诸如 "16UC1" 的格式，表示每个像素为 16 位无符号整数，其中每个值表示到物体的距离。深度图像通常与 RGB 图像结合使用，以提供更全面的环境视图。

使用 YOLO，可以从 RGB 和深度图像中提取并组合信息。例如，YOLO 可以在 RGB 图像中检测物体，这种检测可用于定位深度图像中的相应区域。这允许提取检测到的物体的精确深度信息，增强机器人在三维空间中理解其环境的能力。

!!! warning "RGB-D 相机"

    使用深度图像时，必须确保 RGB 和深度图像正确对齐。RGB-D 相机，如 [Intel RealSense](https://www.realsenseai.com/) 系列，提供同步的 RGB 和深度图像，使组合两个来源的信息更加容易。如果使用单独的 RGB 和深度相机，必须对它们进行校准以确保准确对齐。

#### 深度逐步使用说明

在此示例中，我们使用 YOLO 分割图像，并应用提取的掩码在深度图像中分割物体。这使我们能够确定感兴趣物体的每个像素与相机焦点中心的距离。通过获取这些距离信息，我们可以计算相机与场景中特定物体之间的距离。首先导入必要的库，创建一个 ROS 节点，并实例化一个分割模型和一个 ROS 话题。

```python
import time

import rospy
from std_msgs.msg import String

from ultralytics import YOLO

rospy.init_node("ultralytics")
time.sleep(1)

segmentation_model = YOLO("yolo26m-seg.pt")

classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
```

接下来，定义一个回调函数来处理传入的深度图像消息。该函数等待深度图像和 RGB 图像消息，将它们转换为 NumPy 数组，并将分割模型应用于 RGB 图像。然后，它为每个检测到的物体提取分割掩码，并使用深度图像计算物体距相机的平均距离。大多数传感器有一个最大距离，称为裁剪距离，超出此距离的值表示为 inf（`np.inf`）。在处理之前，重要的是过滤掉这些空值并将其赋值为 `0`。最后，它将检测到的物体及其平均距离发布到 `/ultralytics/detection/distance` 话题。

```python
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image


def callback(data):
    """处理深度图像和 RGB 图像的回调函数。"""
    image = rospy.wait_for_message("/camera/color/image_raw", Image)
    image = ros_numpy.numpify(image)
    depth = ros_numpy.numpify(data)
    result = segmentation_model(image)

    all_objects = []
    for index, cls in enumerate(result[0].boxes.cls):
        class_index = int(cls.cpu().numpy())
        name = result[0].names[class_index]
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        obj = depth[mask == 1]
        obj = obj[~np.isnan(obj)]
        avg_distance = np.mean(obj) if len(obj) else np.inf
        all_objects.append(f"{name}: {avg_distance:.2f}m")

    classes_pub.publish(String(data=str(all_objects)))


rospy.Subscriber("/camera/depth/image_raw", Image, callback)

while True:
    rospy.spin()
```

??? example "完整代码"

    ```python
    import time

    import numpy as np
    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from ultralytics import YOLO

    rospy.init_node("ultralytics")
    time.sleep(1)

    segmentation_model = YOLO("yolo26m-seg.pt")

    classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)


    def callback(data):
        """处理深度图像和 RGB 图像的回调函数。"""
        image = rospy.wait_for_message("/camera/color/image_raw", Image)
        image = ros_numpy.numpify(image)
        depth = ros_numpy.numpify(data)
        result = segmentation_model(image)

        all_objects = []
        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
            obj = depth[mask == 1]
            obj = obj[~np.isnan(obj)]
            avg_distance = np.mean(obj) if len(obj) else np.inf
            all_objects.append(f"{name}: {avg_distance:.2f}m")

        classes_pub.publish(String(data=str(all_objects)))


    rospy.Subscriber("/camera/depth/image_raw", Image, callback)

    while True:
        rospy.spin()
    ```

## 将 Ultralytics 与 ROS `sensor_msgs/PointCloud2` 结合使用

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/detection-segmentation-ros-gazebo-1.avif" alt="ROS Gazebo 中的检测和分割">
</p>

`sensor_msgs/PointCloud2` [消息类型](https://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html) 是 ROS 中用于表示 3D 点云数据的数据结构。此消息类型对机器人应用至关重要，支持 3D 建图、物体识别和定位等任务。

点云是在三维坐标系中定义的点的集合。这些数据点表示物体或场景的外表面，通过 3D 扫描技术捕获。点云中的每个点都有 `X`、`Y` 和 `Z` 坐标，对应其在空间中的位置，还可能包括颜色和强度等附加信息。

!!! warning "参考坐标系"

    使用 `sensor_msgs/PointCloud2` 时，必须考虑获取点云数据的传感器的参考坐标系。点云最初是在传感器的参考坐标系中捕获的。你可以通过监听 `/tf_static` 话题来确定此参考坐标系。但是，根据你的具体应用需求，你可能需要将点云转换到另一个参考坐标系。此转换可以使用 `tf2_ros` 软件包实现，该软件包提供了管理坐标框架和在它们之间转换数据的工具。

!!! tip "获取点云"

    点云可以使用各种传感器获取：

    1. **LIDAR（光检测和测距）**：使用激光脉冲测量到物体的距离，创建高 [精度](https://www.ultralytics.com/glossary/precision) 3D 地图。
    2. **深度相机**：捕获每个像素的深度信息，允许对场景进行 3D 重建。
    3. **立体相机**：利用两个或更多相机通过三角测量获取深度信息。
    4. **结构光扫描仪**：将已知图案投射到表面上，并测量形变以计算深度。

### 将 YOLO 与点云结合使用

要将 YOLO 与 `sensor_msgs/PointCloud2` 类型消息集成，我们可以采用类似于深度图的方法。利用嵌入在点云中的颜色信息，我们可以提取 2D 图像，使用 YOLO 对该图像执行分割，然后将生成的掩码应用于三维点以隔离感兴趣的 3D 物体。

对于点云处理，我们推荐使用 Open3D（`pip install open3d`），一个用户友好的 Python 库。Open3D 提供了强大的工具来管理点云数据结构、可视化它们并无缝执行复杂操作。该库可以显著简化过程，增强我们结合基于 YOLO 的分割来操作和分析点云的能力。

#### 点云逐步使用说明

导入必要的库并实例化用于分割的 YOLO 模型。

```python
import time

import rospy

from ultralytics import YOLO

rospy.init_node("ultralytics")
time.sleep(1)
segmentation_model = YOLO("yolo26m-seg.pt")
```

创建一个函数 `pointcloud2_to_array`，将 `sensor_msgs/PointCloud2` 消息转换为两个 NumPy 数组。`sensor_msgs/PointCloud2` 消息根据获取图像的 `width` 和 `height` 包含 `n` 个点。例如，一个 `480 x 640` 的图像将有 `307,200` 个点。每个点包含三个空间坐标（`xyz`）和对应的 `RGB` 格式颜色。这些可以被视为两个独立的信息通道。

该函数以原始相机分辨率（`width x height`）格式返回 `xyz` 坐标和 `RGB` 值。大多数传感器有一个最大距离，称为裁剪距离，超出此距离的值表示为 inf（`np.inf`）。在处理之前，重要的是过滤掉这些空值并将其赋值为 `0`。

```python
import numpy as np
import ros_numpy


def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
    """将 ROS PointCloud2 消息转换为 NumPy 数组。

    Args:
        pointcloud2 (PointCloud2): PointCloud2 消息

    Returns:
        (tuple): 包含 (xyz, rgb) 的元组
    """
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
    split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
    rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
    xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
    xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
    nan_rows = np.isnan(xyz).all(axis=2)
    xyz[nan_rows] = [0, 0, 0]
    rgb[nan_rows] = [0, 0, 0]
    return xyz, rgb
```

接下来，订阅 `/camera/depth/points` 话题以接收点云消息，并将 `sensor_msgs/PointCloud2` 消息转换为包含 XYZ 坐标和 RGB 值的 NumPy 数组（使用 `pointcloud2_to_array` 函数）。使用 YOLO 模型处理 RGB 图像以提取分割的物体。对于每个检测到的物体，提取分割掩码并将其应用于 RGB 图像和 XYZ 坐标，以在 3D 空间中隔离物体。

处理掩码很简单，因为它由二进制值组成，`1` 表示物体的存在，`0` 表示不存在。要应用掩码，只需将原始通道乘以掩码即可。此操作有效地隔离了图像中感兴趣的物体。最后，创建一个 Open3D 点云对象，并在 3D 空间中可视化分割的物体及其关联的颜色。

```python
import sys

import open3d as o3d

ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
xyz, rgb = pointcloud2_to_array(ros_cloud)
result = segmentation_model(rgb)

if not len(result[0].boxes.cls):
    print("No objects detected")
    sys.exit()

classes = result[0].boxes.cls.cpu().numpy().astype(int)
for index, class_id in enumerate(classes):
    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
    mask_expanded = np.stack([mask, mask, mask], axis=2)

    obj_rgb = rgb * mask_expanded
    obj_xyz = xyz * mask_expanded

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((ros_cloud.height * ros_cloud.width, 3)))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((ros_cloud.height * ros_cloud.width, 3)) / 255)
    o3d.visualization.draw_geometries([pcd])
```

??? example "完整代码"

    ```python
    import sys
    import time

    import numpy as np
    import open3d as o3d
    import ros_numpy
    import rospy
    from sensor_msgs.msg import PointCloud2

    from ultralytics import YOLO

    rospy.init_node("ultralytics")
    time.sleep(1)
    segmentation_model = YOLO("yolo26m-seg.pt")


    def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
        """将 ROS PointCloud2 消息转换为 NumPy 数组。

        Args:
            pointcloud2 (PointCloud2): PointCloud2 消息

        Returns:
            (tuple): 包含 (xyz, rgb) 的元组
        """
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
        split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
        rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
        xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
        xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
        nan_rows = np.isnan(xyz).all(axis=2)
        xyz[nan_rows] = [0, 0, 0]
        rgb[nan_rows] = [0, 0, 0]
        return xyz, rgb


    ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
    xyz, rgb = pointcloud2_to_array(ros_cloud)
    result = segmentation_model(rgb)

    if not len(result[0].boxes.cls):
        print("No objects detected")
        sys.exit()

    classes = result[0].boxes.cls.cpu().numpy().astype(int)
    for index, class_id in enumerate(classes):
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        mask_expanded = np.stack([mask, mask, mask], axis=2)

        obj_rgb = rgb * mask_expanded
        obj_xyz = xyz * mask_expanded

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((ros_cloud.height * ros_cloud.width, 3)))
        pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((ros_cloud.height * ros_cloud.width, 3)) / 255)
        o3d.visualization.draw_geometries([pcd])
    ```

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/point-cloud-segmentation-ultralytics.avif" alt="使用 Ultralytics 进行点云分割">
</p>

## 常见问题

### 什么是机器人操作系统（ROS）？

[机器人操作系统（ROS）](https://www.ros.org/) 是一个广泛应用于机器人领域的开源框架，帮助开发者创建强大的机器人应用程序。它提供了一系列 [库和工具](https://www.ros.org/blog/ecosystem/)，用于构建和连接机器人系统，使复杂应用程序的开发更加容易。ROS 通过 [话题](https://wiki.ros.org/ROS/Tutorials/UnderstandingTopics) 或 [服务](https://wiki.ros.org/ROS/Tutorials/UnderstandingServicesParams) 支持节点之间使用消息进行通信。

### 如何将 Ultralytics YOLO 与 ROS 集成以实现实时目标检测？

将 Ultralytics YOLO 与 ROS 集成涉及设置 ROS 环境并使用 YOLO 处理传感器数据。首先安装所需的依赖，如 `ros_numpy` 和 Ultralytics YOLO：

```bash
pip install ros_numpy ultralytics
```

接下来，创建一个 ROS 节点并订阅一个 [图像话题](../tasks/detect.md) 来处理传入的数据。以下是一个最小示例：

```python
import ros_numpy
import rospy
from sensor_msgs.msg import Image

from ultralytics import YOLO

detection_model = YOLO("yolo26m.pt")
rospy.init_node("ultralytics")
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)


def callback(data):
    array = ros_numpy.numpify(data)
    det_result = detection_model(array)
    det_annotated = det_result[0].plot(show=False)
    det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
rospy.spin()
```

### 什么是 ROS 话题，它们如何与 Ultralytics YOLO 一起使用？

ROS 话题通过使用发布-订阅模型促进 ROS 网络中节点之间的通信。话题是节点用来异步发送和接收消息的命名通道。在 Ultralytics YOLO 的上下文中，你可以让节点订阅图像话题，使用 YOLO 处理图像以执行 [检测](https://docs.ultralytics.com/tasks/detect) 或 [分割](https://docs.ultralytics.com/tasks/segment) 等任务，并将结果发布到新话题。

例如，订阅相机话题并处理传入的图像进行检测：

```python
rospy.Subscriber("/camera/color/image_raw", Image, callback)
```

### 为什么在 ROS 中将深度图像与 Ultralytics YOLO 结合使用？

ROS 中由 `sensor_msgs/Image` 表示的深度图像提供了物体到相机的距离信息，这对避障、3D 建图和定位等任务至关重要。通过将 [深度信息](https://en.wikipedia.org/wiki/Depth_map) 与 RGB 图像结合使用，机器人可以更好地理解其 3D 环境。

使用 YOLO，你可以从 RGB 图像中提取 [分割掩码](https://www.ultralytics.com/glossary/image-segmentation)，并将这些掩码应用于深度图像以获取精确的 3D 物体信息，从而提升机器人导航和与周围环境交互的能力。

### 如何在 ROS 中使用 YOLO 可视化 3D 点云？

要在 ROS 中使用 YOLO 可视化 3D 点云：

1. 将 `sensor_msgs/PointCloud2` 消息转换为 NumPy 数组。
2. 使用 YOLO 分割 RGB 图像。
3. 将分割掩码应用于点云。

以下是使用 [Open3D](https://www.open3d.org/) 进行可视化的示例：

```python
import sys

import open3d as o3d
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2

from ultralytics import YOLO

rospy.init_node("ultralytics")
segmentation_model = YOLO("yolo26m-seg.pt")


def pointcloud2_to_array(pointcloud2):
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
    split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
    rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
    xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
    xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
    return xyz, rgb


ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
xyz, rgb = pointcloud2_to_array(ros_cloud)
result = segmentation_model(rgb)

if not len(result[0].boxes.cls):
    print("No objects detected")
    sys.exit()

classes = result[0].boxes.cls.cpu().numpy().astype(int)
for index, class_id in enumerate(classes):
    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
    mask_expanded = np.stack([mask, mask, mask], axis=2)

    obj_rgb = rgb * mask_expanded
    obj_xyz = xyz * mask_expanded

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((-1, 3)))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((-1, 3)) / 255)
    o3d.visualization.draw_geometries([pcd])
```

这种方法提供了分割物体的 3D 可视化，对于 [机器人应用](https://docs.ultralytics.com/guides/steps-of-a-cv-project) 中的导航和操作等任务非常有用。
