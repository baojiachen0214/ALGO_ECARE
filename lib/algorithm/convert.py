from dataclasses import dataclass
from lib.algorithm.find import *
import os
import subprocess


class FigureConfig:
    def __init__(self,
                 scale_factor: float = 1.0,
                 colormap: str = 'hot',
                 interpolation_method: str = 'cubic',
                 set_figure_size: tuple = (10, 8),
                 fixed_min: float = None,
                 fixed_max: float = None,
                 x_lim: tuple = (0, 0),
                 y_lim: tuple = (0, 0),
                 show_axes: bool = True,
                 show_colorbar: bool = True,
                 show_title: bool = True):
        self.scale_factor = scale_factor
        self.colormap = colormap
        self.interpolation_method = interpolation_method
        self.set_figure_size = set_figure_size
        self.fixed_min = fixed_min
        self.fixed_max = fixed_max
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.show_axes = show_axes
        self.show_colorbar = show_colorbar
        self.show_title = show_title


class VideoConfig:
    def __init__(self,
                 categories: list = None,
                 fps: int = 24,
                 filename: str = "out-video",
                 video_format: str = 'mp4'):
        self.categories = categories if categories is not None else []
        self.fps = fps
        self.filename = filename
        self.video_format = video_format


def process_cross(list1, list2, cross_func, **kwargs):
    """
    对两个列表中的元素进行跨列表处理，cross表示交叉。

    本函数旨在通过cross_func函数对list1和list2中的元素进行配对处理，
    确保两个列表的第一维度长度相同。

    参数:
    :param list1: 第一个列表，其元素将与第二个列表的元素配对。
    :param list2: 第二个列表，其元素将与第一个列表的元素配对。
    :param cross_func: 一个函数，用于处理来自两个列表的配对元素。
    :param **kwargs: 额外的关键字参数，将传递给cross_func函数。

    异常:
    :except 如果list1的长度与list2的第一维度长度不相同，则抛出ValueError异常。
    """

    # 检查两个列表的第一维度长度是否相同
    if len(list1) != len(list2):
        raise ValueError(
            "The length of list1 must be the same as the length of the first dimension of list2")

    # 遍历两个列表，对每一对元素使用cross_func进行处理
    for key1, key2 in zip(list1, list2):
        cross_func(key1, key2, **kwargs)


# def convert_points_set2array(points_set):
#     """
#     从 Mediapipe 的 landmarks 中提取三维坐标 (x, y, z)。
#     已被弃用，get_points_set(results, face_index)方法已经直接提取为 array 格式
#
#     参数:
#     :param landmarks: Mediapipe 的面部关键点数据
#
#     返回:
#     :return 一个包含所有三维坐标的数组 (shape: [num_points, 3])
#     """
#     return np.array([[point.x, point.y, point.z] for point in points_set])


def transform_coordinates(points, new_origin, new_x_dir, new_y_dir):
    """
    针对一般的情况，将三维点集从旧坐标系转换到新的坐标系。

    参数:
    :param points (numpy.ndarray): 原始点集，形状为 (n, 3)。
    :param new_origin (array-like): 新坐标系的原点在旧坐标系下的坐标 [x_0, y_0, z_0]。
    :param new_x_dir (array-like): 新坐标系的 x 轴方向向量在旧坐标系下的表示。
    :param new_y_dir (array-like): 新坐标系的 y 轴方向向量在旧坐标系下的表示。

    返回:
    :return numpy.ndarray: 转换后的点集，形状为 (n, 3)。
    """
    # 将输入转换为 numpy 数组
    points = np.asarray(points)
    new_origin = np.asarray(new_origin)
    new_x_dir = np.asarray(new_x_dir)
    new_y_dir = np.asarray(new_y_dir)

    # 验证输入数据的维度
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须为形状为 (n, 3) 的二维数组")
    if new_origin.size != 3:
        raise ValueError("new_origin 必须为长度为 3 的一维数组")
    if new_x_dir.size != 3:
        raise ValueError("new_x_dir 必须为长度为 3 的一维数组")
    if new_y_dir.size != 3:
        raise ValueError("new_y_dir 必须为长度为 3 的一维数组")

    # 计算新的 z 轴方向 (使用右手坐标系规则)
    new_z_dir = np.cross(new_x_dir, new_y_dir)

    # 规范化方向向量，确保它们是单位向量
    new_x_dir = new_x_dir / np.linalg.norm(new_x_dir)
    new_y_dir = new_y_dir / np.linalg.norm(new_y_dir)
    new_z_dir = new_z_dir / np.linalg.norm(new_z_dir)

    # 组装旋转矩阵 (旧坐标系 -> 新坐标系)
    rotation_matrix = np.vstack([new_x_dir, new_y_dir, new_z_dir]).T

    # 平移点集至新原点
    translated_points = points - new_origin

    # 应用旋转矩阵，将旧坐标系的点转换到新坐标系
    transformed_points = translated_points @ rotation_matrix

    return transformed_points


def transform_coordinates_default(results, face_index, img):
    """
    转换指定面部的坐标系。

    此函数首先获取面部的关键点和新的坐标系原点及方向向量，
    然后通过构建旋转矩阵，将旧坐标系中的点转换到新坐标系中。

    参数:
    :param results: 包含面部检测结果的数据结构。
    :param face_index: 面部的索引。
    :param img: 原始图像数组。

    返回:
    :return transformed_points: 转换后的点集。
    """
    # 获取面部关键点在图像中的位置
    points = get_points_img_array3d(results, face_index, img)

    # 获取新的坐标系原点和方向向量（new_x_dir, new_y_dir的选择决定了投影的效果，开发者可以自行选择合适的坐标系）
    # 自定义的部分及下方几行注释的代码
    new_origin, new_x_dir, new_y_dir = get_new_origin_img_3d(results, face_index, img)

    # # 计算新的 z 轴方向 (使用右手坐标系规则)
    # new_z_dir = np.cross(new_x_dir, new_y_dir)

    # # 规范化方向向量，确保它们是单位向量
    # new_x_dir = new_x_dir / np.linalg.norm(new_x_dir)
    # new_y_dir = new_y_dir / np.linalg.norm(new_y_dir)
    # new_z_dir = new_z_dir / np.linalg.norm(new_z_dir)

    # # 组装旋转矩阵 (旧坐标系 -> 新坐标系)
    # rotation_matrix = np.vstack([new_x_dir, new_y_dir, new_z_dir]).T

    # 前文给出了自定义坐标系的代码（注释的部分），下面给出的是默认坐标系
    rotation_matrix = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])

    # 平移点集至新原点
    translated_points = points - new_origin

    # 应用旋转矩阵，将旧坐标系的点转换到新坐标系
    transformed_points = translated_points @ rotation_matrix

    return transformed_points


def transform_coordinates_array(points, new_oxy, points_list):
    """
    将给定点集从旧坐标系转换到新坐标系。

    参数:
    :param points: numpy数组，表示在旧坐标系中的点集。
    :param new_oxy: 新坐标系的原点和两个方向向量组成的列表或元组。
    :param points_list: 用于存储转换后点集的列表。

    新坐标系由原点和两个方向向量（X轴和Y轴方向）定义，本函数通过计算转换矩阵和应用平移及旋转，将点集转换到新坐标系中。
    """

    # 计算新坐标系的Z轴方向向量
    new_z_dir = np.cross(new_oxy[1], new_oxy[2])

    # 规范化方向向量，确保它们是单位向量
    new_x_dir = new_oxy[1] / np.linalg.norm(new_oxy[1])
    new_y_dir = new_oxy[2] / np.linalg.norm(new_oxy[2])
    new_z_dir = new_z_dir / np.linalg.norm(new_z_dir)

    # 组装旋转矩阵 (旧坐标系 -> 新坐标系)
    rotation_matrix = np.vstack([new_x_dir, new_y_dir, new_z_dir]).T

    # 平移点集至新原点
    translated_points = points - new_oxy[0]

    # 应用旋转矩阵，将旧坐标系的点转换到新坐标系
    transformed_points = translated_points @ rotation_matrix

    # 将转换后的点集添加到结果列表中
    points_list.append(transformed_points)


def transform_coordinates_array_list(points_list):
    """
    将一组点的坐标转换到新的坐标系中。

    该函数接收一个由3D坐标表示的点列表，并将这些点的坐标转换到一个新的坐标系中。新的坐标系由新的原点和新的x轴、y轴方向定义。
    转换过程使用辅助函数计算每个点在新坐标系中的新坐标。

    参数:
    :param points_list (list of tuple): 一个3D点列表，其中每个点由一个(x, y, z)坐标的元组表示。

    返回:
    :return numpy.ndarray: 一个包含转换后坐标的点数组，在新的坐标系中表示。
    """
    # 获取每个点的新坐标系列表
    # new_oxy的结构是 new_origin, new_x_dir, new_y_dir
    new_oxy_s = get_new_origin_img_3d_points_list(points_list)

    # 初始化存储转换后点的列表
    transformed_points_list = []

    # 处理每个点，将其坐标转换到新的坐标系中
    # 使用process_cross函数处理坐标转换
    process_cross(points_list, new_oxy_s, transform_coordinates_array,
                  points_list=transformed_points_list)

    return np.array(transformed_points_list)


# def transform_coordinates_default_array_list(landmarks_list, frame_size):
#     # 获取图像的宽高
#     h, w = frame_size[0], frame_size[1]
#
#     # 使用列表推导式，遍历点ID列表，从指定的人脸 landmarks 中提取每个点的坐标
#     return np.array([[results.multi_face_landmarks[face_index].landmark[point_id].x * w,
#                       results.multi_face_landmarks[face_index].landmark[point_id].y * h,
#                       results.multi_face_landmarks[face_index].landmark[point_id].z]
#                      for point_id in point_id_list])
#
#
#     # 获取新的坐标系原点和方向向量
#     new_origin, new_x_dir, new_y_dir = get_new_origin_img_3d(results, face_index, img)
#     # 计算新的 z 轴方向 (使用右手坐标系规则)
#     new_z_dir = np.cross(new_x_dir, new_y_dir)
#
#     # 规范化方向向量，确保它们是单位向量
#     new_x_dir = new_x_dir / np.linalg.norm(new_x_dir)
#     new_y_dir = new_y_dir / np.linalg.norm(new_y_dir)
#     new_z_dir = new_z_dir / np.linalg.norm(new_z_dir)
#
#     # 组装旋转矩阵 (旧坐标系 -> 新坐标系)
#     rotation_matrix = np.vstack([new_x_dir, new_y_dir, new_z_dir]).T
#
#     # 平移点集至新原点
#     translated_points = points - new_origin
#
#     # 应用旋转矩阵，将旧坐标系的点转换到新坐标系
#     transformed_points = translated_points @ rotation_matrix
#
#     return transformed_points


def transform_landmarks_list_to_array_list(landmarks_list, frame_size):
    """
    将地标列表转换为数组列表。

    此函数遍历地标列表，将每个地标转换为数组形式，并考虑帧的尺寸进行转换。
    每个地标的x、y坐标分别与帧的宽度和高度相乘，以获得实际的像素位置，同时y轴取反，
    以适应大多数图像坐标系中y轴向下的方向。

    参数:
    :param landmarks_list (list of list of landmarks): 嵌套的地标列表，每个内部列表对应一帧中的所有地标。
    :param frame_size (tuple): 帧的尺寸，格式为（宽度，高度）。

    返回:
    :return list of arrays: 转换后的数组列表，每个地标表示为数组。
    """
    # 初始化数组列表
    array_list = []
    # 使用tqdm包装地标列表以显示进度条
    for landmarks in tqdm(landmarks_list):
        # 将每个地标转换为数组形式，并考虑帧尺寸
        array_list.append(np.array([[landmark.x * frame_size[0],
                                     -landmark.y * frame_size[1],
                                     landmark.z] for landmark in landmarks]))
    return array_list


def matplotlib_figure_to_bgr(fig):
    """
    将matplotlib绘制的图像转换为BGR格式的NumPy数组。

    参数:
    :param fig: matplotlib.figure.Figure对象，包含要转换的图像数据。

    返回:
    :return bgr_img: 转换后的BGR格式图像数据，类型为NumPy数组。
    """
    # 创建一个空白图像
    canvas = fig.canvas
    canvas.draw()

    # 获取图像数据
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 转换为BGR格式
    bgr_img = img[..., [2, 1, 0]]

    return bgr_img


def images_to_video_ffmpeg(img_path, output_video_path, config: VideoConfig = None):
    """
    将文件夹中的指定类别图片按照顺序合成视频。

    参数:
    :param img_path: str, 图片所在文件夹路径。
    :param output_video_path: str, 输出视频文件的路径（不带扩展名）。
    :param output_filename: str, 输出视频文件的名称（不带扩展名）。
    :param categories: list, 限定的图片类别，例如['frame']只包含文件名中包含 'frame' 的图片。
    :param fps: int, 视频帧率（每秒帧数）。
    :param video_format: str, 视频格式（默认 'mp4'，可选 'avi', 'mov' 等）。

    输出:
    :return 生成的视频文件。
    """
    # 确保输出视频路径带有文件格式后缀
    config = config or VideoConfig()

    if config.categories is None or config.categories == []:
        config.categories = ['png']
    output_video_path = os.path.join(output_video_path, f"{config.filename}.{config.video_format}")

    # 处理 img_path，确保其格式正确
    img_path = os.path.normpath(img_path)

    # 获取图片列表并进行过滤
    img_files = sorted([
        f for f in os.listdir(img_path)
        if any(category in f for category in config.categories) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=lambda x: int(''.join(filter(str.isdigit, x))))  # 按编号顺序排序

    # 检查是否有图片满足条件
    if not img_files:
        print("没有找到符合条件的图片。")
        return

    # 使用 ffmpeg 的输入文件列表格式创建临时文件
    with open("filelist.txt", "w") as f:
        for img_file in img_files:
            # 写入每一行的绝对路径，确保路径格式的兼容性
            img_path_full = os.path.join(img_path, img_file)
            f.write(f"file '{img_path_full}'\n")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构建 ffmpeg 命令
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-r", str(config.fps),  # 设置帧率
        "-f", "concat",  # 指定输入格式为 concat
        "-safe", "0",  # 禁用安全模式
        "-i", "filelist.txt",  # 输入文件列表
        "-c:v", "libx264",  # 视频编码格式
        "-pix_fmt", "yuv420p",  # 像素格式，确保兼容性
        output_video_path  # 输出文件路径
    ]

    # 执行 ffmpeg 命令
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"视频已生成: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("视频生成失败:", e)
    finally:
        # 删除临时文件
        os.remove("filelist.txt")


