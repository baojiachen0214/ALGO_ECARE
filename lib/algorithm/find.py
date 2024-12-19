import cv2
from tqdm import tqdm
from lib.algorithm.distance import *
from lib.algorithm.idlist import *


def get_points_array3d(results, face_index):
    """
    根据指定的面部索引，从multi_face_landmarks对象中获取单个面部的所有特征点，并返回一个NumPy数组。

    参数:
    :param results -- 包含多个面部特征点信息的对象，通常由面部检测或识别函数返回。
    :param face_index -- 需要获取特征点的面部索引，即第几张人脸。

    返回:
    :return 一个NumPy数组，其中每一行代表一个点的(x, y, z)相对坐标。
    """
    # 提取指定面部的所有特征点的(x, y, z)相对坐标
    return np.array([[landmark.x, landmark.y, landmark.z]
                     for landmark in results.multi_face_landmarks[face_index].landmark])


def get_points_array2d(results, face_index):
    """
    根据指定的面部索引，从multi_face_landmarks对象中获取单个面部的所有二维特征点，并返回一个NumPy数组。
    二维数据用于计算机视觉使用，相比三维的处理函数减轻了一定的计算量

    参数:
    :param results -- 包含多个面部特征点信息的对象，通常由面部检测或识别函数返回。
    :param face_index -- 需要获取特征点的面部索引，即第几张人脸。

    返回:
    :return 一个NumPy数组，其中每一行代表一个点的(x, y)相对坐标。
    """
    # 提取指定面部的所有特征点的(x, y)相对坐标
    return np.array([[landmark.x, landmark.y]
                     for landmark in results.multi_face_landmarks[face_index].landmark])


def get_points_img_array3d(results, face_index, img):
    """
    根据给定的面部特征检测结果，提取某个面部的所有特征点在图像中的位置。

    该函数主要用于将特征点的规范化坐标转换为像素坐标，并返回一个包含所有特征点坐标的数组。
    这在需要对图像中的具体面部特征进行操作或分析时非常有用。

    参数:
    :param results: 面部特征检测的结果，包含多个面部的特征点信息。
    :param face_index: 指定要提取特征点的面部在结果中的索引。
    :param img: 输入的图像，应为一个三维数组，代表图像的高、宽和颜色通道。

    返回:
    :return 一个三维数组，其中每个子数组代表一个特征点的(x, y, z)坐标。
    """
    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 提取指定面部的所有特征点的(x, y, z)绝对坐标
    return np.array([[landmark.x * w, landmark.y * h, landmark.z]
                     for landmark in results.multi_face_landmarks[face_index].landmark])


def get_points_img_array2d(results, face_index, img):
    """
    根据给定的面部特征检测结果，提取某个面部的所有特征点在图像中的坐标。

    参数:
    :param results: 面部特征检测的结果对象，包含多个面部的特征点信息。
    :param face_index: 需要提取特征点的面部在结果中的索引。
    :param img: 输入的图像数组，用于根据图像尺寸调整特征点的坐标。

    返回:
    :return 一个二维数组，其中每个子数组代表面部的一个特征点的(x, y)坐标。
    """
    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 提取指定面部的所有特征点的(x, y)绝对坐标
    return np.array([[landmark.x * w, landmark.y * h]
                     for landmark in results.multi_face_landmarks[face_index].landmark])


def get_special_points_array3d(results, face_index, point_id_list):
    """
    根据给定的面部 landmarks 结果，提取特定点的三维坐标，并返回一个NumPy数组。

    参数:
    :param results: 包含多个人脸 landmarks 信息的对象。
    :param face_index: 用于指定哪一张脸的 landmarks 信息将被使用。
    :param point_id_list: 一个包含点ID的列表，用于指定需要提取坐标的点。

    返回:
    :return 一个NumPy数组，其中每一行代表一个点的 (x, y, z) 相对坐标。
    """
    # 使用列表推导式，遍历点ID列表，从指定的人脸 landmarks 中提取每个点的坐标
    return np.array([[results.multi_face_landmarks[face_index].landmark[point_id].x,
                      results.multi_face_landmarks[face_index].landmark[point_id].y,
                      results.multi_face_landmarks[face_index].landmark[point_id].z]
                     for point_id in point_id_list])


def get_special_points_array2d(results, face_index, point_id_list):
    """
    根据给定的面部 landmarks 结果，提取特定点的二维坐标，并返回一个NumPy数组。

    参数:
    :param results: 包含多个人脸 landmarks 信息的对象。
    :param face_index: 用于指定哪一张脸的 landmarks 信息将被使用。
    :param point_id_list: 一个包含点ID的列表，用于指定需要提取坐标的点。

    返回:
    :return 一个NumPy数组，其中每一行代表一个点的 (x, y) 坐标。
    """
    # 使用列表推导式，遍历点ID列表，从指定的人脸 landmarks 中提取每个点的坐标
    return np.array([[results.multi_face_landmarks[face_index].landmark[point_id].x,
                      results.multi_face_landmarks[face_index].landmark[point_id].y]
                     for point_id in point_id_list])


def get_special_points_img_array3d(results, face_index, point_id_list, img):
    """
    根据给定的面部 landmarks 结果，提取特定点的三维坐标，并返回一个NumPy数组。

    参数:
    :param results: 包含多个人脸 landmarks 信息的对象。
    :param face_index: 用于指定哪一张脸的 landmarks 信息将被使用。
    :param point_id_list: 一个包含点ID的列表，用于指定需要提取坐标的点。

    返回:
    :return 一个NumPy数组，其中每一行代表一个点的 (x, y, z) 坐标。
    """
    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 使用列表推导式，遍历点ID列表，从指定的人脸 landmarks 中提取每个点的坐标
    return np.array([[results.multi_face_landmarks[face_index].landmark[point_id].x * w,
                      results.multi_face_landmarks[face_index].landmark[point_id].y * h,
                      results.multi_face_landmarks[face_index].landmark[point_id].z]
                     for point_id in point_id_list])


def get_special_landmark_by_point_id_list(landmarks_list, point_id_list):
    """
    根据点ID列表从地标列表中获取特定点。

    此函数的目的是从一组地标数据中提取特定的点。它通过遍历地标列表，
    并根据提供的点ID列表来筛选出我们感兴趣的点。这在处理大量地标数据
    并需要特定点进行进一步分析或处理时非常有用。

    参数:
    :param landmarks_list: 地标数据列表，每个地标包含一组点。
    :param point_id_list: 需要提取的点的ID列表。

    返回:
    :return 包含所有指定点的数组。
    """
    # 使用列表推导式和numpy的array函数来构建包含指定点的数组
    # 这里使用tqdm是为了在处理大量数据时提供一个进度条，以便于监控进程
    return np.array([landmarks[point_id_list] for landmarks in tqdm(landmarks_list)])


def get_special_points_img_array2d(results, face_index, point_id_list, img):
    """
    根据给定的面部 landmarks 结果，提取特定点的二维坐标，并返回一个NumPy数组。

    参数:
    :param results: 包含多个人脸 landmarks 信息的对象。
    :param face_index: 用于指定哪一张脸的 landmarks 信息将被使用。
    :param point_id_list: 一个包含点ID的列表，用于指定需要提取坐标的点。

    返回:
    :return 一个NumPy数组，其中每一行代表一个点的 (x, y) 绝对坐标。
    """
    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 使用列表推导式，遍历点ID列表，从指定的人脸 landmarks 中提取每个点的坐标
    return np.array([[results.multi_face_landmarks[face_index].landmark[point_id].x * w,
                      results.multi_face_landmarks[face_index].landmark[point_id].y * h]
                     for point_id in point_id_list])


def find_circle_point(point_a, point_b, distance_cb):
    """
    根据已知点 A, B 的坐标和 CB 间的距离计算满足 ∠ACB = 90° 的点 C 的坐标。

    参数:
    :param point_a: 点 A 的坐标 (x_a, y_a, z_a)，可以是形如 [[x_a, y_a, z_a]] 的数组。
    :param point_b: 点 B 的坐标 (x_b, y_b, z_b)，可以是形如 [[x_b, y_b, z_b]] 的数组。
    :param distance_cb: 点 C 与 B 之间的距离。

    返回:
    :return C: 满足条件的点 C 的坐标
    """
    # 确保 point_a 和 point_b 都是一维数组
    point_a = np.array(point_a).ravel()
    point_b = np.array(point_b).ravel()

    # 计算向量 AB 和其模长（距离 AB）
    vector_ab = point_b - point_a
    distance_ab = np.linalg.norm(vector_ab)

    # 处理 A 和 B 重合的情况
    if distance_ab == 0:
        raise ValueError("A and B cannot be the same point")

    # 单位向量 AB
    unit_ab = vector_ab / distance_ab

    # 找到一个与单位向量 AB 垂直的向量 v
    if unit_ab[0] != 0 or unit_ab[1] != 0:
        v = np.array([-unit_ab[1], unit_ab[0], 0])
    else:
        v = np.array([1, 0, 0])

    # 避免除以零的情况
    if np.linalg.norm(v) == 0:
        v = np.array([1, 0, 0])
    v = v / np.linalg.norm(v)  # 单位化

    # 构造 w，使得 w 与单位向量 AB 和 v 都垂直
    w = np.cross(unit_ab, v)
    w = w / np.linalg.norm(w)  # 单位化

    # 计算 C 的坐标 (取 θ=0, 即 C = B + d_CB * v)
    return point_b + distance_cb * v


def get_new_origin_img_3d(results, face_index, img):
    """
    将垂直于降眉肌且与左右眼泪阜连线所在的垂直平面的相交点记作新坐标系的原点。
    计算并返回图像中面部指定两点的中点位置，以及沿深度轴偏移的新原点位置。
    这个计算的选择直接决定了投影的平面。

    参数:
    :param results: 包含面部 landmarks 信息的结果对象。
    :param face_index: 指定的面部索引，用于从 results 中提取特定面部的 landmarks。
    :param img: 原始图像数据，用于获取图像的宽高。

    返回:
    :return center_point + np.array([0, 0, elevation]): 返回面部指定两点的中点位置，以及沿深度轴根据 elevation 偏移的新原点位置。
    """

    # 分别获取左眼和右眼泪阜的两个关键点的坐标
    point1 = get_special_points_img_array3d(results, face_index, [133], img)
    point2 = get_special_points_img_array3d(results, face_index, [362], img)

    # 获取降眉肌坐标
    depressor = get_special_points_img_array3d(results, face_index, [168], img)

    # 计算两眼泪阜到降眉肌铅锤方向的欧式距离
    elevation = np.linalg.norm(point1 - point2) * RATIO_EYES_NOSE

    # 计算两眼泪阜中点坐标
    center_point = (point1 + point2) / 2

    # 计算线性变换后的原点坐标
    new_origin = find_circle_point(depressor, center_point, elevation)

    return new_origin, point2 - point1, depressor - new_origin


def get_new_origin_img_3d_points_list(points_list):
    """
    根据给定的点列表计算新的三维空间点列表。

    该函数首先通过调用`get_special_points_by_point_id_list`函数，根据特定的点ID筛选出一组特殊的点。
    然后，对于每个筛选出的特殊点组，利用其坐标信息和预定义的比例，计算出新的三维空间点。

    注意，该函数只能处理单一的面部情况，不对 face_index 进行设置。

    参数:
    :param points_list: 一个包含具有时间序列相关点的信息的列表，每个点通过其ID和坐标信息进行标识（要求传入的ID不被打乱）。

    返回:
    :return 一个numpy数组，包含所有计算得到的新三维空间点。
    """

    # 筛选出特殊的点，作为后续计算新三维空间点的基础
    keys = get_special_landmark_by_point_id_list(points_list, [133, 362, 168])

    # 遍历每个特殊点组，计算并返回一个新的三维空间点列表
    result = []

    for key in tqdm(keys):
        point1, point2, depressor = key

        # 计算新三维空间点的高程，基于两点距离和预定义比例
        elevation = np.linalg.norm(point1 - point2) * RATIO_EYES_NOSE

        # 计算两点的中点，作为新三维空间点的参考中心
        center_point = (point1 + point2) / 2

        # 根据计算出的高程和中点，找到新的三维空间点
        new_origin = find_circle_point(depressor, center_point, elevation)

        # 将计算出的新三维空间点及其向量信息添加到结果列表中
        result.append([new_origin, point2 - point1, depressor - new_origin])

    # 返回所有计算得到的新三维空间点的numpy数组
    return np.array(result)


def get_video_dimensions(video_path):
    """
    获取视频的宽度和高度。

    输入：
    :param video_path: 视频文件的路径

    输出：
    :return (width, height) 元组，表示视频的宽度和高度
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        raise IOError("无法打开视频文件")

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 释放资源
    cap.release()

    return width, height


def find_bounding_rectangle_from_subset(points: np.ndarray, indices, tolerance: float = 5.0) -> np.ndarray:
    """
    根据点集的部分点（由序号集合决定）计算xoy平面上的最小矩形顶点，并根据容差值调整矩形范围。

    参数:
    :param points: np.ndarray - 形状为 (n, 3) 的 numpy 数组，表示 n 个三维点的坐标 (x, y, z)
    :param indices: list 或 np.ndarray - 点集中部分点的序号集合，用来挑选要计算的子集点
    :param tolerance: float - 容差值，用来调整矩形的范围，默认为 0（即最小矩形）

    返回:
    :return np.ndarray - 形状为 (4, 2) 的 numpy 数组，表示矩形的四个顶点坐标 (x, y)
    """
    # 如果 indices 不是 list 或 numpy 数组，则转换为 numpy 数组
    indices = np.array(indices)

    # 提取指定序号的子集点
    subset_points = points[indices, :2]  # 只考虑 x 和 y 坐标，忽略 z 坐标

    # 找到 x 和 y 的最小值和最大值
    x_min, y_min = np.min(subset_points, axis=0)
    x_max, y_max = np.max(subset_points, axis=0)

    # 根据容差值调整矩形范围
    x_min -= tolerance
    y_min -= tolerance
    x_max += tolerance
    y_max += tolerance

    # 构建矩形的四个顶点，按顺时针或逆时针排列
    rectangle = np.array([
        [x_min, y_min],  # 左下角
        [x_min, y_max],  # 左上角
        [x_max, y_max],  # 右上角
        [x_max, y_min]  # 右下角
    ])

    return rectangle
