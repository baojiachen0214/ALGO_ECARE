import numpy as np
from scipy.spatial import distance


def distance_point(img, results, retract, point_id1, point_id2):
    """
    计算图像中两个面部关键点之间的欧式距离。

    参数:
    :param img -- 输入的图像。
    :param results -- 面部关键点检测的结果。
    :param retract -- 指定的面部序号。
    :param point_id1 -- 第一个关键点的ID。
    :param point_id2 -- 第二个关键点的ID。

    返回:
    :return dist -- 两个关键点之间的欧式距离。
    """
    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 获取指定面部的两个关键点的坐标
    landmark1 = results.multi_face_landmarks[retract].landmark[point_id1]
    landmark2 = results.multi_face_landmarks[retract].landmark[point_id2]

    # 将 landmark 的 x, y, z 转化为 numpy 数组
    point1 = np.array([landmark1.x * w, landmark1.y * h, landmark1.z])
    point2 = np.array([landmark2.x * w, landmark2.y * h, landmark2.z])

    # 计算欧式距离
    dist = np.linalg.norm(point1 - point2)

    return dist


def find_nearest_point(points: np.ndarray, ids, target_point: np.ndarray) -> int:
    """
    计算点集 `points` 中距离给定点 `target_point` 最近的点，并返回对应节点 ID 序号。

    参数:
    :param points: np.ndarray - 形状为 (n, 3) 的 numpy 数组，存储点集的坐标
    :param ids: list 或 np.ndarray - 序号标签数组，即面部点集的 ID 列表可以是普通的列表或 numpy 数组
    :param target_point: np.ndarray - 形状为 (3,) 的 numpy 数组，存储目标点的坐标

    返回:
    :return int - 距离 `target_point` 最近的点的序号
    """
    # 如果 `ids` 是列表，将其转换为 numpy 数组
    ids = np.array(ids)

    # 计算目标点和每个点的欧几里得距离
    distances = distance.cdist(points, target_point.reshape(1, -1), metric='euclidean').flatten()

    # 找到最小距离的索引
    nearest_index = np.argmin(distances)

    # 返回最近点的序号
    return ids[nearest_index]


def point_to_plane_distance(plane_normal, plane_point, point):
    """
    计算点到平面的距离。

    参数:
    :param plane_normal: 平面的法向量。
    :param plane_point: 平面上的一个点。
    :param point: 需要计算距离的点。

    返回值:
    :return 点到平面的垂直距离。
    """
    # 计算点到平面的垂直距离
    dist = np.dot(plane_normal, (point - plane_point)) / np.linalg.norm(plane_normal)
    return np.abs(dist)


def distance_to_plane(img, results, retract, plane_point_id1, plane_point_id2, plane_point_id3, point_id_list):
    """
    计算指定点到平面的距离。

    参数:
    :param img -- 输入的图像
    :param results -- 包含面部 landmarks 结果的对象
    :param retract -- 用于选择特定面部的索引
    :param plane_point_id1, plane_point_id2, plane_point_id3 -- 定义平面的三个点的 ID
    :param point_id_list -- 一个包含需要计算距离的点 ID 的列表

    返回:
    :return 一个 numpy 数组，包含每个点到平面的距离。
    """

    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 获取三个定义平面的点的坐标
    landmark1 = results.multi_face_landmarks[retract].landmark[plane_point_id1]
    landmark2 = results.multi_face_landmarks[retract].landmark[plane_point_id2]
    landmark3 = results.multi_face_landmarks[retract].landmark[plane_point_id3]

    # 将 landmark 的 x, y, z 坐标转化为 numpy 数组并调整比例
    point1 = np.array([landmark1.x * w, landmark1.y * h, landmark1.z])
    point2 = np.array([landmark2.x * w, landmark2.y * h, landmark2.z])
    point3 = np.array([landmark3.x * w, landmark3.y * h, landmark3.z])

    # 计算平面的法向量
    v1 = point2 - point1
    v2 = point3 - point1

    # 使用np.array_equal来比较两个数组是否完全相等
    if np.array_equal(v1, v2):
        return None
    else:
        plane_normal = np.cross(v1, v2)

    # 计算每个点到平面的距离
    distances = []
    for point_id in point_id_list:
        landmark = results.multi_face_landmarks[retract].landmark[point_id]
        point = np.array([landmark.x * w, landmark.y * h, landmark.z])
        dist = point_to_plane_distance(plane_normal, point1, point)
        distances.append(dist)

    return np.array(distances)

