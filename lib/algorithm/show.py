import cv2
from matplotlib import pyplot as plt


# 定义可视化图像函数
def look_img(img):
    """
    显示图像

    该函数的主要作用是读取并显示一张图片。它首先需要转换图片的颜色通道，
    因为OpenCV读取图片默认的颜色通道顺序是BGR，而Matplotlib显示图片默认的颜色通道顺序是RGB。

    参数:
    :param img: ndarray, 由OpenCV读取的图像数组

    返回值:
    无
    """
    # 将BGR颜色空间转换为RGB颜色空间，以适应Matplotlib的显示要求
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用Matplotlib显示转换后的图像
    plt.imshow(img_RGB)
    plt.show()


def show_point(img, results, retract, point_id):
    """
    在图像上标记出指定的关键点。

    该函数用于在给定的图像上绘制出人脸关键点的位置。它首先计算关键点的图像坐标，
    然后在图像上标记一个红色的点来指示该关键点的位置。

    参数:
    :param img: 输入的图像。
    :param results: 人脸检测和关键点估计的结果。
    :param retract: int, 指定要标记的脸部序号。
    :param point_id: int, 指定要标记的关键点的ID。

    返回:
    :return numpy.ndarray, 标记了关键点的图像。
    """
    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # 获取指定关键点的坐标
    x = int(results.multi_face_landmarks[retract].landmark[point_id].x * w)
    y = int(results.multi_face_landmarks[retract].landmark[point_id].y * h)
    # 在图像上标记关键点位置
    img = cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    return img


def show_points(img, results, retract, point_id_list):
    """
    在图像上标记指定的点。

    该函数根据给定的图像、结果集、回退参数和点ID列表，在图像上标记出相应的点。
    它遍历点ID列表，对于每个点ID，调用show_point函数来在图像上进行标记。
    这里的标记是基于给定的结果集和回退参数进行的，确保点能够准确地绘制在图像上。

    参数:
    :param img: 输入的图像。
    :param results: dict 包含点信息的结果集，用于获取点的位置信息。
    :param retract: int 回退参数，用于调整点标记的位置，以更精确地在图像上标记点。
    :param point_id_list: list of int 需要在图像上标记的点的ID列表。

    返回:
    :return numpy.ndarray 标记了指定点的图像。注意，这是在原始图像上进行修改后的图像。
    """
    # 遍历点ID列表，对于每个点ID，调用show_point函数来在图像上进行标记
    for point_id in point_id_list:
        img = show_point(img, results, retract, point_id)
    return img
