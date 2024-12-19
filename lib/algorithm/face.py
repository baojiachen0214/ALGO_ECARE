from lib.seetaface.api import *


def face_recognition(recognize_frame, character):
    """
    进行人脸识别并计算两张图片中人脸的相似度。

    参数:
    :param recognize_frame: 待识别的图像帧。
    :param character: 用于比较的人脸图像。

    返回:
    :return similar: 两张人脸图像的相似度。
    """

    # 启用多个功能人脸检测、人脸识别和关键点标记
    init_mask = FACE_DETECT | FACERECOGNITION | LANDMARKER5
    seetaface = SeetaFace(init_mask)

    # 提取识别的人脸特征
    detect_result1 = seetaface.Detect(recognize_frame)
    face1 = detect_result1.data[0].pos  # 脸部位置信息face1
    points1 = seetaface.mark5(recognize_frame, face1)  # 标记face1区域的5个关键点
    feature1 = seetaface.Extract(recognize_frame, points1)  # 关键点的脸部特征提取

    # 人脸数据比对
    detect_result2 = seetaface.Detect(character)
    face2 = detect_result2.data[0].pos
    points2 = seetaface.mark5(character, face2)
    feature2 = seetaface.Extract(character, points2)

    # 计算两个特征值的形似度
    similar = seetaface.CalculateSimilarity(feature1, feature2)

    return similar