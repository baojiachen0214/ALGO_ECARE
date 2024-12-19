import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# 导入人脸关键点检测模型
mp_face_mesh = mp.solutions.face_mesh
model = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # 是否是静态图片
    refine_landmarks=True,
    max_num_faces=5,  # 最大人脸数量
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# 导入可视化函数和可视化样式
mp_drawing = mp.solutions.drawing_utils  # 画关键点
# 关键点可视化样式
landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(66, 77, 229))
# 轮廓可视化样式
contour_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(223, 155, 6))

from lib.seetaface.api import *


# 人脸识别比对，返回两张图片的相似度
def face_recognition(recognize_frame, character):
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


# 计算两个关键点之间的距离
import numpy as np


def distance_landmark(img, results, retract, point_id1, point_id2):
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


# 处理单帧的函数
def process_frame(img):
    # 记录该祯开始处理的时间
    t0 = time.time()
    scaler = 1  # 文字大小

    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # BGR to RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将RGB图像输入模型，并得到检测结果
    results = model.process(img_RGB)

    # 如果有检测到人脸，则开始绘制人脸曲面和重点区域轮廓线
    if results.multi_face_landmarks:  # 如果有检测到人脸
        for face_landmarks in results.multi_face_landmarks:  # 遍历每一张人脸
            # 绘制人脸轮廓线
            mp_drawing.draw_landmarks(
                image=img,  # 输入图像
                landmark_list=face_landmarks,  # 关键点
                connections=mp_face_mesh.FACEMESH_CONTOURS,  # 连接点
                landmark_drawing_spec=landmark_drawing_spec,  # 关键点样式,默认为None
                connection_drawing_spec=contour_drawing_spec  # 连接点样式
            )

            # 遍历关键点，添加序号
            for idx, coord in enumerate(face_landmarks.landmark):  # 遍历关键点
                cx = int(coord.x * w)
                cy = int(coord.y * h)
                # 图片、添加的文字、左上角的坐标、字体、字体大小、颜色、字体粗细
                img = cv2.putText(img, 'Face Detected', (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                                  1.25 * scaler, (255, 0, 255), 1)
                img = cv2.putText(img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scaler, (0, 255, 0), 1)
    else:
        # 如果没有检测到人脸，则提示
        # No face detected的提示只能是英文，否则会显示?????
        img = cv2.putText(img, 'No face detected!', (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)

    # 判断是否为闭上右眼
    if distance_landmark(frame, results, 0, 159, 145) <= 5:
        print("right eye was closed")
    else:
        print("right eye was opened")


    # 记录该帧处理完毕的时间
    t1 = time.time()

    # 计算每秒处理图像帧数FPS
    FPS = 1 / (t1 - t0)

    img = cv2.putText(img, 'FPS: ' + str(int(FPS)), (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                      1.25 * scaler, (255, 0, 255))

    return img


if __name__ == '__main__':

    # 测试模块——用于测试识别结果是否正确输出
    # 调用摄像头获取每帧
    # 导入opencv读取视频的库
    import cv2
    import time
    from lib.seetaface.api import *

    try:
        # 获取摄像头，传入0表示获取系统默认摄像头。如果是MacOS，则需要传入1
        cap = cv2.VideoCapture(0)

        # 打开capture
        cap.open(0)

        # 限制人脸识别的频率
        frame_counts = 0

        # 匹配人脸数据
        img = cv2.imread("../users_database/baojiachen.jpg")

        # 无线循环，直至断开连接
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty")
                break
            # 人脸识别
            # similar = face_recognition(frame, img)
            # print(similar)

            # 处理一帧
            frame = process_frame(frame)

            # 显示处理后的帧
            cv2.imshow('frame', frame)

            # 等待键盘输入，27为esc键，ord用于将字符转为ASCII码
            if cv2.waitKey(1) in [27, ord('q')]:
                break
    except Exception as e:
        print("Error!")
    finally:
        # 释放摄像头
        cap.release()

        # 关闭所有窗口
        cv2.destroyAllWindows()
