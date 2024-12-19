from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from lib.seetaface.api import *
import cv2

# 初始化人脸检测器
init_mask = FACE_DETECT
find_face = SeetaFace(init_mask)

# 设置人脸检测器的属性
find_face.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE, 50)
find_face.SetProperty(DetectProperty.PROPERTY_THRESHOLD, 0.6)

# 导入 YOLO 模型
model = YOLO('../lib/recognition/weights/ecare_emotion_yolo8s.pt')


def process_emotion_realtime():
    """
    实时处理摄像头中的人脸情绪识别。
    读取摄像头视频流，进行人脸检测和情绪识别，并实时显示结果。
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧（流结束？）")
            break

        try:
            # 检测人脸
            detect_result = find_face.Detect(frame)

            # 遍历检测到的人脸
            for i in range(detect_result.size):
                face = detect_result.data[i].pos
                x, y, w, h = face.x, face.y, face.width, face.height

                # 裁剪面部图像
                face_image = frame[y:y + h, x:x + w]

                # 将裁剪的面部图像转换为 PIL 图像
                face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

                # 进行情绪识别
                results = model(face_pil)
                result = results[0]

                # 遍历检测到的情绪
                for det in result.boxes:
                    x_min, y_min, x_max, y_max = det.xyxy[0]  # 坐标
                    conf = det.conf  # 置信度
                    cls = det.cls  # 类别ID
                    class_name = result.names[cls[0].item()]  # 类别名称
                    print(f"- Box coordinates: {x_min}, {y_min}, {x_max}, {y_max}.\n"
                          f"- Confidence: {conf}.\n"
                          f"- Class Name: {class_name}.\n")

                    # 在原图上绘制情绪识别框
                    cv2.rectangle(frame, (x + int(x_min), y + int(y_min)), (x + int(x_max), y + int(y_max)),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x + int(x_min), y + int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print('Error processing frame:', e)

        # 显示结果帧
        cv2.imshow('Emotion Detection', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cv2.destroyAllWindows()
    cap.release()
    print('Finished!')


if __name__ == '__main__':
    process_emotion_realtime()
