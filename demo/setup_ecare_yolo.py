from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from lib.seetaface.api import *

# 初始化人脸检测器
init_mask = FACE_DETECT
find_face = SeetaFace(init_mask)

# 设置人脸检测器的属性
find_face.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE, 80)
find_face.SetProperty(DetectProperty.PROPERTY_THRESHOLD, 0.9)

# 导入 YOLO 模型
model = YOLO('../lib/recognition/weights/ecare_emotion_yolo8n.pt')


def process_emotion_v(video_path='../asserts/video1.mp4', output_path='../asserts/output/'):
    """
    处理视频中的人脸情绪识别。
    读取视频文件，进行人脸检测和情绪识别，并将结果保存到输出路径。

    参数:
        video_path: 视频文件路径。
        output_path: 处理结果保存路径。
    """
    # 提取视频文件名
    figurehead = video_path.split('/')[-1]
    # 构造输出文件路径
    output_path = output_path + 'emotion_out_' + figurehead

    # 打印视频路径
    print('video_path:', video_path)

    # 获取视频总帧数
    cap1 = cv2.VideoCapture(video_path)
    total_frames = 0
    while cap1.isOpened():
        ret, frame = cap1.read()
        total_frames += 1
        if not ret:
            break
    cap1.release()
    print('total_frames:', total_frames)

    # 初始化视频读取和写入对象
    cap2 = cv2.VideoCapture(video_path)
    frame_step = (cap2.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = int(cap2.get(cv2.CAP_PROP_FOURCC))
    fps = cap2.get(cv2.CAP_PROP_FPS)

    # 创建视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_step[0]), int(frame_step[1])))

    # 进度条绑定视频的总帧数
    with tqdm(total=total_frames - 1) as pbar:
        try:
            while cap2.isOpened():
                ret, frame = cap2.read()
                if not ret:
                    break

                # 处理帧
                try:
                    # 检测人脸
                    detect_result = find_face.Detect(frame)

                    # 遍历检测到的人脸
                    for i in range(detect_result.size):
                        face = detect_result.data[i].pos
                        x, y, w, h = face.x, face.y, face.width, face.height

                        # 裁剪面部图像
                        face_image = frame[y:y+h, x:x+w]

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

                if ret:
                    out.write(frame)
                    # 更新进度条
                    pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print('Midway interruption:', e)

    # 释放资源
    cv2.destroyAllWindows()
    out.release()
    cap2.release()
    print('Finished!', output_path)


if __name__ == '__main__':
    process_emotion_v()
