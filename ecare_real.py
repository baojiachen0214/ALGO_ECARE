from ecare import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

FACE_INDEX = 0
GRADIENT_INTERVAL = 1  # 控制梯度图处理中取用的图像间隔

model = model_video()
cap = cv2.VideoCapture(0)

# 创建一个可以调整大小的窗口
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)
upper_result = []
last_result = []
frame_counts = 0


def display_frame(frame):
    cv2.putText(frame, 'Press \'q\' to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame', frame)


def figure_to_nparray(fig):
    """
    将 matplotlib.figure.Figure 转换为 NumPy 数组
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty frame")
            break

        try:
            results, frame = process_frame(model, frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        last_result = [results, frame]

        if frame_counts == 0:
            # 第一帧时初始化 upper_result
            upper_result = last_result
        elif frame_counts % GRADIENT_INTERVAL == 0:
            if upper_result and last_result and \
                    hasattr(upper_result[0], 'multi_face_landmarks') and hasattr(last_result[0],
                                                                                 'multi_face_landmarks') and \
                    upper_result[0].multi_face_landmarks and last_result[0].multi_face_landmarks:

                points_array1 = transform_coordinates_default(upper_result[0], FACE_INDEX, upper_result[1])
                points_array2 = transform_coordinates_default(last_result[0], FACE_INDEX, last_result[1])

                gradient_fig = gradient_2d(points_array1, points_array2,
                                           config=GradientFC(
                                               fixed_min=0,
                                               fixed_max=15,
                                               show_axes=True,
                                               show_colorbar=True,
                                               show_title=True)
                                           )

                if isinstance(gradient_fig, plt.Figure):
                    gradient_img = figure_to_nparray(gradient_fig)
                    # 将 RGB 图像转换为 BGR 图像
                    gradient_img = cv2.cvtColor(gradient_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow('gradient', gradient_img)
                else:
                    print(f"gradient_img is not a valid image: {type(gradient_fig)}")

                upper_result = last_result

        display_frame(frame)
        if cv2.waitKey(1) in [27, ord('q')]:
            break

        frame_counts += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
