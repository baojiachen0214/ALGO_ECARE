import os
import cv2
import mediapipe as mp


def model_video(faces_num=5, detail=True, detection_value=0.5, tracking_value=0.5):
    """
    构建基于MediaPipe的人脸关键点检测模型。

    参数:
    :param faces_num: (int): 预计检测的最大人脸数量，默认为5。
    :param detail: (bool, 可选): 是否细化关键点检测，默认为True。
    :param detection_value: (float, 可选): 人脸检测的最小置信度，默认为0.5。
    :param tracking_value: (float, 可选): 人脸跟踪的最小置信度，默认为0.5。

    返回:
    :return model: 初始化的MediaPipe人脸网格（FaceMesh）模型实例。
    """

    # 初始化人脸关键点检测模型，设置模型参数
    model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,  # 是否是静态图片
        refine_landmarks=detail,  # 是否细化关键点检测
        max_num_faces=faces_num,  # 最大人脸数量
        min_detection_confidence=detection_value,  # 设置人脸检测的最小置信度
        min_tracking_confidence=tracking_value  # 设置人脸跟踪的最小置信度
    )
    return model


def model_img(faces_num=5, detail=True, detection_value=0.5, tracking_value=0.5):
    """
    构建基于MediaPipe的人脸关键点检测模型。

    参数:
    :param faces_num: (int): 预计检测的最大人脸数量。
    :param detail: (bool, 可选): 是否细化关键点检测，默认为True。
    :param detection_value: (float, 可选): 人脸检测的最小置信度，默认为0.5。
    :param tracking_value: (float, 可选): 人脸跟踪的最小置信度，默认为0.5。

    返回:
    :return model: 初始化的MediaPipe人脸网格（FaceMesh）模型实例。
    """
    # 导入人脸关键点检测模型
    mp_face_mesh = mp.solutions.face_mesh
    # 初始化人脸关键点检测模型，设置模型参数
    model = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # 是否是静态图片
        refine_landmarks=detail,  # 是否细化关键点检测
        max_num_faces=faces_num,  # 最大人脸数量
        min_detection_confidence=detection_value,  # 设置人脸检测的最小置信度
        min_tracking_confidence=tracking_value  # 设置人脸跟踪的最小置信度
    )
    return model


def style_setting_video(style_num):
    """
    根据给定的样式编号，设置关键点和轮廓的可视化样式。

    参数:
    :param style_num: (int): 样式编号，目前函数中未根据此参数做任何改变。

    返回:
    :return style_point: 关键点的可视化样式。
    :return style_out: 轮廓的可视化样式。
    """
    # 当style_num等于0时
    if style_num == 0:
        # 关键点可视化样式
        style_point = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(66, 77, 229))
        # 轮廓可视化样式
        style_out = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(223, 155, 6))

        return style_point, style_out

    # 当style_num不等于0时
    else:
        # 关键点可视化样式
        style_point = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(66, 77, 229))
        # 轮廓可视化样式
        style_out = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(223, 155, 6))

        return style_point, style_out


def drawing_map():
    """
    初始化人体姿势估计的绘图工具和样式

    此函数旨在获取MediaPipe库中用于绘制人体关键点和连接的人体骨架图的工具和样式设置
    它返回绘图工具和绘图样式两个对象，以便在后续的人体姿势估计可视化中使用

    返回:
    :return mp_drawing: MediaPipe的绘图工具，用于绘制人体关键点和连接
    :return mp_drawing_styles: MediaPipe的绘图样式，用于设置关键点和连接的视觉样式
    """
    # 导入MediaPipe的绘图工具
    mp_drawing = mp.solutions.drawing_utils
    # 导入MediaPipe的绘图样式
    mp_drawing_styles = mp.solutions.drawing_styles

    # 返回绘图工具和绘图样式
    return mp_drawing, mp_drawing_styles


def img2rgb(img):
    """
    将图像从BGR格式转换为RGB格式。

    本函数的主要作用是转换图像颜色格式，以适应不同库的需要。在图像处理中，
    OpenCV库默认以BGR格式读取图像，而Matplotlib库的Imshow函数需要RGB格式的图像。
    因此，在使用Imshow函数展示图像前，需要将图像的格式从BGR转换为RGB。

    参数:
    :param img: numpy.ndarray 一个由OpenCV读取的BGR格式的图像数组。

    返回值:
    :return numpy.ndarray 转换后的RGB格式的图像数组。
    """
    # opencv读入的图像格式为BGR，而matplotlib的imshow函数需要RGB格式，因此需要转换下格式
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def faces(results):
    """
    获取面部特征标记

    该函数用于从给定的检测结果中提取面部特征标记信息

    参数:
    :param results: 调用人脸检测函数返回的结果对象，包含多种面部信息

    返回:
    :return multi_face_landmarks: 包含所有检测到面部的特征标记，用于进一步的面部分析和处理
    """
    return results.multi_face_landmarks


def create_video_file(video_file_path="./asserts/output/"):
    if not os.path.exists(video_file_path):
        os.makedirs(video_file_path)
    return video_file_path


def delete_files(path, extensions=None, patterns=None,
                 ex_mode=True, pa_mode=True,
                 process_sub_dirs=True, keep_dirs=True, show_details=False, confirm_delete=False):
    """
    删除指定目录下符合特定类型的文件或文件名模式的文件，并根据参数控制子目录的处理和目录的保留。
    显示要删除的文件数量、文件名称、文件类型，并让用户输入 y/n 来确认是否要删除。

    参数:
    :param path: (str): 要清理的目录路径。
    :param extensions: (list, 可选): 文件类型列表，例如 ['.txt', '.jpg']。默认为 None，表示不考虑文件类型。
    :param patterns: (list, 可选): 文件名模式列表，例如 ['temp', 'backup']。默认为 None，表示不考虑文件名模式。
    :param ex_mode: (bool, 可选): True 表示保留 `extensions` 中的文件类型，False 表示删除 `extensions` 中的文件类型。默认为 True。
    :param pa_mode: (bool, 可选): True 表示保留 `patterns` 中的文件名模式，False 表示删除 `patterns` 中的文件名模式。默认为 True。
    :param process_sub_dirs: (bool, 可选): 是否处理子目录。默认为 True。
    :param keep_dirs: (bool, 可选): 是否保留目录结构。默认为 True。
    :param show_details: (bool, 可选): 是否显示要删除的文件详情。默认为 False。
    :param confirm_delete: (bool, 可选): 是否需要用户确认删除。默认为 False。

    返回:
        无返回值。
    """
    # 规范化路径
    path = os.path.abspath(os.path.realpath(path))

    # 检查目录是否存在
    if not os.path.exists(path):
        print(f"Directory {path} does not exist")
        return

    # 用于存储要删除的文件列表
    files_to_delete = []

    def collect_files(dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)

            # 如果是文件且符合删除条件，则收集
            if os.path.isfile(item_path):
                # 检查文件扩展名是否在排除列表中
                if extensions is not None:
                    if ex_mode and any(item.endswith(ext) for ext in extensions):
                        continue
                    elif not ex_mode and not any(item.endswith(ext) for ext in extensions):
                        continue

                # 检查文件名是否包含排除模式
                if patterns is not None:
                    if pa_mode and any(pattern in item for pattern in patterns):
                        continue
                    elif not pa_mode and not any(pattern in item for pattern in patterns):
                        continue

                files_to_delete.append(item_path)
            # 如果是目录且需要处理子目录
            elif os.path.isdir(item_path) and process_sub_dirs:
                collect_files(item_path)

    # 收集要删除的文件
    collect_files(path)

    # 显示要删除的文件详情
    if show_details:
        print(f"Number of files to delete:{len(files_to_delete)}")
        for file_path in files_to_delete:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            print(f"File name: {file_name}, file type: {file_ext}")

    # 确认删除
    if confirm_delete:
        user_input = input("Are you sure to delete these files?(y/n): ")
        if user_input.lower() != 'y':
            print("Undelete operation.")
            return

    # 删除文件
    for file_path in files_to_delete:
        os.remove(file_path)

    # 删除空目录
    if not keep_dirs:
        for dir_path, _, _ in os.walk(path, topdown=False):
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
