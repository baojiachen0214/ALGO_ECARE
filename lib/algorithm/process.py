import time
from PIL import Image
import lib.algorithm.preload as model_process
from lib.algorithm.movement import *
from lib.algorithm.preload import *


def crop_image_by_rectangle(image, rectangle) -> np.ndarray:
    """
    根据传入的矩形顶点坐标裁剪图像，并返回裁剪后的 BGR 格式图片。

    参数:
        image: 图像 (可以是 NumPy 数组或 PIL 图像对象)
        rectangle: np.ndarray - 形状为 (4, 2) 的 numpy 数组，表示矩形的四个顶点坐标 (x, y)

    返回:
        np.ndarray - 裁剪后的 BGR 格式图像
    """
    # 确保 rectangle 是 numpy 数组
    rectangle = np.array(rectangle)

    # 如果 image 是 NumPy 数组，转换为 PIL 图像对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 找到矩形的边界框
    x_min = np.min(rectangle[:, 0])
    y_min = np.min(rectangle[:, 1])
    x_max = np.max(rectangle[:, 0])
    y_max = np.max(rectangle[:, 1])

    # 将坐标值转换为整数，因为像素坐标需要整数
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    # 使用 PIL 的 crop 方法进行裁剪
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # 将裁剪后的图像转换为 RGB 格式的 NumPy 数组
    cropped_image_np = np.array(cropped_image)

    # 转换 RGB 图像为 BGR 格式
    cropped_image_bgr = cropped_image_np[:, :, ::-1]

    return cropped_image_bgr


def calculate_average_grayscale(img):
    """
    计算给定图像的平均灰度值。

    参数:
        img: PIL.Image.Image对象，表示输入的图像。

    返回:
        float: 图像的平均灰度值。
    """
    # 打开图片并转换为灰度模式
    pro_img = img.convert('L')

    # 将图片转换为NumPy数组
    img_array = np.array(pro_img)

    # 计算平均灰度值
    avg_grayscale = np.mean(img_array)

    return avg_grayscale


def process_frame(model, img):
    """
    处理单个图像帧，包括人脸检测和关键点标注。

    参数:
        model: 用于人脸检测和关键点检测的模型。
        img: 输入的图像帧，以BGR格式表示。

    返回:
        经过处理的图像帧，包含人脸曲面和关键点标注。
    """
    # 记录该帧开始处理的时间
    t0 = time.time()
    scaler = 1  # 文字大小

    # 获取图像的宽高
    h, w = img.shape[0], img.shape[1]

    # BGR to RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将RGB图像输入模型，并得到检测结果
    results = model.process(img_RGB)

    # 设置样式点和输出样式，这里具体调用的函数和作用需要根据上下文或模型相关文档来明确
    [style_point, style_out] = model_process.style_setting_video(0)

    # 如果有检测到人脸，则开始绘制人脸曲面和重点区域轮廓线
    if results.multi_face_landmarks:  # 如果有检测到人脸
        for face_landmarks in results.multi_face_landmarks:  # 遍历每一张人脸
            # 绘制人脸轮廓线
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,  # 输入图像
                landmark_list=face_landmarks,  # 关键点
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # 连接点
                landmark_drawing_spec=style_point,  # 关键点样式,默认为None
                connection_drawing_spec=style_out  # 连接点样式
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

    # 记录该帧处理完毕的时间
    t1 = time.time()

    # 计算每秒处理图像帧数FPS
    FPS = 1 / (t1 - t0)

    # 在图像上添加FPS数值
    img = cv2.putText(img, 'FPS: ' + str(int(FPS)), (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                      1.25 * scaler, (255, 0, 255))

    return results, img


def process_img(model, img):
    """
    处理单张图片，检测并绘制人脸特征点和轮廓线
    注意，不得将此函数用于连续帧的处理。若需要连续帧处理，需使用 process_img_ 为前缀的函数

    参数:
        model: 用于人脸检测和特征点识别的模型，要求传入类型为用于处理 img 的模型
        img: 输入的图片，采用BGR格式

    返回值:
        results: 模型的检测结果，包含人脸特征点等信息
        img: 绘制了人脸轮廓线和特征点的图片
    """
    # BGR to RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将RGB图像输入模型，并得到检测结果
    results = model.process(img_RGB)

    # 设置人脸特征点和轮廓线的绘制样式
    [style_point, style_out] = model_process.style_setting_video(0)

    # 如果有检测到人脸，则开始绘制人脸曲面和重点区域轮廓线
    if results.multi_face_landmarks:  # 如果有检测到人脸
        for face_landmarks in results.multi_face_landmarks:  # 遍历每一张人脸
            # 绘制人脸轮廓线
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,  # 输入图像
                landmark_list=face_landmarks,  # 关键点
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # 连接点
                landmark_drawing_spec=style_point,  # 关键点样式,默认为None
                connection_drawing_spec=style_out  # 连接点样式
            )

    else:
        print("No face detected")
    return results, img


def get_result_img(model, image1, image2):
    """
    处理两张图片，将其作为输入通过模型处理后返回处理结果。

    参数:
        model: 处理模型，用于执行图像处理或生成任务。
        image1: 第一张图片，numpy数组格式。
        image2: 第二张图片，numpy数组格式。

    返回:
        results_list[0], results_list[1]: 返回两张经过模型处理后的图片。
    """

    # 创建视频文件路径
    file_path = create_video_file()

    # 定义临时视频路径
    temp_video_path = file_path + 'temp_video.avi'

    # 视频尺寸和编码设置
    frame_size = (image1.shape[1], image1.shape[0])
    out = cv2.VideoWriter(temp_video_path, 0, 1, frame_size)  # 使用默认编码器

    # 写入两张图片作为视频帧
    out.write(image1)
    out.write(image2)
    out.release()

    # 使用 generate_video 的逻辑来处理短视频并获取结果
    results_list, _, _ = generate_video(model, input_path=temp_video_path)

    # 删除临时视频文件
    delete_files(path=file_path, patterns=['temp_video'], pa_mode=False)

    # 返回处理后的图片结果
    return results_list[0], results_list[1]


def get_point_array(model, image1, image2, face_index):
    """
    获取两张图片中指定人脸的坐标数组。

    该函数首先对两张图片进行处理，检测并标记人脸的关键点。然后检查每张图片中是否至少检测到一张脸，
    如果没有检测到脸，则抛出错误。最后，函数计算并返回经过坐标系变换后的点坐标数组。

    参数:
    :param model: 用于处理图片并检测人脸的模型。
    :param image1: 第一张图片。
    :param image2: 第二张图片。
    :param face_index: 指定的人脸索引，用于区分多张脸。

    返回:
    :return points_array1, points_array2: 第一张图片和第二章中指定人脸的变换后坐标数组。
    """
    # 处理图片，返回处理后的坐标和图片
    result1, result2 = get_result_img(model, image1, image2)

    # 检查两张图片是否都检测到了人脸
    if not result1.multi_face_landmarks or not result2.multi_face_landmarks:
        raise ValueError("Both images must contain a detectable face.")

    # 计算经过坐标系变换的点坐标
    points_array1 = transform_coordinates_default(result1, face_index, image1)
    points_array2 = transform_coordinates_default(result2, face_index, image2)

    return points_array1, points_array2


def process_img_strain_field(model, image1, image2, face_index=0, config: StrainFieldFC = None):
    """
    计算位移场的梯度，并生成 3D 可视化图像。

    参数:
    :param model: 模型对象，用于计算位移场。
    :param image1: 第一张图像，作为位移场计算的参考图像。
    :param image2: 第二张图像，与参考图像进行比较以计算位移场。
    :param face_index: 默认为0，指定需要分析的面部索引。
    :param config: StrainFieldFC 类型的配置对象，用于自定义位移场计算的配置。如果未提供，则使用默认配置。

    返回:
        位移场的梯度计算结果，以及相应的 3D 可视化表示。
    """
    # 初始化配置，如果未提供则使用默认配置
    if config is None:
        config = StrainFieldFC()

    # 获取两个图像的点阵列，用于后续的位移场计算
    points_array1, points_array2 = get_point_array(model, image1, image2, face_index)

    # 调用 strain_field 函数计算位移场，并返回结果
    return strain_field(points_array1=points_array1, points_array2=points_array2, config=config)


def process_img_gradient_2d(model, image1, image2, face_index=0, config: GradientFC = None):
    """
    在XOY平面上绘制位移场梯度的2D热力图。

    该函数通过计算两张图像之间点的位移向量的梯度大小（应变场），并使用插值方法生成一个平滑的应变场，
    最后在该平面上绘制热力图，以直观展示位移场的梯度变化。

    参数:
        model: 模型，用于获取点数组的函数get_point_array。
        image1, image2: 两张需要处理的图像。
        face_index: 面的索引，用于区分多面对象中的特定面，默认为0。
        scale_factor: 缩放因子，用于调整热力图的显示比例，默认为1.0。
        colormap: 颜色映射，用于热力图的颜色显示，默认为 hot。
        set_figure_size: 图像的大小，默认为(20, 16)。

    返回:
        fig: 绘制的热力图对象。
    """
    # 获取两张图像的点数组
    points_array2, points_array1 = get_point_array(model, image1, image2, face_index)

    # 计算并绘制梯度2D热力图
    return gradient_2d(points_array1, points_array2, config=config)


def process_img_divergence_2d(model, image1, image2, face_index=0, config: DivergenceFC = None):
    """
    处理两张图像，计算散度并生成2D热力图。

    该函数通过模型提取两张图像中的关键点，计算这些关键点之间的散度，并生成相应的2D热力图。

    参数:
        model: 用于处理图像的模型。
        image1: 第一张图像。
        image2: 第二张图像。
        face_index: 需要处理的人脸索引，默认为0。
        config (DivergenceFC): 配置对象，包含热力图的各种配置信息，默认为None。如果为None，则使用默认配置。

    返回:
        fig: 生成的热力图Figure对象。
    """
    # 获取两张图像的点数组
    points_array2, points_array1 = get_point_array(model, image1, image2, face_index)

    # 计算并绘制散度2D热力图
    return divergence_2d(points_array1, points_array2, config)


def generate_video(model, face_index=0, input_path='../asserts/video2.mp4', show_detail=False):
    """
    生成处理过的人脸特征点视频。

    参数:
        model: 用于处理视频帧中人脸特征点的模型。
        face_index: 指定处理的面部索引，默认为0。
        input_path: 输入视频的路径，默认为'../asserts/video2.mp4'（测试用）。

    返回值:
        results_landmarks_list: 包含每一帧中指定面部特征点的列表。
        frame_size: 视频帧的尺寸（宽度，高度）。
    """
    file_path = create_video_file()
    results_list = []
    results_landmarks_list = []
    file_head = input_path.split('/')[-1]
    output_path = file_path + "out-" + file_head

    if show_detail:
        print('Video start processing.', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
    cap.release()

    if show_detail:
        print('The total number of frames:', frame_count)

    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                try:
                    results, frame = process_frame(model, frame)
                    results_list.append(results)
                    results_landmarks_list.append(results.multi_face_landmarks[face_index].landmark)
                except:
                    print('Error: An error occurred in frame processing of the video.')
                    pass

                if success:
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)
        except:
            print('Midway interruption.')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    if show_detail:
        print('Video saved.', output_path)
    return results_list, results_landmarks_list, frame_size


def process_algo_strain_field_v(array_list, temp_dir="./temp/", output_dir="./asserts/output_video/",
                                figure_config: StrainFieldFC = None, video_config: StrainFieldVC = None):
    """
    处理应变场数据并生成视频。

    该函数接收一个二维数组列表，处理每个数组以生成相应的应变场图像，然后将这些图像合成为视频。
    生成的图像和视频将根据给定的配置保存到指定的目录中。

    参数:
        array_list: 一个包含二维数组的列表，每个数组代表一个应变场数据帧。
        temp_dir: 保存输出缓存图像的目录，默认为"./temp/"，一般不建议修改默认缓存地址。
        output_dir: 保存输出视频的目录，默认为"./output_videos/"。
        figure_config: 一个StrainFieldFC配置对象，用于定制图像生成过程。如果未提供，则使用默认配置。
        video_config: 一个StrainFieldVC配置对象，用于定制视频生成过程。如果未提供，则使用默认配置。

    返回值:
        无直接返回值，但会生成并保存图像和视频到指定的目录。
    """
    # 初始化配置对象
    figure_config = figure_config or StrainFieldFC()
    video_config = video_config or StrainFieldVC()

    # 调用函数处理二维数组列表生成梯度，并返回地址
    temp_horizon_dir, temp_vertical_dir = process_algo_strain_field(array_list, output_dir=temp_dir,
                                                                    config=figure_config)

    # 生成水平方向视频
    video_config.filename = f"{video_config.filename}_horizon"
    images_to_video_ffmpeg(img_path=temp_horizon_dir, output_video_path=output_dir, config=video_config)

    # 生成垂直方向视频
    video_config.filename = f"{video_config.filename}_vertical"
    images_to_video_ffmpeg(img_path=temp_vertical_dir, output_video_path=output_dir, config=video_config)

    # 删除临时文件
    delete_files(path=temp_dir, extensions=[f'.{video_config.video_format}'], ex_mode=True, process_sub_dirs=True,
                 keep_dirs=False)


def process_algo_gradient_2d_v(array_list, temp_dir="./temp/", output_dir="./asserts/output_video/",
                               figure_config: GradientFC = None, video_config: GradientVC = None):
    """
    处理二维数组列表以生成梯度图并合成视频。

    该函数首先调用`process_algorithm_gradient_2d`处理二维数组列表生成梯度图，
    然后使用`images_to_video_ffmpeg`函数将这些图像合成视频。

    参数:
        array_list: 二维数组列表，用于生成梯度图。
        temp_dir: 输出缓存图像的目录，默认为"./temp/"，一般不建议修改默认缓存地址。
        output_dir: 输出视频的目录，默认为"./asserts/output_video/"。
        figure_config: 梯度图的配置对象，如果未提供，则使用默认的GradientFC。
        video_config: 视频的配置对象，如果未提供，则使用默认的GradientVC。
    """
    # 初始化梯度图配置和视频配置
    figure_config = figure_config or GradientFC()
    video_config = video_config or GradientVC()

    # 调用函数处理二维数组列表生成梯度图
    process_algo_gradient_2d(array_list, output_dir=temp_dir, config=figure_config)

    # 调用函数将梯度图合成视频
    images_to_video_ffmpeg(temp_dir, output_dir, config=video_config)

    # 删除指定目录下特定格式的文件
    delete_files(path=temp_dir, extensions=[f'.{video_config.video_format}'], ex_mode=True)


def process_algo_divergence_2d_v(array_list, temp_dir="./temp/", video_dir="./asserts/output_video/",
                                 figure_config: DivergenceFC = None, video_config: DivergenceVC = None):
    """
    处理二维数组列表并生成表示算法发散性的视频。

    该函数首先调用`process_algorithm_divergence_2d`处理二维数组列表并生成相应的图像，
    然后使用`images_to_video_ffmpeg`函数将这些图像合成视频，并最后删除生成图像文件。

    参数:
        array_list: 二维数组列表，包含需要处理的数据。
        temp_dir: 输出缓存图像的目录，默认为"./temp/"，一般不建议修改默认缓存位置
        video_dir: 输出视频的目录，默认为"./asserts/output_video/"。
        figure_config: 图像生成的配置对象，类型为DivergenceFC，默认为None。
        video_config: 视频合成的配置对象，类型为DivergenceVC，默认为None。
    """
    # 初始化图像生成配置和视频合成配置
    figure_config = figure_config or DivergenceFC()
    video_config = video_config or DivergenceVC()

    # 调用函数处理算法发散性并生成图像
    process_algo_divergence_2d(array_list, output_dir=temp_dir, config=figure_config)

    # 调用函数将梯度图合成视频
    images_to_video_ffmpeg(temp_dir, video_dir, config=video_config)

    # 删除图像文件，保留视频
    delete_files(path=temp_dir, extensions=[f'.{video_config.video_format}'], ex_mode=True)


def process_video(input_path, pro_func, faces_num=5, face_index=0,
                  detail=True, detection_value=0.5, tracking_value=0.5, **kwargs):
    """处理视频文件，检测并跟踪其中的人脸。

    参数:
      input_path (str): 视频文件的路径。
      pro_func (callable): 处理完成后调用的回调函数。
      faces_num (int): 需要检测的人脸数量。默认为5。
      face_index (int): 人脸的索引。默认为0。
      detail (bool): 是否返回详细信息。默认为True。
      detection_value (float): 人脸检测的阈值。默认为0.5。
      tracking_value (float): 人脸跟踪的阈值。默认为0.5。
      **kwargs: 其他传递给pro_func的关键字参数。

    返回值:
      无直接返回值。通过pro_func处理结果。
    """
    # 初始化视频处理模型
    model = model_process.model_video(
        faces_num=faces_num,
        detail=detail,
        detection_value=detection_value,
        tracking_value=tracking_value
    )

    # 生成处理后的视频结果
    results_lists, results_landmarks_list, frame_size = generate_video(
        model=model, face_index=face_index, input_path=input_path)

    # 将 landmarks 列表转换为数组列表
    array_list = transform_landmarks_list_to_array_list(
        results_landmarks_list, frame_size)

    # 调用回调函数处理数组列表
    pro_func(array_list, **kwargs)


def process_video_strain_field(input_path, temp_dir="./temp/", output_dir="./asserts/output_videos/",
                               faces_num=5, face_index=0, detail=True, detection_value=0.5, tracking_value=0.5,
                               figure_config: StrainFieldFC = None, video_config: StrainFieldVC = None):
    """
    处理视频中的应变场。

    该函数专门用于处理视频文件中的应变场，通过指定的算法进行分析和处理。
    它会根据检测和跟踪值来识别和处理视频中的面孔，并可以选择性地输出详细的处理结果。

    参数:
        input_path (str): 输入视频文件的路径。
        temp_dir (str): 处理缓存结果输出的目录，默认为 "./temp/"。
        ouput_dir (str): 视频输出的目录，默认为 "./asserts/output_videos/"。
        faces_num (int): 需要检测的面孔数量，默认为 5。
        face_index (int): 面孔的索引，默认为 0。
        detail (bool): 是否输出详细结果，默认为 True。
        detection_value (float): 面孔检测的阈值，默认为 0.5。
        tracking_value (float): 面孔跟踪的阈值，默认为 0.5。
        figure_config (StrainFieldFC): 图像处理配置对象，用于指定算法的配置参数。
        video_config (StrainFieldVC): 视频处理配置对象，用于指定视频处理的配置参数。

    返回值:
        None
    """
    # 调用通用的视频处理函数，传入特定的应变场处理算法
    process_video(input_path, process_algo_strain_field_v,
                  faces_num=faces_num, face_index=face_index,
                  detail=detail, detection_value=detection_value, tracking_value=tracking_value,
                  temp_dir=temp_dir, output_dir=output_dir,
                  figure_config=figure_config, video_config=video_config)


def process_video_gradient_2d(input_path, temp_dir="./temp/", output_dir="./asserts/output_videos/",
                              faces_num=5, face_index=0, detail=True, detection_value=0.5, tracking_value=0.5,
                              figure_config: GradientFC = None, video_config: GradientVC = None):
    """
    处理视频中的二维梯度场。

    该函数专门用于处理视频文件中的二维梯度场，通过指定的算法进行分析和处理。
    它会根据检测和跟踪值来识别和处理视频中的面孔，并可以选择性地输出详细的处理结果。

    参数:
        input_path (str): 输入视频文件的路径。
        output_dir (str): 处理结果输出的目录，默认为 "./output_videos/"。
        video_dir (str): 视频输出的目录，默认为 "./output_videos/"。
        faces_num (int): 需要检测的面孔数量，默认为 5。
        face_index (int): 面孔的索引，默认为 0。
        detail (bool): 是否输出详细结果，默认为 True。
        detection_value (float): 面孔检测的阈值，默认为 0.5。
        tracking_value (float): 面孔跟踪的阈值，默认为 0.5。
        figure_config (GradientFC): 图像处理配置对象，用于指定算法的配置参数。
        video_config (GradientVC): 视频处理配置对象，用于指定视频处理的配置参数。

    返回值:
        None
    """
    # 调用通用的视频处理函数，传入特定的二维梯度场处理算法
    process_video(input_path, process_algo_gradient_2d_v,
                  faces_num=faces_num, face_index=face_index,
                  detail=detail, detection_value=detection_value, tracking_value=tracking_value,
                  temp_dir=temp_dir, output_dir=output_dir,
                  figure_config=figure_config, video_config=video_config)


def process_video_divergence_2d(input_path, temp_dir="./temp/", output_dir="./asserts/output_videos/",
                                faces_num=5, face_index=0, detail=True, detection_value=0.5, tracking_value=0.5,
                                figure_config: DivergenceFC = None, video_config: DivergenceVC = None):
    """
    处理视频中的二维发散情况。

    该函数专门用于处理视频中二维发散的现象，它调用了通用的process_video函数来进行视频处理。
    通过指定输入路径、输出目录、视频目录、人脸数量、人脸索引、详细模式以及检测和跟踪值来配置处理过程。

    参数:
        input_path (str): 输入视频的路径。
        output_dir (str): 输出结果的目录，默认为"./output_videos/"。
        video_dir (str): 视频输出的目录，默认为"./output_videos/"。
        faces_num (int): 需要处理的人脸数量，默认为5。
        face_index (int): 指定处理的人脸索引，默认为0。
        detail (bool): 是否启用详细模式，默认为True。
        detection_value (float): 人脸检测的阈值，默认为0.5。
        tracking_value (float): 人脸跟踪的阈值，默认为0.5。
        figure_config (DivergenceFC): 用于配置图像处理的参数对象，默认为None。
        video_config (DivergenceVC): 用于配置视频处理的参数对象，默认为None。

    返回值:
        None
    """
    # 调用通用的process_video函数来进行视频处理，参数通过函数签名进行详细说明
    process_video(input_path, process_algo_divergence_2d_v,
                  faces_num=faces_num, face_index=face_index,
                  detail=detail, detection_value=detection_value, tracking_value=tracking_value,
                  temp_dir=temp_dir, output_dir=output_dir,
                  figure_config=figure_config, video_config=video_config)
