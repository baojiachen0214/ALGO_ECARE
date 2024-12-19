import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import gaussian_kde
from lib.algorithm.convert import *


class GradientFC(FigureConfig):
    def __init__(self,
                 scale_factor: float = 1.0,
                 colormap: str = 'hot',
                 interpolation_method: str = 'cubic',
                 set_figure_size: tuple = (10, 8),
                 fixed_min: float = -10,
                 fixed_max: float = 10,
                 x_lim: tuple = (0, 0),
                 y_lim: tuple = (0, 0),
                 show_axes: bool = True,
                 show_colorbar: bool = True,
                 show_title: bool = True):
        super().__init__(scale_factor=scale_factor,
                         colormap=colormap,
                         interpolation_method=interpolation_method,
                         set_figure_size=set_figure_size,
                         fixed_min=fixed_min,
                         fixed_max=fixed_max,
                         x_lim=x_lim,
                         y_lim=y_lim,
                         show_axes=show_axes,
                         show_colorbar=show_colorbar,
                         show_title=show_title)


class DivergenceFC(FigureConfig):
    def __init__(self,
                 scale_factor: float = 1.0,
                 colormap: str = 'viridis',
                 interpolation_method: str = 'cubic',
                 set_figure_size: tuple = (10, 8),
                 fixed_min: float = -10,
                 fixed_max: float = 10,
                 x_lim: tuple = (0, 0),
                 y_lim: tuple = (0, 0),
                 show_axes: bool = True,
                 show_colorbar: bool = True,
                 show_title: bool = True):
        super().__init__(scale_factor=scale_factor,
                         colormap=colormap,
                         interpolation_method=interpolation_method,
                         set_figure_size=set_figure_size,
                         fixed_min=fixed_min,
                         fixed_max=fixed_max,
                         x_lim=x_lim,
                         y_lim=y_lim,
                         show_axes=show_axes,
                         show_colorbar=show_colorbar,
                         show_title=show_title)


class StrainFieldFC(FigureConfig):
    def __init__(self,
                 scale_factor: float = 1.0,
                 colormap: str = 'jet',
                 interpolation_method: str = 'cubic',
                 set_figure_size: tuple = (10, 8),
                 fixed_min: float = -10,
                 fixed_max: float = 10,
                 x_lim: tuple = (0, 0),
                 y_lim: tuple = (0, 0),
                 show_axes: bool = True,
                 show_colorbar: bool = True,
                 show_title: bool = True,
                 set_t: float = 1,
                 set_sigma: float = 1):
        super().__init__(scale_factor=scale_factor,
                         colormap=colormap,
                         interpolation_method=interpolation_method,
                         set_figure_size=set_figure_size,
                         fixed_min=fixed_min,
                         fixed_max=fixed_max,
                         x_lim=x_lim,
                         y_lim=y_lim,
                         show_axes=show_axes,
                         show_colorbar=show_colorbar,
                         show_title=show_title)
        self.set_t = set_t
        self.set_sigma = set_sigma


class GradientVC(VideoConfig):
    def __init__(self,
                 categories: list = None,
                 fps: int = 24,
                 filename: str = "gradient-video",
                 video_format: str = 'mp4'):
        super().__init__(categories=categories,
                         fps=fps,
                         filename=filename,
                         video_format=video_format)


class DivergenceVC(VideoConfig):
    def __init__(self,
                 categories: list = None,
                 fps: int = 24,
                 filename: str = "divergence-video",
                 video_format: str = 'mp4'):
        super().__init__(categories=categories,
                         fps=fps,
                         filename=filename,
                         video_format=video_format)


class StrainFieldVC(VideoConfig):
    def __init__(self,
                 categories: list = None,
                 fps: int = 24,
                 filename: str = "strain-field-video",
                 video_format: str = 'mp4'):
        super().__init__(categories=categories,
                         fps=fps,
                         filename=filename,
                         video_format=video_format)


# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 手动定义颜色映射
cmap_data = {
    'red': [(0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 1.0, 1.0)],
    'green': [(0.0, 1.0, 1.0),
              (0.5, 1.0, 1.0),
              (1.0, 0.0, 0.0)],
    'blue': [(0.0, 1.0, 1.0),
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)]
}

cmap = LinearSegmentedColormap('RdYlGn_r', cmap_data)


def get_motion_vectors(points_array1, points_array2, start_time=0, end_time=1):
    """
    计算两个时间点之间的运动向量。

    通过计算两个数组中对应点的差值，得到物体在两个时间点的运动向量。

    参数:
    :param points_array1: 第一个时间点的点坐标数组，形状为 (n, 3) 的 numpy 数组。
    :param points_array2: 第二个时间点的点坐标数组，形状应与 points_array1 相同。
    :param start_time: 开始时间，默认值为0。
    :param end_time: 结束时间，默认值为1。

    返回值:
    :return motion_vectors: 运动向量数组，即 points_array2 中的点相对于 points_array1 中点的变动数组。
    """
    # 类型检查
    if not isinstance(points_array1, np.ndarray) or not isinstance(points_array2, np.ndarray):
        raise TypeError("points_array1 and points_array2 must be numpyArray")

    # 形状一致性检查
    if points_array1.shape != points_array2.shape:
        raise ValueError("points_array1 and points_array2 must have the same shape")

    # 除零保护
    if end_time - start_time == 0:
        raise ValueError("The time interval between end_time and start_time must not be zero.")

    # 计算运动向量
    try:
        motion_vectors = (points_array2 - points_array1) / (end_time - start_time)
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating the motion vector: {e}")

    return motion_vectors


def draw_motion_vectors(points_array1, points_array2, scale_factor=1, arrow_length=0.1):
    """
    绘制两个点集之间的三维运动向量。

    参数:
    :param points_array1: 起始点数组，形状为 (n, 3)
    :param points_array2: 结束点数组，形状为 (n, 3)
    :param scale_factor: 向量长度的缩放因子
    :param arrow_length: 图中的箭头长度

    """
    # 验证输入形状
    if points_array1.shape != points_array2.shape or points_array1.ndim != 2 or points_array1.shape[1] != 3:
        raise ValueError("points_array1 and points_array2 must have the same shape and be (n, 3).")

    # 计算运动向量及其大小
    motion_vectors = points_array2 - points_array1
    magnitudes = np.linalg.norm(motion_vectors, axis=1)

    # 对向量大小进行归一化以用于颜色映射
    norm = Normalize(vmin=0, vmax=np.max(magnitudes))

    # 设置 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 计算坐标和缩放
    xs, ys, zs = points_array1.T
    dxs, dys, dzs = (motion_vectors.T * scale_factor)

    # 根据向量大小映射颜色
    colors = cmap(norm(magnitudes))

    # 绘制向量，并通过设置最小长度确保可见性
    min_length = 0.05  # 最小向量长度以确保可见性
    scale_dxs = np.where(dxs < min_length, min_length, dxs)
    scale_dys = np.where(dys < min_length, min_length, dys)
    scale_dzs = np.where(dzs < min_length, min_length, dzs)

    # 通过调整绘图范围确保向量在边界内
    ax.quiver(xs, ys, zs, scale_dxs, scale_dys, scale_dzs,
              color=colors, length=arrow_length, normalize=True)

    # 调整绘图范围以提高可见性
    ax.set_xlim(np.min(xs) - 1, np.max(xs) + 1)
    ax.set_ylim(np.min(ys) - 1, np.max(ys) + 1)
    ax.set_zlim(np.min(zs) - 1, np.max(zs) + 1)

    # 添加标签和标题
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Motion vector visualization')

    plt.show()


def draw_projection(points_array1, points_array2, plane='xoy', scale_factor=1):
    """
    绘制两个点集之间的运动矢量在指定平面上的投影。

    参数:
    :param points_array1 : ndarray 第一个点集数组，每个点由一个三维向量表示。
    :param points_array2 : ndarray 第二个点集数组，与points_array1对应，每个点由一个三维向量表示。
    :param plane : {'xoy', 'zoy', 'xoz'}, 可选运动矢量投影到的平面，默认为 'xoy'。
    :param scale_factor : float, 可选运动矢量的比例因子，默认为1。

    """
    # 计算运动矢量
    motion_vectors = get_motion_vectors(points_array1, points_array2)

    # 计算运动矢量的大小
    magnitudes = np.linalg.norm(motion_vectors, axis=1)
    norm = Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))

    # 创建绘图对象
    fig, ax = plt.subplots()

    # 根据选择的平面设置坐标轴
    if plane == 'xoy':
        x, y = points_array1[:, 0], points_array1[:, 1]
        dx, dy = motion_vectors[:, 0] * scale_factor, motion_vectors[:, 1] * scale_factor
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    elif plane == 'zoy':
        y, z = points_array1[:, 1], points_array1[:, 2]
        dy, dz = motion_vectors[:, 1] * scale_factor, motion_vectors[:, 2] * scale_factor
        ax.set_xlabel('Y-axis')
        ax.set_ylabel('Z-axis')
    elif plane == 'xoz':
        x, z = points_array1[:, 0], points_array1[:, 2]
        dx, dz = motion_vectors[:, 0] * scale_factor, motion_vectors[:, 2] * scale_factor
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')

    # 在选定平面上绘制矢量投影
    ax.quiver(x, y, dx, dy, color=cmap(norm(magnitudes)), angles='xy', scale_units='xy', scale=1)
    ax.set_aspect('equal', 'box')  # 设置2D投影的等比例

    plt.show()


def get_rallying_point(points_array, motion_vectors_array):
    """
    计算集合点的位置。

    该函数通过将运动向量数组与点数组相加并沿第0轴求和，然后除以点数组的数量，
    来估计集合点的位置。

    参数:
    :param points_array: 点的数组，形状为(n, 3)，其中n是点的数量。
    :param motion_vectors_array: 运动向量数组，形状与points_array相同。

    返回值:
    :return p_estimated: 估计的集合点位置，形状为(2,)。
    """

    # 点数组的行数即为点的数量
    n = points_array.shape[0]

    # 处理除零错误
    if n == 0:
        raise ValueError("The number of points must be greater than zero.")

    return np.sum(motion_vectors_array + points_array, axis=0) / n


def get_work_value(points_array, motion_vectors_array):
    """
    计算给定点数组和运动向量数组的做功值。

    做功值是通过将运动向量数组中的每个向量与对应的点相乘，
    然后将结果相加并除以点的数量得到的平均值。

    参数:
    :param points_array: 一个二维数组，其中每行代表一个点。
    :param motion_vectors_array: 一个二维数组，其中每行代表一个运动向量。

    返回:
    :return 做功值: 一个表示整个数组的平均运动量（强度）的数组。

    异常:
    :except ValueError: 如果点的数量为零，无法计算做功值。
    """

    # 点数组的行数即为点的数量
    n = points_array.shape[0]

    # 处理除零错误
    if n == 0:
        raise ValueError("The number of points must be greater than zero.")

    # 计算做功值并返回
    return np.sum(motion_vectors_array * points_array, axis=0) / n


def compute_divergence(points_array, motion_vectors_array):
    """
    计算点集的发散性。

    参数：
    :param motion_vectors_array: 点集的速度，形状为 (n, 3) 的 numpy 数组
    :param points_array: 点集的位置，形状为 (n, 3) 的 numpy 数组

    返回：
    :return ∇v(t)：发散性（浮点数）

    异常:
    :except ValueError: 如果点的数量为零，无法计算做功值。
    """
    # 验证输入数据类型和形状
    if not isinstance(points_array, np.ndarray) or not isinstance(motion_vectors_array, np.ndarray):
        raise ValueError("Inputs must be NumPy arrays")

    if points_array.shape != motion_vectors_array.shape:
        raise ValueError("Shapes of points_array and motion_vectors_array must match")

    if points_array.shape[1] != 3:
        raise ValueError("Arrays must have shape (n, 3)")

    # 初始化发散性
    divergence = 0
    n = len(points_array)

    # 处理除零错误
    if n == 0:
        raise ValueError("The number of points must be greater than zero.")

    # 计算发散性
    for i in range(1, n - 1):
        dvx_dx = ((motion_vectors_array[i + 1, 0] - motion_vectors_array[i - 1, 0]) /
                  (points_array[i + 1, 0] - points_array[i - 1, 0]))
        dvy_dy = ((motion_vectors_array[i + 1, 1] - motion_vectors_array[i - 1, 1]) /
                  (points_array[i + 1, 1] - points_array[i - 1, 1]))
        dvz_dz = ((motion_vectors_array[i + 1, 2] - motion_vectors_array[i - 1, 2]) /
                  (points_array[i + 1, 2] - points_array[i - 1, 2]))
        divergence += dvx_dx + dvy_dy + dvz_dz

    return divergence / n


def if_extrusion(points_array1, points_array2, start_time=0, end_time=1):
    """
    判断点集在时间段内是扩散还是收缩

    参数：
    :param points_array1: 时间 t 的点集，形状为 (n, 3) 的 numpy 数组
    :param points_array2: 时间 t+Δt 的点集，形状为 (n, 3) 的 numpy 数组
    :param start_time: 开始时间
    :param end_time: 结束时间

    返回：
    :return 'Expanding', 'Contracting' 或 'No Change'
    """

    # 计算速度场
    velocities = get_motion_vectors(points_array1, points_array2, start_time, end_time)

    # 计算发散性
    divergence = compute_divergence(velocities, points_array1)

    # 根据发散性判断
    if divergence > 0:
        return "Expanding"  # 发散，扩散
    elif divergence < 0:
        return "Contracting"  # 收缩
    else:
        return "No Change"  # 没有变化


def estimate_convergence_point(a_list, v_list):
    """
    估计点集的聚集/发散点 p（针对更一般的情况）

    参数：
    :param a_list：形状为 (n, 3) 的 numpy 数组，表示点的位置 a_i(t)
    :param v_list：形状为 (n, 3) 的 numpy 数组，表示点的速度 v_i(t)

    返回：
    :return p_estimated：形状为 (3,) 的 numpy 数组，估计的聚集/发散点 p
    """
    n = a_list.shape[0]
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(n):
        a_i = a_list[i]
        v_i = v_list[i]
        v_i_norm = np.linalg.norm(v_i)
        if v_i_norm == 0:
            continue  # 跳过速度为零的点
        v_hat = v_i / v_i_norm  # 单位速度向量
        P_i = np.eye(3) - np.outer(v_hat, v_hat)  # 投影矩阵
        A += P_i
        b += P_i @ a_i

    # 检查矩阵 A 是否可逆
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError(
            "The matrix A is not of rank and cannot uniquely determine the convergence/divergence point p.")

    p_estimated = np.linalg.solve(A, b)
    return p_estimated


def draw_displacement_gradient_linear_3d(points_array1, points_array2, scale_factor=1,
                                         colormap=cmap, set_figure_size=(10, 8)):
    """
    计算位移场的梯度，并生成 3D 可视化图像。

    参数:
    :param points_array1: 第一个时间点的点坐标数组，形状为 (n, 3) 的 numpy 数组。
    :param points_array2: 第二个时间点的点坐标数组，形状应与 points_array1 相同。
    :param scale_factor: 缩放因子，用于调整位移向量显示的大小。
    :param colormap: 自定义颜色映射。
    :param set_figure_size: 图像大小，默认(10, 8)。
    """
    # 计算运动向量
    displacement_vectors = points_array2 - points_array1
    gradients = np.gradient(displacement_vectors, axis=0)  # 按行计算梯度

    # 计算梯度大小（合并每个维度的梯度以获取总大小）
    gradient_magnitude = np.sqrt(sum([g ** 2 for g in gradients]))  # 累加不同轴上的平方

    # 归一化梯度大小，以便在颜色映射中使用
    norm = Normalize(vmin=np.min(gradient_magnitude), vmax=np.max(gradient_magnitude))
    colors = colormap(norm(gradient_magnitude))

    # 可视化梯度场
    fig = plt.figure(figsize=set_figure_size)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制位移向量及其梯度
    xs, ys, zs = points_array1[:, 0], points_array1[:, 1], points_array1[:, 2]
    dx, dy, dz = displacement_vectors[:, 0] * scale_factor, displacement_vectors[:,
                                                            1] * scale_factor, displacement_vectors[:, 2] * scale_factor

    ax.quiver(xs, ys, zs, dx, dy, dz, color=colors, length=0.1, normalize=True)

    # 设置图像边界
    ax.set_xlim([xs.min() - 1, xs.max() + 1])
    ax.set_ylim([ys.min() - 1, ys.max() + 1])
    ax.set_zlim([zs.min() - 1, zs.max() + 1])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Displacement field gradient analysis')

    plt.show()


def draw_displacement_gradient_linear_2d(points_array1, points_array2, scale_factor=1,
                                         colormap=cmap, set_figure_size=(10, 8)):
    """
    在 XOY 平面上投影位移场的梯度，并生成 2D 可视化图像。

    参数:
    :param points_array1: 第一个时间点的点坐标数组，形状为 (n, 3) 的 numpy 数组。
    :param points_array2: 第二个时间点的点坐标数组，形状应与 points_array1 相同。
    :param scale_factor: 缩放因子，用于调整位移向量显示的大小。
    :param colormap: 自定义颜色映射。
    :param set_figure_size: 图像大小，默认(10, 8)。
    """
    # 计算运动向量
    disp_vectors = disp_field(points_array1, points_array2)
    gradients = np.gradient(disp_vectors, axis=0)

    # 计算梯度大小
    gradient_magnitude = np.sqrt(sum([g ** 2 for g in gradients]))

    # 归一化梯度大小，以便在颜色映射中使用
    norm = Normalize(vmin=np.min(gradient_magnitude), vmax=np.max(gradient_magnitude))
    colors = colormap(norm(gradient_magnitude))

    # 2D 投影 (XOY 平面)
    fig, ax = plt.subplots(figsize=set_figure_size)
    xs, ys = points_array1[:, 0], points_array1[:, 1]
    dx, dy = disp_vectors[:, 0] * scale_factor, disp_vectors[:, 1] * scale_factor

    # 使用颜色绘制位移向量
    ax.quiver(xs, ys, dx, dy, color=colors, scale=1, scale_units='xy', angles='xy')

    # 设置图像边界
    ax.set_xlim([xs.min() - 1, xs.max() + 1])
    ax.set_ylim([ys.min() - 1, ys.max() + 1])
    ax.set_aspect('equal')  # 保持横纵比例一致
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2D projection of displacement field gradient analysis in XOY plane')

    plt.show()


def hot_img(points_array1, points_array2, pro_func, config: FigureConfig = None):
    """
    绘制热力图。

    本函数通过计算两个点集之间的位移向量，并利用这些信息生成一个热力图。

    参数:
    :param points_array1: 第一个点集，作为位移计算的起始点。
    :param points_array2: 第二个点集，作为位移计算的结束点。
    :param pro_func: 处理函数，用于计算热力图所需的场值。
    :param config: 图形配置对象，包含图形绘制的各种配置信息。

    返回:
    :return fig: 绘制的图形对象。
    """
    # 计算位移向量
    vectors = points_array2 - points_array1
    # 提取x和y坐标
    xs, ys = points_array1[:, 0], points_array1[:, 1]

    # 调用处理函数计算场值和获取图形标题等信息
    config, field, title, label = pro_func(config, vectors, xs, ys)

    # 使用griddata进行插值，以生成平滑的场
    if config.x_lim == (0, 0):
        config.x_lim = (min(xs), max(xs))
    if config.y_lim == (0, 0):
        config.y_lim = (min(ys), max(ys))

    grid_x, grid_y = np.mgrid[min(xs):max(xs):100j, min(ys):max(ys):100j]
    grid_z = griddata((xs, ys), field, (grid_x, grid_y), method=config.interpolation_method)

    # 创建热力图
    fig, ax = plt.subplots(figsize=config.set_figure_size)
    heatmap = ax.imshow(grid_z.T * config.scale_factor,
                        extent=(config.x_lim[0], config.x_lim[1], config.y_lim[0], config.y_lim[1]),
                        origin='lower', cmap=config.colormap, vmin=config.fixed_min, vmax=config.fixed_max)

    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if not config.show_axes:
        ax.axis('off')

    if config.show_title:
        ax.set_title(title)

    if config.show_colorbar:
        cbar = plt.colorbar(heatmap, ax=ax, shrink=0.75)
        cbar.set_label(label)

    return fig


def hot_img_strain_field(strain, title, extent, config):
    """绘制应变场"""
    fig, ax = plt.subplots(figsize=config.set_figure_size)
    heatmap = ax.imshow(strain.T * config.scale_factor,
                        extent=extent,
                        cmap=config.colormap,
                        origin='lower',
                        vmin=config.fixed_min, vmax=config.fixed_max)

    ax.set_aspect('equal')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    if not config.show_axes:
        ax.axis('off')

    if config.show_title:
        ax.set_title(title)

    if config.show_colorbar:
        fig.colorbar(heatmap, ax=ax, label='Strain')

    return fig


def strain_field(points_array1, points_array2, config: StrainFieldFC = None):
    """
    处理并计算应变场。

    该函数首先计算两点集之间的位移场，然后将位移场和点集投影到二维平面上，
    通过插值方法将散点数据插值到网格上，最后计算并绘制水平和垂直应变场。

    参数:
    :param points_array1: 第一个点集，通常代表初始位置。
    :param points_array2: 第二个点集，通常代表变形后的位置。
    :param config: 配置对象，包含应变场计算的配置参数，如插值方法等。

    返回:
    :return fig_horizontal: 水平应变场的图像对象。
    :return fig_vertical: 垂直应变场的图像对象。
    """
    # 计算位移场
    displacement_field = disp_field(points_array1, points_array2, config.set_t)

    # 获取二维投影的坐标点
    projection_points1 = points_array1[:, [0, 1]]
    projection_displacement = displacement_field[:, [0, 1]]

    # 基于二维平面将点投影到网格
    x_min, x_max = np.min(projection_points1[:, 0]), np.max(projection_points1[:, 0])
    y_min, y_max = np.min(projection_points1[:, 1]), np.max(projection_points1[:, 1])
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # 将散点数据插值到网格
    displacement_x = griddata(projection_points1, projection_displacement[:, 0],
                              (grid_x, grid_y), method=config.interpolation_method)
    displacement_y = griddata(projection_points1, projection_displacement[:, 1],
                              (grid_x, grid_y), method=config.interpolation_method)

    # 计算应变场
    horizontal_strain, vertical_strain = clc_strain(displacement_x, displacement_y, config.set_sigma)

    # 绘制应变场
    fig_horizontal = hot_img_strain_field(horizontal_strain, 'Horizontal Strain Field',
                                          (x_min, x_max, y_min, y_max), config)
    fig_vertical = hot_img_strain_field(vertical_strain, 'Vertical Strain Field',
                                        (x_min, x_max, y_min, y_max), config)

    return fig_horizontal, fig_vertical


def disp_field(points_array1, points_array2, set_t=1):
    """
    计算位移场

    参数:
    :param points_array1: (numpy.ndarray): 第一组点的坐标数组
    :param points_array2: (numpy.ndarray): 第二组点的坐标数组
    :param set_t: (float): 时间间隔

    返回:
    :return numpy.ndarray: 位移场
    """
    # 除零错误处理
    if set_t == 0:
        raise ValueError("The interval set_t cannot be zero")

    try:
        # 计算位移场
        field = (points_array2 - points_array1) / set_t
    except Exception as e:
        raise RuntimeError(f"Error occurred while calculating displacement field: {e}")

    return field


def clc_strain(displacement_x, displacement_y, set_sigma):
    """计算应变场

        通过位移场计算应变场。使用Sobel算子计算x和y方向上的位移梯度，
        然后应用高斯滤波进行平滑处理，以减少噪声影响。

    参数:
    :param displacement_x: x方向上的位移场
    :param displacement_y: y方向上的位移场
    :param set_sigma: 高斯滤波器的sigma值，用于控制平滑程度

    返回:
    :return smoothed_horizontal_strain: 平滑处理后的水平应变场
    :return smoothed_vertical_strain: 平滑处理后的垂直应变场
    """
    # 分别计算水平、垂直方向上的应变
    horizontal_strain = sobel(displacement_x, axis=1)  # 沿水平方向
    vertical_strain = sobel(displacement_y, axis=0)  # 沿垂直方向

    # 分别对水平、垂直方向的应变进行高斯平滑处理
    smoothed_horizontal_strain = gaussian_filter(horizontal_strain, sigma=set_sigma)
    smoothed_vertical_strain = gaussian_filter(vertical_strain, sigma=set_sigma)
    # 返回平滑处理后的应变场
    return smoothed_horizontal_strain, smoothed_vertical_strain


def process_algorithm(array_list, process_func, output_dir="./output_img/", **kwargs):
    """
    使用指定的处理函数处理数组列表中的元素，并生成图像保存到输出目录。

    参数:
    :param array_list: (list): 包含要处理的数组的列表。
    :param process_func: (function): 用于处理数组的函数。
    :param output_dir: (str): 保存输出图像的目录路径，默认为"./output_img/"。
    **kwargs: 传递给处理函数的额外关键字参数。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历数组列表，应用处理函数，并保存生成的图像
    for i in tqdm(range(len(array_list) - 1)):
        # 调用传入的处理函数，处理相邻两个数组元素
        fig = process_func(array_list[i], array_list[i + 1], **kwargs)
        # 构建输出图像的路径
        output_path = os.path.join(output_dir, f"frame_{i}.png")
        # 保存图像并关闭图像对象以释放资源
        fig.savefig(output_path)
        plt.close(fig)


def clc_gradient(config: GradientFC, vectors, xs, ys):
    """
    计算梯度场。

    参数:
    :param config: 配置对象，包含计算所需的配置信息。
    :param vectors: 位移向量集。
    :param xs: 点集的x坐标。
    :param ys: 点集的y坐标。

    返回:
    :return config: 更新后的配置对象。
    :return field: 计算得到的场值。
    :return title: 图形标题。
    :return label: 颜色条标签。
    """
    if config is None:
        config = GradientFC()
    # 计算场值
    gradient_vectors = np.gradient(vectors, axis=1)
    field = np.linalg.norm(gradient_vectors, axis=1)
    title = 'XOY plane displacement gradient field thermal map'
    label = 'gradient field strength'
    return config, field, title, label


def clc_divergence(config: DivergenceFC, vectors, xs, ys):
    """
    计算散度场。

    参数:
    :param config: 配置对象，包含计算所需的配置信息。
    :param vectors: 位移向量集。
    :param xs: 点集的x坐标。
    :param ys: 点集的y坐标。

    返回:
    :return config: 更新后的配置对象。
    :return field: 计算得到的场值。
    :return title: 图形标题。
    :return label: 颜色条标签。
    """
    if config is None:
        config = DivergenceFC()

    # 计算散度场
    dx, dy = vectors[:, 0], vectors[:, 1]
    div_x = np.gradient(dx, xs, axis=0)
    div_y = np.gradient(dy, ys, axis=0)
    field = div_x + div_y
    title = 'XOY plane displacement divergence field thermal map'
    label = 'Divergence field strength'
    return config, field, title, label


def gradient_2d(points_array1, points_array2, config: GradientFC = None):
    """
    在XOY平面上绘制位移场梯度的2D热力图。

    参数:
    :param points_array1: 第一张图像的点数组。
    :param points_array2: 第二张图像的点数组。
    :param config: 配置对象，包含热力图的各种配置信息，默认为None。如果为None，则使用默认配置。

    返回:
    :return fig: 绘制的热力图对象。
    """
    return hot_img(points_array1, points_array2, clc_gradient, config)


def divergence_2d(points_array1, points_array2, config: DivergenceFC = None):
    """
    在XOY平面上绘制散度的2D热力图。

    参数:
    :param points_array1: 第一张图像的点数组。
    :param points_array2: 第二张图像的点数组。
    :param config: 配置对象，包含热力图的各种配置信息，默认为None。如果为None，则使用默认配置。

    返回:
    :return fig: 绘制的热力图对象。
    """
    return hot_img(points_array1, points_array2, clc_divergence, config)


def process_algo_strain_field(array_list, output_dir="./asserts/output_img_strain/", config: StrainFieldFC = None):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建临时目录
    temp_horizon_dir = os.path.join(output_dir, "strain_field_horizon")
    temp_vertical_dir = os.path.join(output_dir, "strain_field_vertical")
    os.makedirs(temp_horizon_dir, exist_ok=True)
    os.makedirs(temp_vertical_dir, exist_ok=True)

    # 遍历数组列表，应用处理函数，并保存生成的图像
    for i in tqdm(range(len(array_list) - 1)):
        # 调用传入的处理函数，处理相邻两个数组元素
        fig_horizontal, fig_vertical = strain_field(array_list[i], array_list[i + 1], config=config)

        # 构建输出图像的路径
        output_path_horizontal = os.path.join(temp_horizon_dir, f"strain_field_horizontal_frame_{i}.png")
        output_path_vertical = os.path.join(temp_vertical_dir, f"strain_field_vertical_frame_{i}.png")

        # 保存图像并关闭图像对象以释放资源
        fig_horizontal.savefig(output_path_horizontal)
        plt.close(fig_horizontal)
        fig_vertical.savefig(output_path_vertical)
        plt.close(fig_vertical)

    return temp_horizon_dir, temp_vertical_dir


def process_algo_gradient_2d(array_list, output_dir="./asserts/output_img_gradient/", config: GradientFC = None):
    """
    使用二维梯度算法处理输入的数组列表。

    参数:
    :param array_list: (list of np.ndarray): 需要处理的数组列表，每个数组代表一个数据样本。
    :param output_dir: (str): 输出目录的路径，默认为"./asserts/output_img_gradient/"。该目录将存储处理后的图像。
    :param config: (GradientFC): 配置对象，包含热力图的各种配置信息，默认为None。如果为None，则使用默认配置。

    返回:
    无返回值。处理结果会直接保存到指定的输出目录。
    """
    # 调用process_algorithm函数进行二维梯度处理，该函数封装了通用的处理逻辑
    process_algorithm(array_list, gradient_2d, output_dir=output_dir, config=config)


def process_algo_divergence_2d(array_list, output_dir="./asserts/output_img_divergence/", config: DivergenceFC = None):
    """
    使用二维散度算法处理输入的数组列表。

    参数:
    :param array_list: (list of np.ndarray): 需要处理的数组列表，每个数组代表一个数据样本。
    :param output_dir: (str): 输出目录的路径，默认为"./output_img_divergence/"。该目录将存储处理后的图像。
    :param config: (DivergenceFC): 配置对象，包含热力图的各种配置信息，默认为None。如果为None，则使用默认配置。

    返回:
    无返回值。处理结果会直接保存到指定的输出目录。
    """
    process_algorithm(array_list, divergence_2d, output_dir=output_dir, config=config)


def get_statistics(points_array1, points_array2):
    """
    计算位移并返回描述性统计数据。

    参数：
    :param points_array1: (ndarray): 初始点集 (N, 3)。
    :param points_array2: (ndarray): 变换后的点集 (N, 3)。

    返回：
    :return dict: 描述性统计数据，包括均值、标准差、最大值、最小值。
    """
    # 确保输入为 numpy 数组
    points_array1 = np.array(points_array1)
    points_array2 = np.array(points_array2)

    # 计算位移
    displacements = points_array2 - points_array1

    # 计算描述性统计
    stats = {
        'mean': np.mean(displacements, axis=0),
        'std_dev': np.std(displacements, axis=0),
        'max': np.max(displacements, axis=0),
        'min': np.min(displacements, axis=0)
    }

    return stats


def plot_statistics(stats):
    """
    绘制描述性统计量的图表。

    参数：
    :param stats: (dict): 描述性统计数据，包括均值、标准差、最大值、最小值和中位数。
    """
    labels = ['X', 'Y', 'Z']
    mean_values = stats['mean']
    std_dev_values = stats['std_dev']
    median_values = stats['median']

    # 条形图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))  # 标签位置
    width = 0.2  # 条形宽度

    plt.bar(x - width, mean_values, width, label='Mean', color='b', yerr=std_dev_values, capsize=5)
    plt.bar(x, median_values, width, label='Median', color='g')
    plt.bar(x + width, stats['max'], width, label='Max', color='r')
    plt.bar(x + 2 * width, stats['min'], width, label='Min', color='y')

    # 添加标签和标题
    plt.ylabel('Displacement')
    plt.title('Displacement Statistics')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid()
    plt.show()


def plot_displacement_distribution(points_array1, points_array2):
    """
    绘制位移分布的直方图和KDE曲线。

    参数：
    :param points_array1: (ndarray): 初始点集 (N, 3)。
    :param points_array2: (ndarray): 变换后的点集 (N, 3)。
    """
    # 计算位移数组
    displacements = points_array2 - points_array1

    # 计算位移的绝对值
    abs_displacements = np.linalg.norm(displacements, axis=1)

    # 创建直方图
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(abs_displacements, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')

    # 计算KDE
    kde = gaussian_kde(abs_displacements)
    x = np.linspace(min(abs_displacements), max(abs_displacements), 100)
    plt.plot(x, kde(x), color='r', linewidth=2)

    # 添加标题和标签
    plt.title('Displacement Distribution')
    plt.xlabel('Displacement Magnitude')
    plt.ylabel('Density')

    # 显示图例
    plt.legend(['KDE', 'Histogram'])
    plt.grid()

    # 展示图形
    plt.show()


def plot_displacement_scatter(points_array1, points_array2):
    """
    绘制位移的散点图。

    参数：
    :param points_array1: (ndarray): 初始点集 (N, 3)。
    :param points_array2: (ndarray): 变换后的点集 (N, 3)。
    """
    displacements = points_array2 - points_array1
    plt.figure(figsize=(10, 8))

    plt.scatter(displacements[:, 0], displacements[:, 1], c=displacements[:, 2], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Displacement in Z-axis')
    plt.title('Displacement Scatter Plot')
    plt.xlabel('Displacement in X-axis')
    plt.ylabel('Displacement in Y-axis')
    plt.grid()
    plt.show()
