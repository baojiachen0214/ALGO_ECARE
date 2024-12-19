import os
import logging
import warnings
import cv2
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from typing import Optional
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from lib.algorithm.show import look_img
from lib.algorithm.preload import model_video, model_img
from lib.algorithm.convert import FigureConfig, VideoConfig
from lib.algorithm.convert import transform_coordinates_default, transform_landmarks_list_to_array_list
from lib.algorithm.movement import GradientFC, DivergenceFC
from lib.algorithm.movement import StrainFieldFC, GradientVC
from lib.algorithm.movement import DivergenceVC, StrainFieldVC
from lib.algorithm.movement import gradient_2d, strain_field, divergence_2d
from lib.algorithm.process import process_video_gradient_2d, process_img_gradient_2d, process_algo_gradient_2d_v
from lib.algorithm.process import process_video_strain_field, process_img_strain_field, process_algo_strain_field_v
from lib.algorithm.process import process_video_divergence_2d, process_img_divergence_2d, process_algo_divergence_2d_v
from lib.algorithm.process import process_frame, process_img, generate_video


@dataclass
class ImgData:
    img_path: str = "./asserts/time1.jpg"
    img1_path: str = "./asserts/time1.jpg"
    img2_path: str = "./asserts/time2.jpg"
    output_path: str = "./asserts"
    img_array: Optional[np.ndarray] = None
    img1_array: Optional[np.ndarray] = None
    img2_array: Optional[np.ndarray] = None
    fig_points: Optional[plt.Figure] = None
    fig_strain_field_horizontal: Optional[plt.Figure] = None
    fig_strain_field_vertical: Optional[plt.Figure] = None
    fig_gradient: Optional[plt.Figure] = None
    fig_divergence: Optional[plt.Figure] = None


class VidData:
    vid_path: str = "./asserts/video2.mp4"
    output_path: str = "./asserts/output/"
    condition: bool = False
    array_list: list = None
    results_lists: list = None
    results_landmarks_list: list = None
    frame_size: tuple = None


@dataclass
class ImgModel:
    faces_num: int = 5
    detail: bool = True
    detection_value: float = 0.5
    tracking_value: float = 0.5


class ECare:
    def __init__(self,
                 vid_model=model_video(),
                 img_model: ImgModel = ImgModel(),
                 figure_config: FigureConfig = None,
                 video_config: VideoConfig = None,
                 img_data: ImgData = ImgData(),
                 video_data: VidData = VidData(),
                 ):
        self.vid_model = vid_model
        self.img_model = img_model
        self.figure_config = figure_config
        self.video_config = video_config
        self.img_data = img_data
        self.vid_data = video_data

    def create_model_img(self):
        return model_img(faces_num=self.img_model.faces_num,
                         detail=self.img_model.detail,
                         detection_value=self.img_model.detection_value,
                         tracking_value=self.img_model.tracking_value)

    def process_image(self, style, face_index=0):
        if style not in ['points', 'strain', 'gradient', 'divergence']:
            raise ValueError("Invalid style value: "
                             "'style' must be one of 'points', 'strain', 'gradient', or 'divergence'.")
        if style == "points":
            model = self.create_model_img()
            if self.img_data.img_path is None:
                logging.error("Missing image path.")
                raise ValueError("Configure the image path first.")
            elif self.img_data.img_path is not None:
                self.img_data.img_array = cv2.imread(self.img_data.img_path)
                results, self.img_data.fig_points = process_img(model=model, img=self.img_data.img_array)
                return look_img(self.img_data.fig_points)

        elif self.img_data.img1_path is None or self.img_data.img2_path is None:
            logging.error("Missing image path.")
            raise ValueError("Configure the image path first.")
        else:
            if self.figure_config is None:
                warnings.warn("Your config is unsetting!")
            if self.img_data.img1_array is None and self.img_data.img2_array is None:
                self.img_data.img1_array = cv2.imread(self.img_data.img1_path)
                self.img_data.img2_array = cv2.imread(self.img_data.img2_path)

            if style == "strain":
                model = self.create_model_img()
                if not isinstance(self.figure_config, StrainFieldFC) or self.figure_config is None:
                    warnings.warn("Your config is not StrainField Class")
                    config = StrainFieldFC()
                else:
                    config = self.figure_config
                self.img_data.fig_strain_field_horizontal, self.img_data.fig_strain_field_vertical = (
                    process_img_strain_field(model=model,
                                             image1=self.img_data.img1_array,
                                             image2=self.img_data.img2_array,
                                             face_index=face_index,
                                             config=config))
                return self.img_data.fig_strain_field_horizontal, self.img_data.fig_strain_field_vertical

            elif style == "gradient":
                model = self.create_model_img()
                if not isinstance(self.figure_config, GradientFC) or self.figure_config is None:
                    warnings.warn("Your config is not Gradient Class")
                    config = GradientFC()
                else:
                    config = self.figure_config
                self.img_data.fig_gradient = (
                    process_img_gradient_2d(model=model,
                                            image1=self.img_data.img1_array,
                                            image2=self.img_data.img2_array,
                                            face_index=face_index,
                                            config=config))
                return self.img_data.fig_gradient

            elif style == "divergence":
                model = self.create_model_img()
                if not isinstance(self.figure_config, DivergenceFC) or self.figure_config is None:
                    warnings.warn("Your config is not Divergence Class")
                    config = DivergenceFC()
                else:
                    config = self.figure_config
                self.img_data.fig_divergence = (
                    process_img_divergence_2d(model=model,
                                              image1=self.img_data.img1_array,
                                              image2=self.img_data.img2_array,
                                              face_index=face_index,
                                              config=config))
                return self.img_data.fig_divergence

    def process_video(self, style, face_index=0):
        if style not in ['points', 'strain', 'gradient', 'divergence']:
            raise ValueError("Invalid style value: "
                             "'style' must be one of 'points', 'strain', 'gradient', or 'divergence'.")

        elif self.vid_data.vid_path is None:
            logging.error("Missing video path.")
            raise ValueError("Configure the video path first.")

        else:
            if self.figure_config is None or self.video_config is None:
                warnings.warn("Your config is unsetting!")
            if self.vid_data.condition is False:
                self.vid_data.results_lists, self.vid_data.results_landmarks_list, self.vid_data.frame_size = (
                    generate_video(model=self.vid_model,
                                   face_index=face_index,
                                   input_path=self.vid_data.vid_path,
                                   show_detail=True))
                self.vid_data.array_list = transform_landmarks_list_to_array_list(self.vid_data.results_landmarks_list,
                                                                                  self.vid_data.frame_size)
                self.vid_data.condition = True
                print("Work Done: process_face_points.")

            if style == "strain":

                if not isinstance(self.figure_config, StrainFieldFC) or self.figure_config is None:
                    warnings.warn("Your figure config is not StrainField Class")
                    fig_config = StrainFieldFC()
                else:
                    fig_config = self.figure_config

                if not isinstance(self.video_config, StrainFieldVC) or self.video_config is None:
                    warnings.warn("Your config is not StrainField Class")
                    vid_config = StrainFieldVC()
                else:
                    vid_config = self.video_config
                process_algo_strain_field_v(array_list=self.vid_data.array_list,
                                            output_dir=self.vid_data.output_path,
                                            figure_config=fig_config,
                                            video_config=vid_config)
                print("Work Done: process_strain_field_video")

            elif style == "gradient":
                if not isinstance(self.figure_config, GradientFC) or self.figure_config is None:
                    warnings.warn("Your figure config is not Gradient Class")
                    fig_config = GradientFC()
                else:
                    fig_config = self.figure_config
                if not isinstance(self.video_config, StrainFieldVC) or self.video_config is None:
                    warnings.warn("Your config is not Gradient Class")
                    vid_config = GradientVC()
                else:
                    vid_config = self.video_config
                process_algo_gradient_2d_v(array_list=self.vid_data.array_list,
                                           output_dir=self.vid_data.output_path,
                                           figure_config=fig_config,
                                           video_config=vid_config)
                print("Work Done: process_gradient_video")

            elif style == "divergence":
                if not isinstance(self.figure_config, DivergenceFC) or self.figure_config is None:
                    warnings.warn("Your figure config is not Divergence Class")
                    fig_config = DivergenceFC()
                else:
                    fig_config = self.figure_config
                if not isinstance(self.video_config, DivergenceVC) or self.video_config is None:
                    warnings.warn("Your config is not Divergence Class")
                    vid_config = DivergenceVC()
                else:
                    vid_config = self.video_config
                process_algo_divergence_2d_v(array_list=self.vid_data.array_list,
                                             video_dir=self.vid_data.output_path,
                                             figure_config=fig_config,
                                             video_config=vid_config)
                print("Work Done: process_divergence_video.")


temp_configs = ECare()


class ConfigUI(tk.Frame):
    """动态生成配置UI，允许实时调整配置参数"""

    def __init__(self, master, config_instance, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.config_instance = config_instance
        self.widgets = {}  # 用于存储每个控件的引用
        self.build_ui()

    def build_ui(self):
        if not hasattr(self.config_instance, '__dict__'):
            tk.Label(self, text="无效的配置实例").pack()
            return

        for i, (attr, value) in enumerate(vars(self.config_instance).items()):
            tk.Label(self, text=f"{attr}:").grid(row=i, column=0, sticky='w')

            if value is None:
                var = tk.DoubleVar(value=50.0)
                scale = tk.Scale(self, from_=-100, to=100 if isinstance(value, int) else 2.0,
                                 resolution=0.1 if isinstance(value, float) else 1,
                                 orient=tk.HORIZONTAL, variable=var,
                                 command=lambda v, a=attr: setattr(self.config_instance, a, float(v)))
                scale.grid(row=i, column=1)
                self.widgets[attr] = var
                continue

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                checkbutton = tk.Checkbutton(self, text=f"{attr}", variable=var,
                                             command=lambda v=var, a=attr: setattr(self.config_instance, a, v.get()))
                checkbutton.grid(row=i, column=1)
                self.widgets[attr] = var

            elif isinstance(value, (int, float)):
                var = tk.DoubleVar(value=value)
                scale = tk.Scale(self, from_=0, to=100 if isinstance(value, int) else 2.0,
                                 resolution=0.1 if isinstance(value, float) else 1,
                                 orient=tk.HORIZONTAL, variable=var,
                                 command=lambda v, a=attr: setattr(self.config_instance, a, float(v)))
                scale.grid(row=i, column=1)
                self.widgets[attr] = var

            elif isinstance(value, (str, list)):
                var = tk.StringVar(value=value if value else "None")  # 设置默认值为 "jet" 如果是 None
                entry = tk.Entry(self, textvariable=var)
                entry.grid(row=i, column=1)
                var.trace('w', lambda *_, v=var, a=attr: setattr(self.config_instance, a,
                                                                 v.get() if v.get() != "None" else None))
                self.widgets[attr] = var

            elif isinstance(value, tuple):
                var = tk.StringVar(value=str(value))
                entry = tk.Entry(self, textvariable=var)
                entry.grid(row=i, column=1)
                var.trace('w', lambda *_, v=var, a=attr: setattr(self.config_instance, a,
                                                                 eval(v.get()) if v.get() != "None" else None))
                self.widgets[attr] = var

    def get_config(self):
        return self.config_instance


def main():
    """
    主函数，用于创建GUI界面并处理用户交互。
    """
    # 创建主窗口并设置窗口属性
    root = tk.Tk()
    root.title("E-Care")
    root.geometry("450x900")

    # 创建ECare实例用于后续处理
    ecare = ECare()
    # 初始化配置界面变量
    config_ui_frame = None
    # 创建处理模式变量并设置默认值为"image_points"
    mode_var = tk.StringVar(value="image_points")

    def select_file(file_type="image_points"):
        """
        根据处理模式选择文件。

        参数:
        file_type (str): 文件类型，可以是"single"（单个图像）、"dual"（双图像）或"video"（视频）。
        """
        if file_type == "single":
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
            ecare.img_data.img_path = file_path
            update_file_label()

        elif file_type == "video":
            file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
            ecare.vid_data.vid_path = file_path
            update_file_label()

    def select_file_dual1():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        ecare.img_data.img1_path = file_path
        update_file_label()

    def select_file_dual2():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        ecare.img_data.img2_path = file_path
        update_file_label()

    def select_output_folder():
        """
        选择输出文件夹。
        """
        folder_path = filedialog.askdirectory()
        if folder_path:
            ecare.img_data.output_path = folder_path
            ecare.vid_data.output_path = folder_path
            update_file_label()

    def update_file_label():
        """
        更新文件选择标签以显示当前选择的文件路径。
        """
        mode = mode_var.get()
        if mode == "image_points":
            file_label.config(text=f"选择的图像文件: \n{ecare.img_data.img_path}\n"
                                   f"选择的输出文件夹: \n{ecare.img_data.output_path}")

        elif mode == "image_strain":
            file_label.config(text=f"选择的前帧图像: \n{ecare.img_data.img1_path}\n"
                                   f"选择的后帧图像: \n{ecare.img_data.img2_path}\n"
                                   f"选择的输出文件夹: \n{ecare.img_data.output_path}")

        elif mode == "image_divergence":
            file_label.config(text=f"选择的前帧图像: \n{ecare.img_data.img1_path}\n"
                                   f"选择的后帧图像: \n{ecare.img_data.img2_path}\n"
                                   f"选择的输出文件夹: \n{ecare.img_data.output_path}")

        elif mode == "image_gradient":
            file_label.config(text=f"选择的前帧图像: \n{ecare.img_data.img1_path}\n"
                                   f"选择的后帧图像: \n{ecare.img_data.img2_path}\n"
                                   f"选择的输出文件夹: \n{ecare.img_data.output_path}")

        elif mode == "video_points":
            file_label.config(text=f"选择的视频文件: \n{ecare.vid_data.vid_path}\n"
                                   f"选择的输出文件夹: \n{ecare.vid_data.output_path}")

        elif mode == "video_strain":
            file_label.config(text=f"选择的视频文件: \n{ecare.vid_data.vid_path}\n"
                                   f"选择的输出文件夹: \n{ecare.vid_data.output_path}")

        elif mode == "video_divergence":
            file_label.config(text=f"选择的视频文件: \n{ecare.vid_data.vid_path}\n"
                                   f"选择的输出文件夹: \n{ecare.vid_data.output_path}")

        elif mode == "video_gradient":
            file_label.config(text=f"选择的视频文件: \n{ecare.vid_data.vid_path}\n"
                                   f"选择的输出文件夹: \n{ecare.vid_data.output_path}")

    def update_config_ui():
        """
        更新配置界面以反映当前处理模式。
        """
        nonlocal config_ui_frame
        mode = mode_var.get()

        # 根据处理模式创建相应的配置实例
        if mode == "image_points":
            ecare.figure_config = FigureConfig()
            ecare.video_config = VideoConfig()
            print("image_points_config_setting")

        elif mode == "image_strain":
            ecare.figure_config = StrainFieldFC()
            ecare.video_config = VideoConfig()
            print("image_strain_config_setting")

        elif mode == "image_divergence":
            ecare.figure_config = DivergenceFC()
            ecare.video_config = VideoConfig()
            print("image_divergence_config_setting")

        elif mode == "image_gradient":
            ecare.figure_config = GradientFC()
            ecare.video_config = VideoConfig()
            print("image_gradient_config_setting")

        elif mode == "video_points":
            ecare.figure_config = FigureConfig()
            ecare.video_config = VideoConfig()
            print("video_points_config_setting")

        elif mode == "video_strain":
            ecare.figure_config = StrainFieldFC()
            ecare.video_config = StrainFieldVC()
            print("video_strain_config_setting")

        elif mode == "video_divergence":
            ecare.figure_config = DivergenceFC()
            ecare.video_config = DivergenceVC()
            print("video_divergence_config_setting")

        elif mode == "video_gradient":
            ecare.figure_config = GradientFC()
            ecare.video_config = GradientVC()
            print("video_gradient_config_setting")

        # 销毁旧的配置界面并创建新的配置界面
        if config_ui_frame:
            config_ui_frame.destroy()
        config_ui_frame = tk.Frame(root)
        config_ui_frame.pack(pady=10)

        # 添加配置UI到新框架中
        if mode.startswith("image_"):
            if hasattr(ecare, 'figure_config'):
                new_config = ConfigUI(config_ui_frame, ecare.figure_config)
                new_config.pack()
                return new_config
        elif mode.startswith("video_"):
            if hasattr(ecare, 'figure_config'):
                ConfigUI(config_ui_frame, ecare.figure_config).pack()
            if hasattr(ecare, 'video_config'):
                ConfigUI(config_ui_frame, ecare.video_config).pack()

    def create_file_selection_widgets():
        """
        根据当前处理模式创建文件选择控件。
        """
        for widget in file_selection_frame.winfo_children():
            widget.destroy()

        mode = mode_var.get()
        if mode == "image_points":
            tk.Button(file_selection_frame, text="选择图像文件", command=lambda: select_file("single")).pack(pady=5)
            tk.Button(file_selection_frame, text="选择输出文件夹", command=select_output_folder).pack(pady=5)

        elif mode == "image_strain" or mode == "image_divergence" or mode == "image_gradient":
            tk.Button(file_selection_frame, text="选择前帧图像", command=select_file_dual1).pack(pady=5)
            tk.Button(file_selection_frame, text="选择后帧图像", command=select_file_dual2).pack(pady=5)
            tk.Button(file_selection_frame, text="选择输出文件夹", command=select_output_folder).pack(pady=5)

        elif mode == "video_points" or mode == "video_strain" or mode == "video_divergence" or mode == "video_gradient":
            tk.Button(file_selection_frame, text="选择视频文件", command=lambda: select_file("video")).pack(pady=5)
            tk.Button(file_selection_frame, text="选择输出文件夹", command=select_output_folder).pack(pady=5)

    def start_processing():
        """
        开始处理根据当前处理模式。
        """
        mode = mode_var.get()

        # 根据处理模式调用相应的处理方法
        if mode == "image_points":
            ecare.process_image(style="points")

        elif mode == "image_strain":
            print("Start Process: image_strain")
            fig1, fig2 = ecare.process_image(style="strain")
            fig1.show()
            if ecare.img_data.output_path is not None:
                fig1.savefig(os.path.join(ecare.img_data.output_path, "fig1.png"))
                open_image(os.path.join(ecare.img_data.output_path, "fig1.png"))
                print("Have Saved!")
            fig2.show()
            if ecare.img_data.output_path is not None:
                fig2.savefig(os.path.join(ecare.img_data.output_path, "fig2.png"))
                open_image(os.path.join(ecare.img_data.output_path, "fig2.png"))
                print("Have Saved!")
            print("Process Succeed: image_strain")

        elif mode == "image_divergence":
            print("Start Process: image_divergence")
            fig = ecare.process_image(style="divergence")
            fig.show()
            if ecare.img_data.output_path is not None:
                fig.savefig(os.path.join(ecare.img_data.output_path, "fig.png"))
                open_image(os.path.join(ecare.img_data.output_path, "fig.png"))
                print("Have Saved!")
            print("Process Succeed: image_divergence")

        elif mode == "image_gradient":
            print("Start Process: image_gradient")
            fig = ecare.process_image(style="gradient")
            fig.show()
            if ecare.img_data.output_path is not None:
                fig.savefig(os.path.join(ecare.img_data.output_path, "fig.png"))
                open_image(os.path.join(ecare.img_data.output_path, "fig.png"))
                print("Have Saved!")
            print("Process Succeed: image_gradient")

        elif mode == "video_points":
            print("Start Process: video_points")
            ecare.process_video(style="points")
            print("Process Succeed: video_points")

        elif mode == "video_strain":
            print("Start Process: video_strain")
            ecare.process_video(style="strain")
            print("Process Succeed: video_strain")

        elif mode == "video_divergence":
            print("Start Process: video_divergence")
            ecare.process_video(style="divergence")
            print("Process Succeed: video_divergence")

        elif mode == "video_gradient":
            print("Start Process: video_gradient")
            ecare.process_video(style="gradient")
            print("Process Succeed: video_gradient")
        else:
            messagebox.showerror("错误", "未选择处理模式。")

    def open_image(image_path):
        """
        使用系统默认图片查看器打开图片。

        参数:
        image_path (str): 图片文件路径。
        """
        if os.name == 'nt':  # Windows
            os.startfile(image_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open', image_path])

    # 创建并打包选择处理模式的标签和下拉菜单
    tk.Label(root, text="选择处理模式:").pack(pady=5)
    mode_menu = tk.OptionMenu(root, mode_var,
                              "image_points",
                              "image_strain", "image_divergence", "image_gradient",
                              "video_points", "video_strain", "video_divergence", "video_gradient",
                              command=lambda _: (update_config_ui(), create_file_selection_widgets()))

    mode_menu.pack(pady=5)

    # 创建文件选择控件的容器
    file_selection_frame = tk.Frame(root)
    file_selection_frame.pack(pady=5)

    # 创建文件选择标签
    file_label = tk.Label(root, text="请选择文件")
    file_label.pack(pady=5)

    # 初始化文件选择控件
    create_file_selection_widgets()

    ecare.figure_config = temp_configs.figure_config
    ecare.video_config = temp_configs.video_config

    # 创建并打包开始处理按钮
    tk.Button(root, text="开始处理", command=start_processing, width=20).pack(pady=10)

    # 进入主循环
    root.mainloop()


if __name__ == "__main__":
    main()
