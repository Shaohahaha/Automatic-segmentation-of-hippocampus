from U_net_tools import *
import tkinter as tk
from tkinter import ttk  # 需要导入ttk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import nibabel as nib
import torch
from torchvision import transforms
import cv2
import trimesh

matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端
# 加载训练好的模型
def load_model(device = 'cpu',net_type= 'Unet',model = './model/Unet.pth'):
    try:
        if net_type == 'Unet':
            net = UNet(1, 1)
        elif net_type == 'ResUnet':
            net = ResUNet(1, 1)
        else:
            raise ValueError("Unsupported model type")

        if not os.path.exists(model):
            raise FileNotFoundError(f"Model file {model} not found")

        net = net.to(device)
        net.load_state_dict(torch.load(model, map_location=device))
        net.eval()
        return net
    except Exception as e:
        messagebox.showerror("加载错误", f"加载模型失败: {str(e)}")
        return None

# 进行神经网络预测
def predict(image, model,device = 'cpu',threshold = 0.5):
    try:
        # 图像预处理
        if image is None:
            raise ValueError("无法读取图像")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = cv2.resize(image, (240, 320), interpolation=cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor)
            p = pred[0].cpu().numpy().squeeze()
            p = (p >= threshold).astype(np.uint8)
        return p
    except Exception as e:
        messagebox.showerror("预测错误", f"预测失败: {str(e)}")
        return None

# 创建Tkinter窗口
def create_window():
    window = tk.Tk()
    window.title("海马体区域预测")
    window.geometry("400x300")

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None  # 初始化为空

    def load_model_file():  # 类型选择窗口
        type_window = tk.Toplevel()
        type_window.title("选择模型类型")
        net_type = tk.StringVar(value="Unet")
        ttk.Radiobutton(type_window, text="Unet", variable=net_type, value="Unet").pack(padx=20, pady=5)
        ttk.Radiobutton(type_window, text="ResUnet", variable=net_type, value="ResUnet").pack(padx=20, pady=5)
        def on_confirm():
            type_window.destroy()
            model_path = filedialog.askopenfilename(
                title="选择模型文件",
                filetypes=[("PTH Files", "*.pth"), ("All Files", "*.*")]
            )
            if model_path:
                nonlocal model
                model = load_model(device, net_type.get(), model_path)
                if model:
                    messagebox.showinfo("成功", f"已加载{net_type.get()}模型")
        tk.Button(type_window, text="确认", command=on_confirm).pack(pady=10)


    def predict_image(device = 'cpu'):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            prediction = predict(image, model,device)
            saveimg = display_image(image, prediction)
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            if save_path:
                # 保存预测图像
                saveimg = Image.fromarray((saveimg.squeeze()).astype(np.uint8))  # 转换为 PIL 图像
                saveimg.save(save_path, 'PNG')

                messagebox.showinfo("保存成功", f"预测结果已保存：{save_path}")


    def predict_nii_file(device = 'cpu'):
        file_path = filedialog.askopenfilename(title="选择NIfTI文件", filetypes=[("NIfTI Files", "*.nii;*.nii.gz")])
        if file_path:
            nii_img = nib.load(file_path)
            images = nii_img.get_fdata()  # 获取图像数据，形状为 (height, width, depth)

            # 确保切片方向为轴向（从头顶到脖子）
            # NIfTI 文件的第三个维度通常是轴向（depth）
            depth = images.shape[2]  # 获取轴向切片的数量

            # 创建一个空的数组来保存所有预测结果
            prediction_stack = np.zeros((240, 320,depth))
            image_stack = np.zeros((240, 320, depth))

            # 对每个轴向切片进行预测
            for z in range(depth):
                slice_data = images[:, :, z]  # 获取某个轴向切片的原始数据
                slice_data_normalized = np.interp(slice_data, (slice_data.min(), slice_data.max()), (0, 255)).astype(
                    np.uint8)
                slice_img = Image.fromarray(slice_data_normalized)
                rotated_image = slice_img.rotate(-90, expand=True)
                prediction = predict(np.array(rotated_image.copy()), model, device)  # 对当前切片进行预测
                prediction = np.rot90(prediction,k=1)

                # 将预测结果保存到 prediction_stack 对应的切片
                prediction_stack[:, :, z] = prediction*255.0
            display_3D(prediction_stack)

            # 在预测完成后，展示保存格式选择按钮
            show_save_buttons(prediction_stack, nii_img)

    def show_save_buttons(prediction_stack, nii_img):
        # 创建一个新窗口，显示保存格式的按钮
        save_window = tk.Toplevel(window)
        save_window.title("选择保存格式")

        # PLY按钮
        def save_as_ply():
            save_ply(prediction_stack)
            save_window.destroy()  # 关闭保存格式选择窗口

        ply_button = tk.Button(save_window, text="保存为 PLY", command=save_as_ply)
        ply_button.pack(pady=10)

        # NII按钮
        def save_as_nii():
            save_nii(prediction_stack, nii_img)
            save_window.destroy()  # 关闭保存格式选择窗口

        nii_button = tk.Button(save_window, text="保存为 NII", command=save_as_nii)
        nii_button.pack(pady=10)

        # 图片集按钮
        def save_as_images():
            save_images_as_collection(prediction_stack, nii_img)
            save_window.destroy()  # 关闭保存格式选择窗口

        images_button = tk.Button(save_window, text="保存为图片集", command=save_as_images)
        images_button.pack(pady=10)

    def save_ply(prediction_stack):
        save_path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY Files", "*.ply")])
        if save_path:
            # 获取非零体素的坐标
            non_zero_voxels = np.argwhere(prediction_stack > 0)
            # 创建点云对象
            point_cloud = trimesh.points.PointCloud(non_zero_voxels)
            # 保存为 ply 文件
            point_cloud.export(save_path)
            messagebox.showinfo("保存成功", f"预测结果已保存为 ply 文件在：{save_path}")

    def save_nii(prediction_stack, nii_img):
        save_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIfTI Files", "*.nii;*.nii.gz")])
        if save_path:
            output_nii = nib.Nifti1Image(prediction_stack, nii_img.affine)
            nib.save(output_nii, save_path)
            messagebox.showinfo("保存成功", f"预测结果已保存为 nii 文件在：{save_path}")

    def save_images_as_collection(prediction_stack, nii_img):
        # 获取3D文件的名称，并创建保存文件夹
        file_name = os.path.basename(nii_img.get_filename()).split('.')[0] + '_predict'
        save_dir = filedialog.askdirectory(title="选择保存文件夹")
        if save_dir:
            save_dir = os.path.join(save_dir, file_name)
            os.makedirs(save_dir, exist_ok=True)
            images = nii_img.get_fdata()
            # 保存每个切片的图像
            for i in range(prediction_stack.shape[2]):
                slice_data = images[:, :, i]  # 获取某个轴向切片的原始数据
                slice_data_normalized = np.interp(slice_data, (slice_data.min(), slice_data.max()), (0, 255)).astype(
                    np.uint8)
                slice_img = Image.fromarray(slice_data_normalized)
                image = slice_img.rotate(-90, expand=True)
                image = image.resize((240, 320), Image.Resampling.LANCZOS)
                image = np.uint8(image)
                prediction = prediction_stack[:, :, i]
                prediction = np.rot90(prediction,k=3)
                prediction = cv2.resize(prediction, (240, 320), interpolation=cv2.INTER_AREA)
                prediction = np.uint8(prediction)
                if len(image.shape) == 2:  # 对于灰度图像，复制一个 3 通道的图像
                    image_rgb = np.stack([image, image, image], axis=-1)
                else:
                # 对于 RGB 图像，直接使用原图
                    image_rgb = image
                # 合并原图与预测结果
                overlay = np.zeros_like(image_rgb)
                overlay[prediction == 255] = [255, 255, 0]  # 黄色 (R=255, G=255, B=0)

                # 叠加原图与预测
                output_image = (image_rgb * 0.5 + overlay * 0.5).astype(np.uint8)
                saveimg = Image.fromarray(
                    (output_image.squeeze()).astype(np.uint8))  # 转换为 PIL 图像
                # 保存叠加图像
                save_path = os.path.join(save_dir, f"slice_{i + 1}.png")
                saveimg.save(save_path)

            messagebox.showinfo("保存成功", f"预测结果已保存为图像集：{save_dir}")

    def display_3D(images):
        # 将 3D 图像数据中的非零部分转换为体素
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 创建一个布尔掩码，表示体素值大于 0 的部分
        non_zero_voxels = images > 0

        # 获取非零体素的坐标
        x, y, z = np.where(non_zero_voxels)

        # 绘制这些体素
        ax.scatter(x, y, z, c='r', marker='o', s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def display_image(image, prediction):
        """
        在原图上叠加海马体区域，用红色标注。

        :param image: 原始图像 (PIL Image 或 numpy 数组)
        :param prediction: 海马体区域的预测结果（二值图像）
        """
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((240, 320), Image.Resampling.LANCZOS)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 将原图转换为 numpy 数组，方便后续处理
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 确保预测结果是二值化的（海马体区域为1，其它为0）
        prediction = prediction.astype(np.uint8)
        # 如果原图是灰度图像，确保其形状是二维的
        if len(image.shape) == 2:
            # 对于灰度图像，复制一个 3 通道的图像
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            # 对于 RGB 图像，直接使用原图
            image_rgb = image

        # 创建一个 RGB 图像作为红色叠加层
        overlay = np.zeros_like(image_rgb)

        # # 调整 prediction 的尺寸以匹配 overlay
        # prediction_resized = cv2.resize(prediction.squeeze(0), (image_rgb.shape[1], image_rgb.shape[0]),
        #                                 interpolation=cv2.INTER_NEAREST)
        # 将海马体区域设为黄色
        overlay[prediction == 1] = [255, 255, 0]  # 黄色 (R=255, G=255, B=0)

        # 创建一个透明背景的图像，将原图和黄色叠加层融合
        output_image = np.copy(image)
        alpha = 0.5  # 设置叠加的透明度
        output_image = (1 - alpha) * output_image + alpha * overlay

        # 显示原图与叠加后的图像
        plt.figure(figsize=(10, 5))

        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("原始图像")
        plt.axis('off')

        # 显示叠加后的图像
        plt.subplot(1, 2, 2)
        plt.imshow(output_image.astype(np.uint8))
        plt.title("叠加海马体区域")
        plt.axis('off')

        plt.show()

        return output_image


    # 界面布局
    control_frame = ttk.Frame(window)
    control_frame.pack(pady=20)

    ttk.Button(control_frame, text="加载模型", command=load_model_file).grid(row=0, column=0, padx=10)
    ttk.Button(control_frame, text="预测图片", command=lambda: predict_image(device)).grid(row=0, column=1, padx=10)
    ttk.Button(control_frame, text="预测NIfTI", command=lambda: predict_nii_file(device)).grid(row=0, column=2, padx=10)

    status_bar = ttk.Label(window, text=f"当前设备: {device}", relief=tk.SUNKEN)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    window.mainloop()


# 启动窗口
create_window()