from U_net_tools import *
import tkinter as tk
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
def load_model(device = 'cpu',model = './model/pretrained_model.pth'):
    net = UNet(1, 1)
    net = net.to(device)
    if str(device) == "cpu":
        net.load_state_dict(torch.load(model, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(model))
    net.eval()
    return net

# 进行神经网络预测
def predict(image, model,device = 'cpu',threshold = 0.5):
    image = Image.fromarray(image.astype(np.uint8))
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        pred = model(image_tensor)
        if device != 'cpu' :
            p = pred[0].cpu().detach().numpy()
        else:
            p = pred[0].numpy()
        p[p >= threshold] = 1
        p[p < threshold] = 0
    return p


# 创建Tkinter窗口
def create_window():
    window = tk.Tk()
    window.title("海马体区域预测")

    def predict_image(device = 'cpu'):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            prediction = predict(image, model,device)
            display_image(image, prediction)
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            if save_path:
                # 保存预测图像
                prediction_img = Image.fromarray((prediction.squeeze()*255).astype(np.uint8))  # 转换为 PIL 图像
                prediction_img.save(save_path, 'JPEG')

                messagebox.showinfo("保存成功", f"预测结果已保存：{save_path}")


    def predict_nii_file(device = 'cpu'):
        file_path = filedialog.askopenfilename(title="选择NIfTI文件", filetypes=[("NIfTI Files", "*.nii;*.nii.gz")])
        if file_path:
            nii_img = nib.load(file_path)
            images = nii_img.get_fdata()  # 获取图像数据
            depth,_,_ = images.shape  # 获取图像的尺寸

            # 创建一个空的数组来保存所有预测结果
            prediction_stack = np.zeros((depth,320, 240))

            # 对每个切片进行预测，并将结果保存到 3D 数组中
            for x in range(depth):
                slice_data = images[x, :, :]  # 获取某个切片的原始数据
                prediction = predict(slice_data.copy(), model,device)  # 对当前切片进行预测

                # 将预测结果保存到 prediction_stack 对应的切片
                prediction_stack[x, :,:] = prediction*255.0
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

        # 将海马体区域设为黄色
        overlay[prediction.squeeze(0) == 1] = [255, 255, 0]  # 黄色 (R=255, G=255, B=0)

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

    def display_3D(images):
        # 将 3D 图像数据中的非零部分转换为体素
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 创建一个布尔掩码，表示体素值大于 0 的部分
        non_zero_voxels = images > 0

        # 获取非零体素的坐标
        x, y, z = np.where(non_zero_voxels)

        # 绘制这些体素
        ax.scatter(z, y, x, c='r', marker='o', s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 创建按钮
    predict_image_button = tk.Button(window, text="预测单张图片", command=lambda: predict_image(device))
    predict_image_button.pack(pady=10)

    predict_nii_button = tk.Button(window, text="预测.nii文件", command=lambda: predict_nii_file(device))
    predict_nii_button.pack(pady=10)

    # 加载模型
    global model
    model = load_model(device,model='./pretrained_model.pth')

    window.mainloop()


# 启动窗口
create_window()