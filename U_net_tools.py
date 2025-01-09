import os
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import trimesh
import nibabel as nib
import matplotlib.pyplot as plt

class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.unit = nn.Sequential(
            # 保持图像大小
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 保持图像大小
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.unit(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2), # 最大池化，将图像大小变为原来的1/2
            ConvUnit(in_channels, out_channels) # 卷积单元
        )

    def forward(self, x):
        return self.layer(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        # 上采样层，将通道数变为1/2是为了在concat后保持通道数不变
        self.layer = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 卷积单元
        self.conv = ConvUnit(in_channels, out_channels)

    def forward(self, x, r):
        # 对x进行上采样，同时通道数减半
        x = self.layer(x)
        # 将x与r在通道维度连接，恢复原本通道数
        x = torch.cat((x, r), dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # 输入层
        self.conv = ConvUnit(in_channels, 64)
        # 定义四个下采样层和四个上采样层
        self.D1 = DownSampling(64, 128)
        self.D2 = DownSampling(128, 256)
        self.D3 = DownSampling(256, 512)
        self.D4 = DownSampling(512, 1024)
        self.U1 = UpSampling(1024, 512)
        self.U2 = UpSampling(512, 256)
        self.U3 = UpSampling(256, 128)
        self.U4 = UpSampling(128, 64)
        # 输出层，输出图像像素保持在0～1以进行二分类
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入和UNet的左半部分
        L1 = self.conv(x)
        L2 = self.D1(L1)
        L3 = self.D2(L2)
        L4 = self.D3(L3)
        Bottom = self.D4(L4)
        # UNet的右半部分和输出
        R1 = self.U1(Bottom, L4)
        R2 = self.U2(R1, L3)
        R3 = self.U3(R2, L2)
        R4 = self.U4(R3, L1)
        return self.out(R4)


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def transform_image_data(predict: np.ndarray, label: np.ndarray):
    predict = predict.astype(np.bool_).astype(np.int_)
    label = label.astype(np.bool_).astype(np.int_)
    return predict, label

def dice_coef(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (2. * intersection + epsilon) / (predict.sum() + label.sum() + epsilon)

def ppv_compute(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (intersection + epsilon) / (predict.sum() + epsilon)

def binary_to_point_set(binary_img):
    """
    将二值图像转换为点集（坐标列表）
    :param binary_img: 二值图像，非零点将被视为目标点
    :return: 点集 (list of tuples)
    """
    points = np.argwhere(binary_img > 0)  # 返回非零点的坐标
    return np.array(points)



def hd95_compute(predict: np.ndarray, label: np.ndarray, distance="euclidean"):
    """
    计算 HD95 距离
    :param predict: 预测的二值图像数组 (N, H, W)
    :param label: 真值的二值图像数组 (N, H, W)
    :param distance: 距离度量方法
    :return: 平均 HD95 距离 (整数)
    """
    total_hd95 = 0
    invalid_cases = 0

    for i in range(predict.shape[0]):  # 遍历每张图像
        predict_img = predict[i]
        label_img = label[i]

        predict_img, label_img = transform_image_data(predict_img, label_img)

        predict_points = binary_to_point_set(predict_img)
        label_points = binary_to_point_set(label_img)

        if predict_points.size == 0 or label_points.size == 0:
            # 如果任一图像为空集，增加无效计数
            invalid_cases += 1
            continue

        # 计算点到点的距离矩阵
        distances = cdist(predict_points, label_points, metric='euclidean')

        # 计算 HD95 距离
        hd95 = np.percentile(distances, 95)

        total_hd95 += hd95

    if invalid_cases == predict.shape[0]:
        return 0

    return int(total_hd95 / (predict.shape[0] - invalid_cases))


def jaccard_compute(predict: np.ndarray, label: np.ndarray):
    predict, label = transform_image_data(predict, label)
    intersection = np.intersect1d(predict,label)
    union = np.union1d(predict, label)
    jaccard_similarity = intersection.size / union.size
    return jaccard_similarity

def reconstruction(images,idx = 1,save_path = "./save",show = False):
    # 确保图像尺寸为240x320
    height, width = 240, 320
    assert images[0].shape == (height, width), "图片尺寸不匹配!"

    nii_save_path = save_path + '/nii_save/'
    ply_save_path = save_path + '/ply_save/'
    create_dir_not_exist(nii_save_path)
    create_dir_not_exist(ply_save_path)

    # 将二维图像堆叠成三维数组，假设图像顺序按Z轴堆叠
    hippocampus_3d = np.stack(images, axis=-1)

    # 获取非零体素的坐标
    non_zero_voxels = np.argwhere(hippocampus_3d > 0)  # 只保留预测为海马体区域的体素

    # 创建点云对象
    point_cloud = trimesh.points.PointCloud(non_zero_voxels)

    # 保存为 ply 文件
    point_cloud.export(ply_save_path+'hippocampus_3d_{}.ply'.format(idx))

    # 创建一个NIfTI对象
    nifti_img = nib.Nifti1Image(hippocampus_3d, affine=np.eye(4))  # 使用单位矩阵作为affine

    # 保存为nii文件
    nib.save(nifti_img, nii_save_path+'hippocampus_3d_{}.nii'.format(idx))

    if show:

        # 可视化3D：显示海马体区域的非零体素
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 在3D空间中绘制这些体素
        ax.scatter(non_zero_voxels[:, 0], non_zero_voxels[:, 1], non_zero_voxels[:, 2], c='r', marker='o', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
