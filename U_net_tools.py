import os
import numpy as np
import torch
import torch.nn as nn
import hausdorff
import torchvision.models as models
import torch.nn.functional as F

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

class DownSamplingResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingResNet, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化，降低图像尺寸
            ResBlock(in_channels, out_channels)  # 使用残差块
        )

    def forward(self, x):
        return self.layer(x)

class UpSamplingResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingResNet, self).__init__()
        # 上采样层：用反卷积（ConvTranspose2d）将图像尺寸放大
        self.layer = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 使用残差块
        self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x, r):
        # 上采样过程
        x = self.layer(x)
        # 跟对应的编码器层特征图连接（skip connection）
        x = torch.cat((x, r), dim=1)  # 拼接通道维度
        # 使用残差块
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道数不一致，使用1x1卷积进行匹配
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # 残差连接
        out = self.relu(self.bn1(self.conv1(x)))  # 第一个卷积层
        out = self.bn2(self.conv2(out))  # 第二个卷积层
        out += residual  # 加上残差连接
        out = self.relu(out)  # 激活函数
        return out

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()
        # 输入层，使用残差块
        self.conv = ResBlock(in_channels, 64)

        # 编码器（DownSampling）部分，使用残差块
        self.D1 = DownSamplingResNet(64, 128)  # 下采样层
        self.D2 = DownSamplingResNet(128, 256)
        self.D3 = DownSamplingResNet(256, 512)
        self.D4 = DownSamplingResNet(512, 1024)

        # 解码器（UpSampling）部分
        self.U1 = UpSamplingResNet(1024, 512)
        self.U2 = UpSamplingResNet(512, 256)
        self.U3 = UpSamplingResNet(256, 128)
        self.U4 = UpSamplingResNet(128, 64)

        # 输出层，使用 Sigmoid 激活函数进行二分类
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码器部分
        L1 = self.conv(x)
        L2 = self.D1(L1)
        L3 = self.D2(L2)
        L4 = self.D3(L3)
        Bottom = self.D4(L4)

        # 解码器部分
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



def hd95_compute(predict: np.ndarray, label: np.ndarray, method="euclidean"):
    total_hd95 = 0

    for i in range(predict.shape[0]):  # 遍历每张图像
        predict_img = predict[i]
        label_img = label[i]

        predict_img, label_img = transform_image_data(predict_img, label_img)

        hddistance = hausdorff.hausdorff_distance(predict_img.squeeze(), label_img.squeeze(), distance=method)

        hd95 = hddistance*0.95

        total_hd95 += hd95

    return total_hd95 / predict.shape[0]


def jaccard_compute(predict: np.ndarray, label: np.ndarray):
    predict, label = transform_image_data(predict, label)
    intersection = np.intersect1d(predict,label)
    union = np.union1d(predict, label)
    jaccard_similarity = intersection.size / union.size
    return jaccard_similarity

# 测试代码
if __name__ == "__main__":
    model = ResUNet(in_channels=1, out_channels=1)  # 输入通道为1（灰度图像），输出通道为1（二分类任务）

    # 输入尺寸为(4, 1, 320, 240)
    input_tensor = torch.randn(4, 1, 320, 240)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # 输出的形状