import cv2
import argparse
import glob
import os
import random
import hausdorff
import numpy as np
import torch
import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import random_split

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

def hd95_compute(predict: np.ndarray, label: np.ndarray, distance="euclidean"):
    predict, label = transform_image_data(predict, label)
    hd95_values = 0
    for i in range(predict.shape[0]):  # 遍历每个图像
        predict_t = predict[i].squeeze(0)
        label_t = label[i].squeeze(0)

        # 计算 Hausdorff 距离
        distance_value = hausdorff.hausdorff_distance(predict_t, label_t, distance=distance)

        # 计算 95% Hausdorff 距离
        hd95_values += distance_value * 0.95

    return hd95_values

def jaccard_compute(predict: np.ndarray, label: np.ndarray):
    predict, label = transform_image_data(predict, label)
    intersection = np.intersect1d(predict,label)
    union = np.union1d(predict, label)
    jaccard_similarity = intersection.size / union.size
    return jaccard_similarity

class MRIDataset(Dataset):
    def __init__(self, url, W, H, transform=None):
        self.label = glob.glob(os.path.join(url, 'label_combine/*/*/*.jpg'))
        self.transform = transform
        self.W = W
        self.H = H

    def augment(self, img, flip):
        # flip = 1: 水平翻转
        # flip = 0: 垂直翻转
        # flip = -1: 同时进行水平翻转和垂直翻转
        img_flipped = cv2.flip(img, flip)
        return img_flipped

    def __getitem__(self, index):
        # 通过索引获取原图和标签的URL
        label_url = self.label[index]
        img_url = label_url.replace("label_combine","original")
        # 通过cv2库读取图像
        image = cv2.imread(img_url)
        label = cv2.imread(label_url)
        #尺寸统一
        image = cv2.resize(image, (self.W,self.H), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (self.W,self.H), interpolation=cv2.INTER_AREA)
        # 转换为灰度图，3通道转换为1通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # 修正标签，将海马体部分像素设置为1，非海马体部分设置为0
        if label.max() > 1:
            label = label / 255
        label[label >= 0.5] = 1
        label[label < 0.5] = 0
        # 进行随机的数据增强
        flip = random.choice([-1, 0, 1, 2])
        if flip != 2:
            image = self.augment(image, flip)
            label = self.augment(label, flip)
        # 转换数据
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

    def __len__(self):
        return len(self.label)

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

def merge_labels_and_save(base_path, subject, output_path):
    """
    合并左右标签并保存结果

    :param base_path: 数据集根目录
    :param subject: 具体的子文件夹（如'35'）
    :param output_path: 输出保存路径
    """
    original_path = os.path.join(base_path, 'original', subject)
    label_path = os.path.join(base_path, 'label', f'{subject}label')

    # 获取original文件夹中的所有子文件夹
    folders = os.listdir(original_path)

    # 遍历每个子文件夹
    for folder in folders:
        label_folder_path = os.path.join(label_path, folder)
        left_label_folder = os.path.join(label_folder_path, folder.replace('tal_noscale','L').replace('ACPC','L'))
        right_label_folder = os.path.join(label_folder_path, folder.replace('tal_noscale','R').replace('ACPC','R'))

        # 检查是否同时存在L和R文件夹
        if not os.path.exists(left_label_folder) or not os.path.exists(right_label_folder):
            print(f"Missing L or R folder in {folder}. Skipping folder.")
            continue

        # 获取原始图像（假设图像是jpg格式）
        left_labels = sorted([img for img in os.listdir(left_label_folder) if img.endswith('.jpg')])
        right_labels = sorted([img for img in os.listdir(right_label_folder) if img.endswith('.jpg')])

        # 确保每个文件夹中的图像数量一致
        if len(left_labels) != len(right_labels):
            print(f"Image count mismatch in {folder}. Skipping folder.")
            continue

        # 遍历每个图像
        for idx in range(len(left_labels)):
            left_label_path = os.path.join(left_label_folder, left_labels[idx])
            left_label = cv2.imread(left_label_path, cv2.IMREAD_GRAYSCALE)

            right_label_path = os.path.join(right_label_folder, right_labels[idx])
            right_label = cv2.imread(right_label_path, cv2.IMREAD_GRAYSCALE)

            # 合并左右标签(值相加）
            combined_label = cv2.add(left_label, right_label)  # 标签值相加

            # 输出路径
            output_folder = os.path.join(output_path, subject, folder)
            os.makedirs(output_folder, exist_ok=True)

            # 保存文件名保持原始文件名结构
            if(label_folder_path[-4:]=='ACPC'):
                output_image_name = os.path.basename(left_labels[idx].replace('L','ACPC'))
            else:
                output_image_name = os.path.basename(left_labels[idx].replace('L', 'tal_noscale'))
            output_image_path = os.path.join(output_folder, output_image_name)
            cv2.imwrite(output_image_path, combined_label)

            # 可视化或调试（可选）
            # cv2.imshow(f"Combined Image {folder} - {idx}", combined_image)
            # cv2.waitKey(0)

    # cv2.destroyAllWindows()

def load_data(url="./dataset", W=240, H=320, batch_size=64, shuffle=True, split=None):
    # 实例化自定义Dataset，加载数据集
    mri = MRIDataset(url, W, H, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    # 分割数据集
    if split is None:
        split = [0.7, 0.3]
    n_total = len(mri)
    n_train = int(split[0] * n_total)
    n_test = n_total - n_train
    train_set, test_set = random_split(mri, [n_train, n_test])
    # 将数据集封装入DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

def train(loader, device="cpu",batch_size=16, lr=1e-3, epochs=10, model="model/"):
    assert(isinstance(loader, DataLoader))
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir='runs/train_'+formatted_time)
    net = UNet(1, 1) # 实例化网络
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 优化器
    criterion = nn.BCELoss() # 损失函数
    model_savepath = model+'/' + formatted_time + '/'
    create_dir_not_exist(model_savepath)
    record = open(model_savepath+'record.txt', "a")
    record.write('tringing_'+ formatted_time + "\n")
    # 开始训练
    for epoch in range(epochs):
        net.train()
        batch_id = 1
        train_num = 0
        total_batch = len(loader)
        total_loss = 0
        total_dice = 0
        total_ppv = 0
        total_jaccard = 0
        total_hd95 = 0
        for image, label in loader:
            image = image/255 #输入归一化
            logit = net(image.to(device))
            #计算指标
            np_logit = logit.cpu().detach().numpy()
            np_label = label.numpy()
            dice = dice_coef(np_logit,np_label)
            ppv = ppv_compute(np_logit,np_label)
            jaccard = jaccard_compute(np_logit,np_label)
            hd95 = hd95_compute(np_logit,np_label)
            # 更新模型参数
            loss_bce = criterion(logit, label.float().to(device))  # 计算损失
            loss_dice = 1 - dice
            loss = (loss_bce + loss_dice) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录每个batch的损失和指标
            total_loss += loss.item()
            total_dice += dice
            total_ppv += ppv
            total_jaccard += jaccard
            total_hd95 += hd95

            # if batch_id % batch_size  == 0:
            #     r = "[Epoch: {}/{}][Batch: {}/{}][Loss: {}][Dice: {}][jaccard: {}][ppv: {}][hd95: {}]"\
            #         .format(epoch + 1, epochs, batch_id, total_batch, loss.item(),dice,jaccard,ppv,hd95)
            #     # print(r)
            #     record.write(r + "\n")

            # TensorBoard记录
            writer.add_scalar('Loss/train', loss.item(), train_num)
            writer.add_scalar('Dice/train', dice, train_num)
            writer.add_scalar('Jaccard/train', jaccard, train_num)
            writer.add_scalar('PPV/train', ppv, train_num)
            writer.add_scalar('HD95/train', hd95, train_num)
            writer.add_images('Input Images', image * 255, train_num)
            writer.add_images('Output Images', logit * 255, train_num)
            writer.add_images('Label', label * 255, train_num)

            train_num += 1

        torch.save(net.state_dict(), model_savepath + 'train_' + str(epoch) + '.pth')
    writer.close()
    record.write("[Training Finished]"+ "\n")
    # 存储网络参数


def predict(loader, model="model/model.pth", pred_dir="/pred",device="cpu",record_pred="./model/record_pred.txt"):
    assert (isinstance(loader, DataLoader))
    # 初始化网络并加载权重
    net = UNet(1, 1)
    net = net.to(device)
    net.load_state_dict(torch.load(model))
    net.eval()
    #预测值归0
    total_dice = 0
    total_jaccard = 0
    total_ppv = 0
    total_hd95 =0
    # 预测
    order = 1
    record_pred = open(record_pred, "w")
    for image, label in loader:
        pred = net(image)
        for i, p in zip(image.detach().numpy(), pred.detach().numpy()):
            # i: [1, w, h], p: [1, w, h]
            i, p = i[0], p[0]
            p[p >= 0.5] = 255
            p[p < 0.5] = 0
            # 存储预测结果
            cv2.imwrite("{}/{}.jpg".format(pred_dir, order), i * 255)
            cv2.imwrite("{}/{}_L.jpg".format(pred_dir, order), p * 255)
            order += 1
        # 计算指标
        np_pred = np.asarray(pred.cpu())
        np_label = np.asarray(label)
        dice = dice_coef(np_pred, np_label)
        ppv = ppv_compute(np_pred, np_label)
        jaccard = jaccard_compute(np_pred, np_label)
        hd95 = hd95_compute(np_pred, np_label)
        r = "[order:{}][Dice: {}][jaccard: {}][ppv: {}][hd95: {}]" \
            .format(order,dice, jaccard, ppv, hd95)
        print(r)
        total_dice = total_dice+dice
        total_ppv = total_ppv+ppv
        total_jaccard = total_jaccard+jaccard
        total_hd95 = total_hd95+hd95
    r = "[Dice: {}][jaccard: {}][ppv: {}][hd95: {}]" \
        .format(total_dice/order, total_jaccard/order, total_ppv/order, total_hd95/order)
    record_pred.write(r + "\n")

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('--base_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation',
                        help='数据集真值合并预处理根目录')
    parser.add_argument('--output_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation/label_combine',
                        help='数据集真值合并预处理输出目录')
    parser.add_argument('--data_preprocess', default=False,
                        help='是否进行数据集真值合并预处理(default: True)')
    parser.add_argument('--model_path', type=str,
                        default='./model',
                        help='预训练模型路径')
    parser.add_argument('--pred_path', type=str,
                        default='./pred',
                        help='测试结果输出路径')
    parser.add_argument('--H', type=int, default=320,
                        help='resize后图片的高(default: 320)')
    parser.add_argument('--W', type=int, default=240,
                        help='resize后图片的宽(default: 240)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='barch_size大小(default:4)')
    parser.add_argument('--lr', type=int, default=1e-2,
                        help='训练时学习率(default:1e-4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epoch大小(default:100)')
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='是否使用GPU加速网络(default: False)')
    parser.add_argument('--no_display', default=False, action='store_true',
                        help='是否要展示预测结果图片(default: False).')
    parser.add_argument('--write', action='store_true', default=True,
                        help='是否要保存预测结果图片(default: True)')
    parser.add_argument('--write_dir', type=str, default='./output',
                        help='保存预测结果图片地址(default: ./output).')
    opt = parser.parse_args()
    print(opt)

    # 数据预处理
    if(opt.data_preprocess):
        # 可以根据需要选择不同的subject进行处理
        subjects = ['35', '100']  # 例如，处理'35'和'100'

        for subject in subjects:
            merge_labels_and_save(opt.base_path, subject, opt.output_path)

    # 加载数据集
    train_loader, test_loader = load_data(opt.base_path,opt.W,opt.H,opt.batch_size)

    if(opt.cuda):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("目前使用的为：" + str(device))
    #训练
    train(train_loader,device,opt.batch_size,opt.lr,opt.epochs,opt.model_path)
    # 预测
    # predict(test_loader,device,opt.pred_dir)

if __name__ == "__main__":
    main()
