import cv2
import argparse
import glob
import random
import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import random_split
from U_net_tools import *

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target)
        return 1 - (2. * intersection + self.epsilon) / (union + self.epsilon)

class MRIDataset(Dataset):
    def __init__(self, url, W, H, transform=None):
        self.label = glob.glob(os.path.join(url, 'label_combine/*/*/*.jpg'))
        self.transform = transform
        self.W = W
        self.H = H

    def augment(self, img, flip):
        if flip in [-1, 0, 1]:
            img_flipped = cv2.flip(img, flip)
            return img_flipped
        return img

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


def filter_non_zero_samples(base_path, subject, output_path):
    """
    筛选并保存真值不为0的样本

    :param base_path: 数据集根目录
    :param subject: 数据集子文件夹（如'100'）
    :param output_path: 输出保存路径
    """
    label_path = os.path.join(base_path, 'label', f'{subject}label')

    # 获取label文件夹中的所有子文件夹
    folders = os.listdir(label_path)

    # 遍历每个子文件夹
    for folder in folders:
        label_folder_path = os.path.join(label_path, folder)
        left_label_folder = os.path.join(label_folder_path, folder.replace('tal_noscale', 'L').replace('ACPC', 'L'))
        right_label_folder = os.path.join(label_folder_path, folder.replace('tal_noscale', 'R').replace('ACPC', 'R'))

        # 检查是否同时存在L和R文件夹
        if not os.path.exists(left_label_folder) or not os.path.exists(right_label_folder):
            print(f"Missing L or R folder in {folder}. Skipping folder.")
            continue

        # 获取标签图像（假设图像是jpg格式）
        left_labels = sorted([img for img in os.listdir(left_label_folder) if img.endswith('.jpg')])
        right_labels = sorted([img for img in os.listdir(right_label_folder) if img.endswith('.jpg')])

        # 确保每个文件夹中的图像数量一致
        if len(left_labels) != len(right_labels):
            print(f"Label count mismatch in {folder}. Skipping folder.")
            continue

        # 遍历每个标签图像
        for idx in range(len(left_labels)):
            left_label_path = os.path.join(left_label_folder, left_labels[idx])
            left_label = cv2.imread(left_label_path, cv2.IMREAD_GRAYSCALE)

            right_label_path = os.path.join(right_label_folder, right_labels[idx])
            right_label = cv2.imread(right_label_path, cv2.IMREAD_GRAYSCALE)

            # 确保标签尺寸一致
            if left_label.shape != right_label.shape:
                print(f"Label size mismatch for {left_labels[idx]} and {right_labels[idx]} in {folder}. Skipping.")
                continue

            # 标签值相加
            combined_label = cv2.add(left_label, right_label)  # 标签值相加

            # 如果真值最大值小于50，则跳过
            if np.max(combined_label)<50:
                # print(f"Combined label {left_labels[idx]} in {folder} is all zeros. Skipping.")
                continue

            # 输出路径
            output_folder = os.path.join(output_path, subject, folder)
            os.makedirs(output_folder, exist_ok=True)

            # 保存文件名保持原始文件名结构
            if (label_folder_path[-4:] == 'ACPC'):
                output_image_name = os.path.basename(left_labels[idx].replace('L', 'ACPC'))
            else:
                output_image_name = os.path.basename(left_labels[idx].replace('L', 'tal_noscale'))
            output_image_path = os.path.join(output_folder, output_image_name)
            cv2.imwrite(output_image_path, combined_label)

            # 可视化或调试（可选）
            # cv2.imshow(f"Filtered Label {folder} - {idx}", combined_label)
            # cv2.waitKey(0)

    # cv2.destroyAllWindows()

def load_data(url="./dataset", W=240, H=320, batch_size=64, shuffle=True, split=None):
    # 实例化自定义Dataset，加载训练数据集
    mri = MRIDataset(url, W, H, transform=transforms.Compose([
        transforms.ToTensor()  # 转换为张量
    ]))
    # 分割数据集
    split = split or [0.7, 0.3]
    n_train = int(split[0] * len(mri))
    train_set, test_set = random_split(mri, [n_train, len(mri) - n_train])
    # 将数据集封装入DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(loader, device="cpu",batch_size=4, lr=1e-3, epochs=10, model="model/",threshold = 0.5):
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
    print("目前使用的为：" + str(device))
    # 开始训练
    train_num = 0
    for epoch in range(epochs):
        net.train()
        batch_id = 1
        total_batch = len(loader)
        total_loss = 0
        total_dice = 0
        total_ppv = 0
        total_jaccard = 0
        total_hd95 = 0
        for image, label in loader:
            logit = net(image.to(device))
            # 更新模型参数
            loss_bce = criterion(logit, label.float().to(device))  # 计算损失
            dice_loss = DiceLoss()
            loss_dice = dice_loss(logit, label.float().to(device))
            loss = (loss_bce + loss_dice) / 2
            # loss = loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 阈值分割
            binarized_logit = torch.where(logit < threshold, torch.tensor(0.0, device=logit.device), logit)
            binarized_logit = torch.where(binarized_logit >= threshold, torch.tensor(1.0, device=binarized_logit.device), binarized_logit)
            # 计算指标
            np_logit = binarized_logit.cpu().detach().numpy()
            np_label = label.numpy()
            dice = dice_coef(np_logit, np_label)
            ppv = ppv_compute(np_logit, np_label)
            jaccard = jaccard_compute(np_logit, np_label)
            hd95 = hd95_compute(np_logit, np_label)
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

            r = "[step: {}][Loss: {}][Dice: {}][jaccard: {}][ppv: {}][hd95: {}]" \
                         .format(train_num, loss.item(),dice,jaccard,ppv,hd95)
            record.write(r + "\n")

            # TensorBoard记录
            writer.add_scalar('Loss/train', loss.item(), train_num)
            writer.add_scalar('Dice/train', dice, train_num)
            writer.add_scalar('Jaccard/train', jaccard, train_num)
            writer.add_scalar('PPV/train', ppv, train_num)
            writer.add_scalar('HD95/train', hd95, train_num)
            writer.add_images('Input Images', image, train_num)
            writer.add_images('Output Images', np_logit, train_num)
            writer.add_images('Label', label, train_num)

            train_num += 1
            if train_num % 100 == 0:
                torch.save(net.state_dict(), model_savepath + 'train_' + str(train_num) + 'echo.pth')
    writer.close()
    record.write("[Training Finished]"+ "\n")
    # 存储网络参数

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch Automatic-segmentation-of-hippocampus train.')
    parser.add_argument('--base_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation',
                        help='数据集真值合并预处理根目录')
    parser.add_argument('--output_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation/label_combine',
                        help='数据集真值合并预处理输出目录')
    parser.add_argument('--data_preprocess', default=False,
                        help='是否进行数据集真值合并预处理(default: True)')
    parser.add_argument('--model_path', type=str,
                        default='./model',
                        help='保存预训练模型路径')
    parser.add_argument('--H', type=int, default=320,
                        help='resize后图片的高(default: 320)')
    parser.add_argument('--W', type=int, default=240,
                        help='resize后图片的宽(default: 240)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='barch_size大小(default:4)')
    parser.add_argument('--lr', type=int, default=1e-4,
                        help='训练时学习率(default:1e-4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epoch大小(default:100)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='阈值threshold大小(default:0.5)，大于此值才判断为海马体区域')
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='是否使用GPU加速网络(default: False)')
    opt = parser.parse_args()
    print(opt)

    # 数据预处理
    if(opt.data_preprocess):

        # 使用100作为训练集
        filter_non_zero_samples(opt.base_path, '100', opt.output_path)

        # 使用35作为测试集（无需过滤）
        # predict_path = os.path.join(opt.base_path, 'label', '35label')

    # 加载数据集
    train_loader, test_loader = load_data(opt.base_path,opt.W,opt.H,opt.batch_size)

    if(opt.cuda):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #训练
    train(train_loader,device,opt.batch_size,opt.lr,opt.epochs,opt.model_path,opt.threshold)

if __name__ == "__main__":
    main()
