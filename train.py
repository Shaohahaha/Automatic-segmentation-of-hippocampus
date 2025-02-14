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
import nibabel as nib
from PIL import Image

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target)
        return 1 - (2. * intersection + self.epsilon) / (union + self.epsilon)

class MRIDataset(Dataset):
    def __init__(self,traindata_aug, url,url1, W, H, transform=None):
        self.label = glob.glob(os.path.join(url, 'label_combine/*/*/*.jpg'))
        if traindata_aug:
            self.label+= glob.glob(os.path.join(url1, 'hippocampus_staple/hippocampus_staple/*/*.jpg'))
            random.shuffle(self.label)
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
        img_url = label_url.replace("label_combine","original").replace("hippocampus_staple","Original").replace("_Original","")
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


def nii_to_images(nii_file):
    # 1. 读取 .nii 文件
    nii_img = nib.load(nii_file)
    images = nii_img.get_fdata()  # 获取图像数据

    # 2. 获取 .nii 文件的路径和文件名（不包含扩展名）
    nii_dir = os.path.dirname(nii_file)
    nii_name = os.path.splitext(os.path.basename(nii_file))[0]

    # 3. 创建与 .nii 文件名称相同的新文件夹
    output_dir = os.path.join(nii_dir, nii_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. 遍历切片，并保存每一张切片为图片
    depth, height, width = images.shape
    for z in range(width):
        # 获取当前切片
        slice_data = images[:, :, z]

        # 将切片数据归一化为 0-255 的范围（适用于灰度图像）
        slice_data_normalized = np.interp(slice_data, (slice_data.min(), slice_data.max()), (0, 255)).astype(np.uint8)

        # 检查图像是否全为255
        if np.all(slice_data_normalized == 255):
            continue

        # 将切片数据转换为 PIL 图像
        slice_img = Image.fromarray(slice_data_normalized)

        # 顺时针旋转 90 度
        rotated_image = slice_img.rotate(-90, expand=True)

        # 保存图像到新文件夹，格式改为 .jpg
        slice_filename = os.path.join(output_dir, f"{nii_name}_slice_{z + 1}.jpg")
        rotated_image.save(slice_filename, 'JPEG', quality=95)

def convert_all_nii_in_folder(folder_path):
    # 1. 获取文件夹中的所有 .nii 文件
    nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 2. 对每个 .nii 文件调用 nii_to_images 函数进行处理
    for nii_file in nii_files:
        nii_file_path = os.path.join(folder_path, nii_file)
        nii_to_images(nii_file_path)

def filter_non_zero_samples_AD(base_path, subject, output_path):
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

def load_data(traindata_aug=False,url="./dataset/MRI_Hippocampus_Segmentation",url1='./dataset/calgary_campinas_version', W=240, H=320, batch_size=64, shuffle=True, split=None):
    # 实例化自定义Dataset，加载训练数据集
    mri = MRIDataset(traindata_aug,url,url1, W, H, transform=transforms.Compose([
        transforms.ToTensor()  # 转换为张量
    ]))
    # 将数据集封装入DataLoader
    train_loader = DataLoader(mri, batch_size=batch_size, shuffle=shuffle)

    return train_loader

def train(loader, net_type,device="cpu",batch_size=4, lr=1e-3, epochs=10, model="model/",threshold = 0.5):
    assert(isinstance(loader, DataLoader))
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir='runs/train_'+ str(net_type) +'_'+formatted_time)
    if net_type == 'Unet':
        net = UNet(1, 1) # 实例化网络
    else:
        net = ResUNet(1, 1)  # 实例化网络
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 优化器
    criterion = nn.BCELoss() # 损失函数
    model_savepath = model+'/' + formatted_time + '/'
    create_dir_not_exist(model_savepath)
    record = open(model_savepath+'record.txt', "a")
    record.write('tringing_'+ net_type +'_' +formatted_time + "\n")
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
    parser.add_argument('--AD_base_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation',
                        help='AD患者数据集真值合并预处理根目录')
    parser.add_argument('--AD_output_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation/label_combine',
                        help='AD患者数据集真值合并预处理输出目录')
    parser.add_argument('--C_base_path', type=str, default='./dataset/calgary_campinas_version',
                        help='正常人数据集预处理根目录')
    parser.add_argument('--AD_data_preprocess', default=False, action='store_true',
                        help='是否进行AD患者数据集真值合并预处理(default: False)')
    parser.add_argument('--C_data_preprocess', default=False, action='store_true',
                        help='是否进行正常人数据集真值合并预处理(default: False)')
    parser.add_argument('--traindata_aug', default=False, action='store_true',
                        help='是否使用正常人数据集和AD数据集进行训练(default: False)')
    parser.add_argument('--net_type', type=str, default='Unet',
                        choices=['Unet', 'ResUnet'],
                        help='网络模型选择 (default: Unet).')
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
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='是否使用GPU加速网络(default: False)')
    opt = parser.parse_args()
    print(opt)

    # 数据预处理
    if(opt.AD_data_preprocess):

        # 使用100作为训练集
        filter_non_zero_samples_AD(opt.AD_base_path, '100', opt.AD_output_path)

        # 使用35作为测试集（无需过滤）
        # predict_path = os.path.join(opt.base_path, 'label', '35label')

    if (opt.C_data_preprocess):

        # convert_all_nii_in_folder(opt.C_base_path + '/Original/Original')

        convert_all_nii_in_folder(opt.C_base_path + '/hippocampus_staple/hippocampus_staple')

    # 加载数据集
    train_loader = load_data(opt.traindata_aug,opt.AD_base_path,opt.C_base_path,opt.W,opt.H,opt.batch_size)

    if(opt.cuda):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # #训练
    train(train_loader,opt.net_type,device,opt.batch_size,opt.lr,opt.epochs,opt.model_path,opt.threshold)

if __name__ == "__main__":
    main()
