import cv2
import argparse
import glob
import random
import time
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from U_net_tools import *


class MRIDataset(Dataset):
    def __init__(self, url, W, H, transform=None):
        self.label = glob.glob(os.path.join(url, 'label_combine_pred/*/*/*.jpg'))
        self.transform = transform
        self.W = W
        self.H = H

    def __getitem__(self, index):
        # 通过索引获取原图和标签的URL
        label_url = self.label[index]
        img_url = label_url.replace("label_combine_pred","original")
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
        # 转换数据
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

    def __len__(self):
        return len(self.label)

def load_data(url="./dataset", W=240, H=320, batch_size=4, shuffle=False, split=None):
    # 实例化自定义Dataset，加载训练数据集
    mri = MRIDataset(url, W, H, transform=transforms.Compose([
        transforms.ToTensor()  # 转换为张量
    ]))
    test_loader = DataLoader(mri, batch_size=batch_size, shuffle=False)

    return test_loader

def merge_labels_and_save(predict_path, output_path):
    """
    合并左右标签并保存结果

    :param predict_path: 测试数据集根目录
    :param output_path: 输出保存路径
    """
    original_path = predict_path
    label_path = original_path.replace('35','35label').replace('original','label')

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
            output_folder = os.path.join(output_path, '35', folder)
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

def predict(loader, model="model/model.pth",net_type='Unet', pred_dir="/pred",write=True,show = True,device="cpu",threshold = 0.5,record_pred="./model/record_pred.txt"):
    assert (isinstance(loader, DataLoader))
    # 初始化网络并加载权重
    if net_type == 'Unet':
        net = UNet(1, 1)  # 实例化网络
    else:
        net = ResUNet(1, 1)  # 实例化网络
    net = net.to(device)
    if str(device)=="cpu":
        net.load_state_dict(torch.load(model,map_location='cpu'))
    else:
        net.load_state_dict(torch.load(model))
    net.eval()
    #预测值归0
    total_dice = 0
    total_jaccard = 0
    total_ppv = 0
    total_hd95 =0
    total_time = 0
    # 预测
    order = 1
    porder = 1
    create_dir_not_exist(pred_dir)
    create_dir_not_exist(pred_dir+'/'+net_type)
    pred_dir = pred_dir + '/' + net_type
    # create_dir_not_exist(record_pred)
    # record_pred=record_pred+'/record_pred.txt'
    print("目前使用的为：" + str(device))
    # record_pred = open(record_pred, "a+")
    for image, label in loader:
        start_time = time.time()
        tensor_image = image.to(device)
        pred = net(tensor_image)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        if write:
            for i, p ,l in zip(image.detach().numpy(), pred.cpu().detach().numpy(),label.detach().numpy()):
                # i: [1, w, h], p: [1, w, h]
                i, p, l = i[0], p[0], l[0]
                p[p >= threshold] = 1
                p[p < threshold] = 0
                # 检查是否存在非零区域（即是否有海马体）
                if np.any(p == 1):  # 如果有海马体区域
                    # 存储预测结果
                    output_image = np.zeros((i.shape[0], i.shape[1], 3), dtype=np.uint8)

                    output_image[(p == 1) & (l == 1)] = [0, 255, 0]  # Green

                    output_image[(p == 1) & (l == 0)] = [0, 0, 255]  # Red

                    output_image[(p == 0) & (l   == 1)] = [255, 0, 0]  # Blue

                    output_image[:, :, 0] = np.maximum(output_image[:, :, 0], i)

                    output_image_name = f"pred_{porder}.jpg"
                    output_image_path = os.path.join(pred_dir, output_image_name)

                    cv2.imwrite(output_image_path, output_image)
                    porder += 1
        order += 1
        # 阈值分割
        binarized_pred = torch.where(pred < threshold, torch.tensor(0.0, device=pred.device), pred)
        binarized_pred = torch.where(binarized_pred >= threshold,torch.tensor(1.0, device=binarized_pred.device), binarized_pred)
        # 计算指标
        np_pred = binarized_pred.cpu().detach().numpy()
        np_label = label.numpy()
        dice = dice_coef(np_pred, np_label)
        ppv = ppv_compute(np_pred, np_label)
        jaccard = jaccard_compute(np_pred, np_label)
        hd95 = hd95_compute(np_pred, np_label)
        # r = "[order:{}][Dice: {}][jaccard: {}][ppv: {}][hd95: {}][time:{}]" \
        #         .format(order,dice, jaccard, ppv, hd95,elapsed_time_ms)
        # if show:
        #     print(r)
        # record_pred.write(r + "\n")
        total_dice = total_dice+dice
        total_ppv = total_ppv+ppv
        total_jaccard = total_jaccard+jaccard
        total_hd95 = total_hd95+hd95
        total_time = total_time + elapsed_time_ms
    r = "[Dice: {}][jaccard: {}][ppv: {}][hd95: {}][time:{}]" \
        .format(total_dice/order, total_jaccard/order, total_ppv/order, total_hd95/order,total_time/order)
    if show:
        print(r)
    # record_pred.write(r + "\n")
    # record_pred.close()

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch Automatic-segmentation-of-hippocampus predict')
    parser.add_argument('--base_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation',
                        help='数据集真值合并预处理输出目录')
    parser.add_argument('--data_preprocess', default=False, action='store_true',
                        help='是否进行数据集真值合并预处理(default: False)')
    parser.add_argument('--output_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation/label_combine_pred',
                        help='数据集真值合并预处理输出目录')
    parser.add_argument('--net_type', type=str, default='Unet',
                        choices=['Unet', 'ResUnet'],
                        help='网络模型选择 (default: Unet).')
    parser.add_argument('--predict_path', type=str, default='./dataset/MRI_Hippocampus_Segmentation/original/35',
                        help='数据集真值合并预处理输出目录')
    parser.add_argument('--H', type=int, default=320,
                        help='resize后图片的高(default: 320)')
    parser.add_argument('--W', type=int, default=240,
                        help='resize后图片的宽(default: 240)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='barch_size大小(default:4)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='阈值threshold大小(default:0.5)，大于此值才判断为海马体区域')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='是否使用GPU加速网络(default: False)')
    parser.add_argument('--model_path', type=str, default='./model/ResUnet.pth',
                        help='训练完成模型路径')
    parser.add_argument('--ifshow', action='store_true', default=True,
                        help='是否要print预测指标结果(default: True)')
    parser.add_argument('--write', action='store_true', default=True,
                        help='是否要保存预测结果图片(default: True)')
    parser.add_argument('--write_dir', type=str, default='./output',
                        help='保存预测结果图片地址(default: ./output).')
    opt = parser.parse_args()
    print(opt)

    # 数据预处理
    if(opt.data_preprocess):

        merge_labels_and_save(opt.predict_path, opt.output_path)

    # 加载数据集
    test_loader = load_data(opt.base_path,opt.W,opt.H,opt.batch_size)

    if(opt.cuda):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #预测
    record_path = opt.write_dir+opt.model_path[34:39]
    predict(test_loader, opt.model_path,opt.net_type,opt.write_dir,opt.write,opt.ifshow,device,opt.threshold,record_pred=record_path)

if __name__ == "__main__":
    main()
