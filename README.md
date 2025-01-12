# Automatic-segmentation-of-hippocampus
![预测单张图片可视化](https://github.com/Shaohahaha/Automatic-segmentation-of-hippocampus/raw/main/image/show1.jpg)
![预测.nii文件可视化](https://github.com/Shaohahaha/Automatic-segmentation-of-hippocampus/raw/main/image/show2.jpg)

## 数据集：
    1.https://www.kaggle.com/datasets/sabermalek/mrihs（目前使用）
    2.https://portal.conp.ca/dataset?id=projects/calgary-campinas（还未使用）

## 预训练模型和可执行程序
链接: [https://pan.baidu.com/s/1zH4z2XuDQ35S6IzwzJNKKQ](https://pan.baidu.com/s/1zH4z2XuDQ35S6IzwzJNKKQ) 提取码: vm93

## 文件路径：
    .
    |-- dataset
    |   `--- MRI_Hippocampus_Segmentation
    |       |-- label
    |       `-- original
	|
	|-- pretrained_model.pth（百度网盘下载）
	|
	|-- predict.py
    |
	|-- compute_metrics.py
	|
	|-- U_net_tools.py
    |
	`-- train.py

## 环境安装
创建新环境
```Bash
conda create -n py38hip python=3.8
```
进入新环境
```Bash
conda activate py38hip
```
安装所需库
```Bash
pip install tk matplotlib Pillow nibabel torch torchvision opencv-python trimesh numpy scipy argparse
```
（可选）安装GPU版本的torch
```Bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
## 程序运行
训练
```Bash
python train.py
```
或直接使用预训练模型（和predict.py程序放同一路径里）进行预测
```Bash
python predict.py
```

## 神经网络框架代码参考：
    https://blog.csdn.net/qq_43116030/article/details/114803171