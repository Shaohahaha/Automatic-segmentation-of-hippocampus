# Automatic-segmentation-of-hippocampus
![预测单张图片可视化](https://github.com/Shaohahaha/Automatic-segmentation-of-hippocampus/raw/main/image/show1.jpg)
![预测.nii文件可视化](https://github.com/Shaohahaha/Automatic-segmentation-of-hippocampus/raw/main/image/show2.jpg)

## 数据集：
    1.https://www.kaggle.com/datasets/sabermalek/mrihs
    2.https://portal.conp.ca/dataset?id=projects/calgary-campinas

## 预训练模型和可执行程序
链接: [https://pan.baidu.com/s/1zH4z2XuDQ35S6IzwzJNKKQ](https://pan.baidu.com/s/1zH4z2XuDQ35S6IzwzJNKKQ) 提取码: vm93

## 实验结果：
| 模型(训练数据集)\指标           | Dice     | Jaccard  | ppv      | Hd95(%)  | time(ms/image) |
|--------------------------------|----------|----------|----------|----------|----------------|
| U_net(MRI_Hippocampus_Segmentation/100 | 0.944883 | 0.993051 | 0.947766 | 0.433784 | 17.114667      |
| U_net(MRI_Hippocampus_Segmentation/100 calgary_campinas_version | 0.920224 | 0.981873 | 0.928092 | 0.458490 | 19.333923      |
| Res_U_net(MRI_Hippocampus_Segmentation/100 | 0.914852 | 0.977644 | 0.919404 | 0.460553 | 17.313071      |
| Res_U_net(MRI_Hippocampus_Segmentation/100 calgary_campinas_version | 0.933903 | 0.988218 | 0.937460 | 0.438584 | 19.375479      |


## 文件路径：
    .
    |-- dataset
    |   |--- MRI_Hippocampus_Segmentation
    |   |   |-- label
    |   |   `-- original
    |   `--- calgary_campinas_version
    |       |-- hippocampus_staple
    |       `-- Original	
    |
	|-- <预训练模型>.pth（百度网盘下载）
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
pip install tk matplotlib Pillow nibabel torch torchvision opencv-python trimesh numpy scipy argparse hausdorff numba
```
（可选）安装自己对应GPU版本的torch,安装后后面运行步骤中可输入--cuda加速运行。
```Bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/<对应版本>
```
## 程序运行
查看可选设定项
```Bash
python train.py -help
```
第一次运行需要加上--AD_data_preprocess进行数据集预处理，耗时较长，后续训练不用再输入
下面命令以训练Unet为例，还可换为ResUnet，只需在net_tpye可选项中将'Unet'替换为'ResUnet'。
```Bash
python train.py --AD_data_preprocess --net_type='Unet' (--cuda)
```

使用两个数据集进行混合训练，同样先要进行数据预处理（第一次需要，后面不用）
```Bash
python train.py --C_data_preprocess --traindata_aug --net_type='Unet' (--cuda)
```

训练过程中以及结束后可在terminal里在项目最开始路径下输入，查看可视化训练指标
```Bash
tensorboard --logdir=runs
```

或直接使用预训练模型进行预测验证实验数据
```Bash
python compute_metrics.py --model_path='<预训练模型路径>' (--cuda)
```



## 神经网络框架代码参考：
    https://blog.csdn.net/qq_43116030/article/details/114803171