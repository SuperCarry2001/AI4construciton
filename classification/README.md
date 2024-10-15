# 功能介绍
包括多种网络的图片分类网络，可用作动作分类、工程机械分类等任务

# 数据集的格式要求
- 数据集需按照如下的格式构造好
```
dataset-folder/
├── train/
│   ├── label_1
│   │   ├── picture_1
│   │   └── picture_2
│   │     ...
│   │   └── picture_m
│   └── label_2
│   ...
│   └── label_n
├── val/
│   ├── label_1
│   └── label_2
│   ...
│   └── label_n
├── test/
│   ├── label_1
│   └── label_2
│   ...
│   └── label_n

```


# 使用方法
## 参数说明
```
parser.add_argument('--n',default=5,type=int,help='分类的类别数')
parser.add_argument('--dataset_path',default=r'C:\Users\win10\Desktop\ZJ\DWQ_dataset',type=str,help='整理好的数据集路径')
parser.add_argument('--weight_path', default='',type=str,help='测试时输入权重文件的路径')
parser.add_argument('--finetune',default=True, action='store_true', help='是否为迁移学习' )
parser.add_argument('--train', action='store_true', help='是否为训练模式' )
parser.add_argument('--test', action='store_true', help='是否测试模式' )
parser.add_argument('--model', type=str, default='resnet101', choices=['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet201'],
                    help='选择模型')

```
## 使用说明（已经安装好环境）
```
# 训练
python classification.py --train --n 类别数 --dataset_path 数据集路径 --model 指定模型

#测试
python classification.py --test --n 类别数 --dataset_path 数据集路径 --model 指定模型 --weight_path 权重文件路径
```

## demo
```
## 使用大湾区拍摄的挖掘机动作分类数据集 ##
python classification.py --train --n 5 --dataset_path C:/Users/win10/Desktop/ZJ/DWQ_dataset --model alexnet --finetune
python classification.py --test --n 5 --dataset_path C:/Users/win10/Desktop/ZJ/DWQ_dataset --model alexnet --weight_path C:/Users/win10/Desktop/ZJ/DWQ_dataset/best_alexnet.pth

## 准确率达到95%以上 ##
```