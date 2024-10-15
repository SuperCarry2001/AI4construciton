# 功能介绍
基于MOCS数据集实现目标检测

# 数据集的格式要求
- 数据集需按照如下的格式构造好（类似COCO数据集、MOCS数据集）
- /data_example.json
```
{
    "images": [
        {
            "height": 675,
            "width": 1200,
            "id": 1,
            "file_name": "0000001.jpg"
        },
    ],
    "categories": [
        {
            "supercategory": "Construction",
            "id": 1,
            "name": "Worker"
        },
    ],
    "annotations": [
        {
            "image_id": 19404,
            "bbox": [
                295.0,
                334.0,
                12.0,
                20.0
            ],
            "category_id": 11,
            "id": 116945
        }
    ]
}
```
- 在 transfer_dataset.py中找到合适的函数把数据集转化成要求的格式

# 使用方法
## yolo-v8(推荐)
```
使用transfer_dataset.py中的mocs2yolo函数，把json文件转化成对应的.txt文件存储
yolov8_original.pt是官方预训练的权重文件
MOCS.yaml上是网络的设置，包括训练集、验证集、测试集的路径，类别数
```
## 使用说明（已经安装好环境）
```
# train
python yolov8.py
```
## Reference
Here are some great resources we benefit:
- YOLOv8 is from [YOLOv8](https://github.com/ultralytics/ultralytics).
- MOCS dataset is from [MOCS dataset](http://www.anlab340.com/Archives/IndexArctype/index/t_id/17.html).
