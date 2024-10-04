## detection part ##

## using differernt methods (YOLO\fastrcnn\...) ##
## using MOCS dataset (same data structure and format as COCO dataset) ##
#### difference1: category_id, MOCS is from 1 13
#### difference2: train\val\test img_file and json file are all separated, that is, i have train_json\val_json\test_json, and same for img_file

"""
step1 dataloader part: like COCO dataset  
step2 train part: finetune or train from zero, saving best weight
step3 test part: return result like mAP and so on

"""

import json
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score


class MOCSDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        print(f"Loaded {len(self.annotations['images'])} images and {len(self.annotations['annotations'])} annotations.")

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        boxes = []
        labels = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                x, y, width, height = ann['bbox']
                # 仅保留有效的边界框
                if width > 0 and height > 0:
                    boxes.append([x, y, x + width, y + height])  # 转换为 [x_min, y_min, x_max, y_max]
                    labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels

def create_dataloader(json_file, img_dir, batch_size=16, shuffle=True):
    dataset = MOCSDataset(json_file, img_dir, transform=transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))


# 模型定义
def get_fastrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


# 训练与验证
def train_model(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=0.03):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, boxes, labels in tqdm(train_dataloader):
            images = [image.to(device) for image in images]
            targets = [{"boxes": box.to(device), "labels": label.to(device)} for box, label in zip(boxes, labels)]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 保存模型
    torch.save(model.state_dict(), r"C:\Users\ZJ\Desktop\AI4construciton\detection\model_last.pth")

def compute_map(model, dataloader, num_classes, device):
    model.eval()  # 设置模型为评估模式
    all_boxes = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, boxes, labels in dataloader:
            images = [image.to(device) for image in images]
            targets = [{"boxes": box.to(device), "labels": label.to(device)} for box, label in zip(boxes, labels)]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # 提取每个图像的预测框、标签和置信度
                boxes_pred = output['boxes'].cpu().numpy()
                labels_pred = output['labels'].cpu().numpy()
                scores_pred = output['scores'].cpu().numpy()

                # 过滤低置信度的预测（例如，置信度小于 0.5）
                indices = scores_pred > 0.5
                boxes_pred = boxes_pred[indices]
                labels_pred = labels_pred[indices]
                scores_pred = scores_pred[indices]

                all_boxes.append(boxes_pred)
                all_labels.append(labels_pred)
                all_scores.append(scores_pred)

    # 计算 mAP
    mAPs = []
    for class_id in range(num_classes):
        true_boxes = []
        true_labels = []
        pred_boxes = []
        pred_scores = []

        for idx in range(len(all_labels)):
            # 获取当前图像的真实标签和框
            true_labels_class = (labels[idx] == class_id).nonzero(as_tuple=True)[0] if len(labels[idx]) > 0 else []
            true_boxes_class = boxes[idx][true_labels_class] if true_labels_class.numel() > 0 else torch.empty((0, 4))

            true_boxes.append(true_boxes_class)
            true_labels.append([class_id] * len(true_boxes_class))
            pred_boxes.append(all_boxes[idx][all_labels[idx] == class_id])
            pred_scores.append(all_scores[idx][all_labels[idx] == class_id])

        # 转换为张量并移动到 GPU
        true_boxes_tensor = torch.cat(true_boxes).to(device) if len(true_boxes) > 0 else torch.empty((0, 4)).to(device)
        pred_boxes_tensor = torch.cat(pred_boxes).to(device) if len(pred_boxes) > 0 else torch.empty((0, 4)).to(device)

        # 计算 IoU
        if true_boxes_tensor.numel() > 0 and pred_boxes_tensor.numel() > 0:
            iou_scores = box_iou(true_boxes_tensor, pred_boxes_tensor)

            # 计算 Average Precision
            average_precision = average_precision_score(true_labels, pred_scores)
        else:
            average_precision = 0.0

        mAPs.append(average_precision)

    mean_ap = sum(mAPs) / len(mAPs)
    print(f"Mean Average Precision (mAP) @ IoU=0.5: {mean_ap:.4f}")

    return mean_ap


# 使用示例
# mean_ap = compute_map(model, val_dataloader, num_classes, device)


# 使用示例
num_classes = 14  # Background + 13 classes
model = get_fastrcnn_model(num_classes)
# Create data loaders for training and validation
train_json_file = r"E:\MOCS_dataset\instances_train_mini.json"
train_img_dir = r"E:\MOCS_dataset\instances_train_mini"
val_json_file = r"E:\MOCS_dataset\instances_train_mini.json"
val_img_dir = r"E:\MOCS_dataset\instances_train_mini"

train_dataloader = create_dataloader(train_json_file, train_img_dir, batch_size=16)
val_dataloader = create_dataloader(val_json_file,val_img_dir, batch_size=16, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model, train_dataloader, val_dataloader,num_epochs=30)

#model = get_fastrcnn_model(num_classes)

# 加载训练好的模型权重
#model.load_state_dict(torch.load(r"C:\Users\ZJ\model_epoch_5.pth", map_location=device))  # 请根据实际情况修改文件名
#model.to(device)

# 计算 mAP
# 这里假设 val_dataloader 已经定义并提供验证集
#mean_ap = compute_map(model, val_dataloader, num_classes, device)

