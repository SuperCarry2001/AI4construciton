## 从零开始 ##
##  只用作训练 ##

import json
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pycocotools.cocoeval import COCOeval

class customDataset(Dataset):
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

# 通过可视化的方法检查数据集是否正确加载
def visualize_dataloader(dataloader):
    images, boxes, labels = next(iter(dataloader))

    for i in range(min(3, len(images))):
        image = images[i].permute(1, 2, 0).cpu().numpy()  
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for j, box in enumerate(boxes[i]):
            box = box.cpu().numpy()
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            label = labels[i][j].item()
            ax.text(x_min, y_min, f"ID: {i}, Class: {label}", color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        plt.title(f"Image ID: {i}")
        plt.show()

def create_dataloader(json_file, img_dir, batch_size=16, shuffle=True):
    def collate_fn(batch):
        images, boxes, labels = zip(*batch)
        return torch.stack(images, 0), boxes, labels

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    dataset = MOCSDataset(json_file, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

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
            original_size = image.size
            image = self.transform(image)
            new_size = image.size()[1:]  # (C, H, W) -> (H, W)
            scale_x = new_size[1] / original_size[0]
            scale_y = new_size[0] / original_size[1]
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])

        return image, boxes, labels

num_classes = 14  # Background + 13 classes
train_json_file = r"E:\MOCS_dataset\instances_train_mini.json"
train_img_dir = r"E:\MOCS_dataset\instances_train_mini"
val_json_file = r"E:\MOCS_dataset\instances_val_mini.json"
val_img_dir = r"E:\MOCS_dataset\instances_val_mini"
#val_json_file = r"E:\MOCS_dataset\instances_train.json"
#val_img_dir = r"E:\MOCS_dataset\instances_train\instances_train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = create_dataloader(train_json_file, train_img_dir, batch_size=16)
val_dataloader = create_dataloader(val_json_file, val_img_dir, batch_size=16)


# train
def get_fastrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=0.03):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_map = 0.0
    best_weights = None

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

        # calculate mAP
        model.eval()
        coco_gt = val_dataloader.dataset.coco
        coco_dt = []
        with torch.no_grad():
            for images, _, _ in val_dataloader:
                images = [image.to(device) for image in images]
                outputs = model(images)
                for i, output in enumerate(outputs):
                    image_id = val_dataloader.dataset.ids[i]
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        coco_dt.append({
                            "image_id": image_id,
                            "category_id": label,
                            "bbox": box.tolist(),
                            "score": score
                        })

        coco_dt = coco_gt.loadRes(coco_dt)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_score = coco_eval.stats[0]  # mAP

        if map_score > best_map:
            best_map = map_score
            best_weights = model.state_dict()

    if best_weights:
        torch.save(best_weights, 'best_model_weights.pth')
        print(f"Best model weights saved with mAP: {best_map:.4f}")

model = get_fastrcnn_model(num_classes)   
train_model(model, train_dataloader, val_dataloader,num_epochs=30)
        


       
