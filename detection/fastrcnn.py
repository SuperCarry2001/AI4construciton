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
from pycocotools.coco import COCO
import torchvision.transforms as T
from ultralytics import YOLO



def visualize_dataloader(dataloader, num_images=5):

    for i, (images, boxes, labels, img_ids, cat_ids, bbox_ids) in enumerate(dataloader):
        if i >= num_images:
            break

        for j in range(min(5,len(images))):
            fig, ax = plt.subplots(1)
            img = images[j].permute(1, 2, 0).numpy()  # 将图像从 (C, H, W) 转换为 (H, W, C)
            ax.imshow(img)

            for box, label, img_id, cat_id, bbox_id in zip(boxes[j], labels[j], img_ids[j], cat_ids[j], bbox_ids[j]):
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'(bbox)id: {bbox_id}\nCategory_id: {cat_id}\nimg_id: {img_id}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

            plt.title(f'Image ID: {img_ids[j]}')
            plt.show()


def visualize_predictions(model, dataloader, device, num_images=5, score_threshold=0.5):
    model.eval()
    transform = T.ToPILImage()
    with torch.no_grad():
        for i, (images, _, _, img_ids, _, bbox_ids) in enumerate(dataloader):
            if i >= num_images:
                break
            images = [image.to(device) for image in images]
            outputs = model(images)
            for j, output in enumerate(outputs):
                image = transform(images[j].cpu())
                plt.imshow(image)
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if score >= score_threshold:
                        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=2))
                        plt.text(box[0], box[1], f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
                plt.show()


def create_dataloader(json_file, img_dir, batch_size=16, shuffle=True):
    def collate_fn(batch):
        images, boxes, labels, img_ids, cat_ids, bbox_ids = zip(*batch)
        return torch.stack(images, 0), boxes, labels, img_ids, cat_ids, bbox_ids

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    dataset = MOCSDataset(json_file, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

class MOCSDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.coco = COCO(json_file)
        with open(json_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = [img['id'] for img in self.annotations['images']]  # 提取所有图像的 ID

        print(f"Loaded {len(self.annotations['images'])} images and {len(self.annotations['annotations'])} annotations.")

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        boxes = []
        labels = []
        img_ids = []
        cat_ids = []
        bbox_ids = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                x, y, width, height = ann['bbox']
                # 仅保留有效的边界框
                if width > 0 and height > 0:
                    boxes.append([x, y, x + width, y + height])  # 转换为 [x_min, y_min, x_max, y_max]
                    labels.append(ann['category_id'])
                    img_ids.append(ann['image_id'])
                    cat_ids.append(ann['category_id'])
                    bbox_ids.append(ann['id'])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        img_ids = torch.tensor(img_ids, dtype=torch.int64) if img_ids else torch.empty((0,), dtype=torch.int64)
        cat_ids = torch.tensor(cat_ids, dtype=torch.int64) if cat_ids else torch.empty((0,), dtype=torch.int64)
        bbox_ids = torch.tensor(bbox_ids, dtype=torch.int64) if bbox_ids else torch.empty((0,), dtype=torch.int64)

        if self.transform:
            original_size = image.size
            image = self.transform(image)
            new_size = image.size()[1:]  # (C, H, W) -> (H, W)
            scale_x = new_size[1] / original_size[0]
            scale_y = new_size[0] / original_size[1]
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])

        return image, boxes, labels, img_ids, cat_ids, bbox_ids

    def get_image_id(self, index):
        return self.ids[index]


def test(model, val_json_file, val_img_dir, batch_size=8):
    model.eval()
    val_dataloader = create_dataloader(val_json_file, val_img_dir, batch_size=batch_size, shuffle=False)
    coco_gt = val_dataloader.dataset.coco
    coco_dt = []
    
    with torch.no_grad():
        for images, _, _, img_ids, _, bbox_ids in val_dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                image_id_list = img_ids[i].cpu().numpy().tolist()  # 获取图像 ID 列表
                bbox_id_list = bbox_ids[i].cpu().numpy().tolist()  # 获取边界框 ID 列表
                
                # 遍历每个边界框及其对应的图像 ID 和边界框 ID
                for box, score, label, bbox_id in zip(boxes, scores, labels, bbox_id_list):
                    for image_id in image_id_list:
                        coco_dt.append({
                            "image_id": image_id,
                            "category_id": label,
                            "bbox": box.tolist(),
                            "score": score,
                            "bbox_id": bbox_id  # 使用对应的 bbox_id
                        })

        #visualize_predictions(model, val_dataloader, device, num_images=5, score_threshold=0.3)

        coco_dt = coco_gt.loadRes(coco_dt)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_score = coco_eval.stats[0]  # mAP
    print(f"mAP: {map_score:.4f}")
    return map_score
    

num_classes = 14  # Background + 13 classes
train_json_file = r"E:\MOCS_dataset\chosen_train_mini.json"
train_img_dir = r"E:\MOCS_dataset\chosen_train_mini"
val_json_file = train_json_file
val_img_dir = train_img_dir
#train_json_file = r"E:\MOCS_dataset\instances_train.json"
#train_img_dir = r"E:\MOCS_dataset\instances_train\instances_train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = create_dataloader(train_json_file, train_img_dir, batch_size=8,shuffle=True)
val_dataloader = create_dataloader(val_json_file, val_img_dir, batch_size=8,shuffle=False)

#visualize_dataloader(val_dataloader)

# train
def get_fastrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def train_model(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=0.003):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    best_map = 0.0
    best_weights = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, boxes, labels, img_ids, cat_ids, bbox_ids in tqdm(train_dataloader):
            images = [image.to(device) for image in images]
            targets = []
            for box, label, img_id, cat_id, bbox_id in zip(boxes, labels, img_ids, cat_ids, bbox_ids):
                for b, l, i, c, b_id in zip(box, label, img_id, cat_id, bbox_id):
                    targets.append({
                        "boxes": b.unsqueeze(0).to(device),
                        "labels": l.unsqueeze(0).to(device),
                        "image_id": i.unsqueeze(0).to(device),
                        "category_id": c.unsqueeze(0).to(device),
                        "bbox_id": b_id.unsqueeze(0).to(device)
                    })

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        lr_scheduler.step()

        # calculate mAP
        model.eval()
        coco_gt = val_dataloader.dataset.coco
        coco_dt = []
        '''
        with torch.no_grad():
            for images, _, _, img_ids, _, bbox_ids in val_dataloader:
                images = [image.to(device) for image in images]
                outputs = model(images)
                for i, output in enumerate(outputs):

                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    bbox_id = bbox_ids[i].item() # 获取 bbox_id
                    for box, score, label in zip(boxes, scores, labels):
                        coco_dt.append({
                            "image_id": img_ids[i].item(),  # 仍然保留 image_id
                            "category_id": label,
                            "bbox": box.tolist(),
                            "score": score,
                            "bbox_id": bbox_id  # 添加 bbox_id
                        })
        '''

        with torch.no_grad():
            for images, _, _, img_ids, _, bbox_ids in val_dataloader:
                images = [image.to(device) for image in images]
                outputs = model(images)
                for i, output in enumerate(outputs):
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    image_id_list = img_ids[i].cpu().numpy().tolist()  # 获取图像 ID 列表
                    bbox_id_list = bbox_ids[i].cpu().numpy().tolist()  # 获取边界框 ID 列表
                    
                    # 遍历每个边界框及其对应的图像 ID 和边界框 ID
                    for box, score, label, bbox_id in zip(boxes, scores, labels, bbox_id_list):
                        for image_id in image_id_list:
                            coco_dt.append({
                                "image_id": image_id,
                                "category_id": label,
                                "bbox": box.tolist(),
                                "score": score,
                                "bbox_id": bbox_id  # 使用对应的 bbox_id
                            })

        #visualize_predictions(model, val_dataloader, device, num_images=5, score_threshold=0.3)

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
        torch.save(best_weights, r'C:\Users\ZJ\desktop\best_model_weights1008.pth')
        print(f"Best model weights saved with mAP: {best_map:.4f}")

model = get_fastrcnn_model(num_classes)
model = YOLO(r'C:\Users\ZJ\Desktop\AI4construciton\detection\yolov8_original.pt')
model.to(device)  
#test(model,val_json_file, val_img_dir, batch_size=8)

train_model(model, train_dataloader, val_dataloader, num_epochs=30,learning_rate=0.003)

        


       
