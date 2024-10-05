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
import os
import json
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class MOCSDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.image_id_to_annotations = self._create_image_id_to_annotations()
        print(f"Loaded {len(self.annotations['images'])} images and {len(self.annotations['annotations'])} annotations.")

    def _create_image_id_to_annotations(self):
        image_id_to_annotations = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in image_id_to_annotations:
                image_id_to_annotations[ann['image_id']] = []
            image_id_to_annotations[ann['image_id']].append(ann)
        return image_id_to_annotations

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        boxes = []
        labels = []
        if img_info['id'] in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[img_info['id']]:
                x, y, width, height = ann['bbox']
                if width > 0 and height > 0:
                    boxes.append([x, y, x + width, y + height])  # Convert to [x_min, y_min, x_max, y_max]
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



num_classes = 14  # Background + 13 classes

test_json_path = r"E:\MOCS_dataset\instances_train_mini.json"
test_img_dir = r"E:\MOCS_dataset\instances_train_mini"

def get_fastrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

model = get_fastrcnn_model(num_classes)

custom_weights_path = None

def load_custom_weights(model, weights_path):
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded custom weights from {weights_path}")
    return model

model = load_custom_weights(model, custom_weights_path)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)

def evaluate_model(model, test_img_dir, test_json_path):
    model.eval()
    test_images = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if img.endswith('.jpg') or img.endswith('.png')]
    
    coco = COCO(test_json_path)
    coco_results = []
    
    for img_path in test_images:
        image_id = int(os.path.splitext(os.path.basename(img_path))[0])
        image = load_image(img_path)
        
        # 检查图像是否正确加载
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        
        with torch.no_grad():
            prediction = model([image])
        
        # 检查模型的预测输出
        if len(prediction[0]['boxes']) == 0:
            print(f"No predictions for image {img_path}")
            continue
        
        for i, box in enumerate(prediction[0]['boxes']):
            score = prediction[0]['scores'][i].item()
            category_id = prediction[0]['labels'][i].item()
            bbox = box.tolist()
            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "score": score
            })
    
    if len(coco_results) == 0:
        print("No results to save.")
        return
    
    with open("coco_results.json", "w") as f:
        json.dump(coco_results, f, indent=4)
    
    coco_dt = coco.loadRes("coco_results.json")
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

evaluate_model(model, test_img_dir, test_json_path)

