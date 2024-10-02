import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

class CustomCocoDataset(Dataset):
    def __init__(self, img_dir, coco_json, num_classes, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes

        # 读取 COCO 格式的 JSON 文件
        with open(coco_json) as f:
            self.coco_data = json.load(f)

        self.img_files = [img['file_name'] for img in self.coco_data['images']]
        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.annotations = {img['id']: [] for img in self.coco_data['images']}
        for ann in self.coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} does not exist.")

        img = Image.open(img_path).convert("RGB")

        img_id = self.coco_data['images'][idx]['id']
        target = self.parse_annotations(self.annotations[img_id])
        target['image_id'] = torch.tensor(img_id) 

        if self.transform:     
            img = self.transform(img)

        return img, target

    def parse_annotations(self, annotations):
        boxes = []
        labels = []
        for ann in annotations:
            x_min  = ann['bbox'][0]
            y_min  = ann['bbox'][1]
            width  = ann['bbox'][2]
            height = ann['bbox'][3]
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id'] - 1)  
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

def get_dataloader(img_dir, coco_json, num_classes, batch_size=4, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    dataset = CustomCocoDataset(img_dir, coco_json, num_classes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
    return dataloader


def get_faster_rcnn(num_classes):

    model = fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    weights = torch.load(r'E:\MOCS_dataset\model_0930.pth')
    weights = {k: v for k, v in weights.items() if 'roi_heads.box_predictor' not in k}
    model.load_state_dict(weights, strict=False)

    return model

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(dataloader, desc="Training", unit="batch"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(dataloader)

def test_with_mAP(model, test_loader, test_coco_json, device):
    print("Evaluating on test set...")
    coco_stats = evaluate_coco(model, test_loader, test_coco_json, device)
    print(f"Test mAP@0.5:0.95: {coco_stats[0]}")  # 打印测试集的 mAP@[0.5:0.95]

def evaluate_coco(model, dataloader, coco_json, device):
    model.eval()
    coco_gt = COCO(coco_json)  # Ground truth COCO annotations

    coco_results = []
    img_ids = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = [img.to(device) for img in images]

            outputs = model(images)
            for i, output in enumerate(outputs):
                img_id = targets[i].get('image_id', None).item()
                #if img_id is None:
                #    print(f"Warning: 'image_id' not found for index {i}. Using dataset image_id instead.")
                #    img_id = dataloader.dataset.image_ids[i]  # 确保存在 image_ids 列表

                img_ids.append(img_id)

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": label + 1,  # 类别从 1 开始
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score)
                    })


    # 使用 COCO 官方评估 API 进行 mAP 计算
    coco_dt = coco_gt.loadRes(coco_results)  # 预测的结果
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')  # iouType='bbox' 评估检测框
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats  #



def train_and_evaluate(model, train_loader, epochs=10, lr=0.001, save_path="model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss}")

    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

# argparse 管理参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'yolov8'],
                        help='Model name to use for training.')
    parser.add_argument('--train_img_dir', type=str, default=r"E:\MOCS_dataset\instances_train\instances_train", help='Path to the training image directory.')
    parser.add_argument('--train_coco_json', type=str, default=r"E:\MOCS_dataset\instances_train.json", help='Path to the COCO format JSON file for training.')
    parser.add_argument('--val_img_dir', type=str, default=r"E:\MOCS_dataset\instances_val\instances_val", help='Path to the validation image directory.')
    parser.add_argument('--val_coco_json', type=str, default=r"E:\MOCS_dataset\instances_val.json", help='Path to the COCO format JSON file for validation.')
    parser.add_argument('--test_img_dir', type=str, default=r"E:\MOCS_dataset\instances_test\instances_test", help='Path to the test image directory.')
    parser.add_argument('--test_coco_json', type=str, default=r"E:\MOCS_dataset\instances_test.json", help='Path to the COCO format JSON file for testing.')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of object classes.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--save_model_path', type=str, default="E:\MOCS_dataset\model_0930.pth", help='Path to save trained model weights.')

    args = parser.parse_args()

    # 数据加载
    train_loader = get_dataloader(args.train_img_dir, args.train_coco_json, args.num_classes, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.val_img_dir, args.val_coco_json, args.num_classes, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(args.test_img_dir, args.test_coco_json, args.num_classes, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_faster_rcnn(args.num_classes).to(device)

    # 训练与验证
    #train_and_evaluate(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, save_path=args.save_model_path)

    # 测试集评估 mAP
    test_with_mAP(model, test_loader, args.test_coco_json, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
