import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n',default=5,type=int,help='分类的类别')
parser.add_argument('--dataset_path',default=r'C:\Users\win10\Desktop\ZJ\DWQ_dataset',type=str,help='整理好的数据集路径')
parser.add_argument('--weight_path', default='',type=str,help='权重路径')
parser.add_argument('--finetune',default=True, action='store_true', help='是否为迁移学习' )
parser.add_argument('--train', action='store_true', help='是否训练' )
parser.add_argument('--test', action='store_true', help='是否测试' )
parser.add_argument('--model', type=str, default='resnet101', choices=['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet201'],
                    help='选择模型')
opt = parser.parse_args()

# 设置超参数
n = opt.n
data_dir = opt.dataset_path

batch_size = 32

if opt.finetune:
    learning_rate = 0.001
    num_epochs = 5
else:
    learning_rate = 0.005
    num_epochs = 100

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                             shuffle=False if x =='test' else True, num_workers=0)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
def get_model(model_name):
    if model_name == "alexnet":
        return models.alexnet(pretrained=True)
    elif model_name == "vgg16":
        return models.vgg16(pretrained=True)
    elif model_name == "vgg19":
        return models.vgg19(pretrained=True)
    elif model_name == "resnet18":
        return models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        return models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        return models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        return models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=True)
    elif model_name == "densenet121":
        return models.densenet121(pretrained=True)
    elif model_name == "densenet169":
        return models.densenet169(pretrained=True)
    elif model_name == "densenet201":
        return models.densenet201(pretrained=True)
    else:
        raise ValueError("Unknown model name")

def modify_model(model, num_classes):
    if 'resnet' in opt.model:
        # ResNet 使用 fc 层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'alexnet' in opt.model or 'vgg' in opt.model:
        # AlexNet 和 VGG 使用 classifier 层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif 'densenet' in opt.model:
        # DenseNet 使用 classifier 层
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model type {opt.model} not supported for modification")


# model = models.resnet101(pretrained=True)
model = get_model(opt.model)

modify_model(model, num_classes=n)

'''
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, n)  # 替换最后的全连接层 for resnet
'''

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train_model(model, criterion, optimizer, num_epochs=num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 
            
            running_loss = 0.0
            running_corrects = 0
            

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    

    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloaders, weight_path):

    model.load_state_dict(torch.load(weight_path))

    model.eval()
    running_corrects = 0
    

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    
    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    print(f'Test Acc: {test_acc:.4f}')


if opt.train:
    model = train_model(model, criterion, optimizer, num_epochs)
    model_name = opt.model
    out_path = os.path.join(data_dir,f"best_{model_name}.pth")
    torch.save(model.state_dict(), out_path)

if opt.test:
    test_model (model, dataloaders, weight_path=opt.weight_path)
