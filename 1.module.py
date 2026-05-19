from torchvision import transforms
from torchvision.datasets import Flowers102
import torchvision.models as models
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import conv3x3
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

# 模型初始化
def build_baseline_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)   # 加载预训练权重
    in_features = model.fc.in_features         # 512
    model.fc = nn.Linear(in_features, 102)     # 替换为新分类头
    return model

# 模型训练和验证函数
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if scaler is None:
        scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:          # GPU 上启用 AMP
            with torch.amp.autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:                           # CPU 上正常训练
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def get_dataloaders(batch_size=32, num_workers=4):
    # 训练集增强 + 归一化
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 验证集和测试集仅 resize + 中心裁剪
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = Flowers102(root="./data", split="train", transform=train_transform, download=True)
    val_dataset   = Flowers102(root="./data", split="val", transform=val_transform, download=True)
    test_dataset  = Flowers102(root="./data", split="test", transform=val_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return train_loader, val_loader, test_loader

# ---------------------引入注意力机制-----------------------

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 reduction=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SEBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SEBasicBlock")
        
        # 两个 3x3 卷积层（与 BasicBlock 完全一致）
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        # SE 注意力模块
        self.se = SELayer(planes, reduction)
        
        # 下采样层（shortcut 需要时使用）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)           # 插入 SE

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def build_se_resnet18(pretrained=True):
    # 用自定义 block 构建 ResNet-18 结构
    model = ResNet(SEBasicBlock, [2, 2, 2, 2])
    if pretrained:
        # 加载官方 ResNet-18 预训练权重
        pretrained_dict = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).state_dict()
        model_dict = model.state_dict()
        # 过滤掉 fc 层和 SE 模块新增的参数，严格匹配其余部分
        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model_dict and 'fc' not in k and 'se' not in k}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)  # 允许 SE 部分随机初始化
    # 替换最后的全连接层
    model.fc = nn.Linear(512, 102)
    return model
