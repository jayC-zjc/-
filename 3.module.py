# ========== module_3.py ==========
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# =====================================================
# 模块 1：数据加载 (Dataset & DataLoader)
# =====================================================

class JointTransform:
    """同时作用在图像和标签上的变换，保证几何变换一致"""
    def __init__(self, size=(256, 256), augment=False):
        self.size = size
        self.augment = augment

    def __call__(self, image, label):
        # 调整大小：图像用双线性插值，标签用最近邻插值
        image = transforms.functional.resize(image, self.size)
        label = transforms.functional.resize(label, self.size,
                                             interpolation=transforms.InterpolationMode.NEAREST)
        if self.augment:
            # 随机水平翻转
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.hflip(image)
                label = transforms.functional.hflip(label)

        image = transforms.ToTensor()(image)
        # 标签转为 LongTensor
        label = torch.from_numpy(np.array(label)).long()
        return image, label

class StanfordBackgroundDataset(Dataset):
    """Stanford Background 数据集加载类"""
    def __init__(self, img_dir, label_dir, file_list, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, name + '.jpg')
        # 标签文件夹改为 labels（你真实的文件夹名）
        label_path = os.path.join(self.label_dir, 'labels', name + '.regions.txt')

        image = Image.open(img_path).convert('RGB')
        label = np.loadtxt(label_path, dtype=np.int64)
        label[label == -1] = 255
        label = Image.fromarray(label.astype(np.uint8))

        if self.transform:
            image, label = self.transform(image, label)
        return image, label

def get_dataloaders(img_dir, label_dir, batch_size=8, num_workers=0, size=(256,256)):
    """划分数据集并返回训练和验证 DataLoader"""
    all_files = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]

    train_dataset = StanfordBackgroundDataset(
        img_dir, label_dir, train_files,
        transform=JointTransform(size, augment=True)
    )
    val_dataset = StanfordBackgroundDataset(
        img_dir, label_dir, val_files,
        transform=JointTransform(size, augment=False)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# =====================================================
# 模块 2：U-Net 网络结构 (unet)
# =====================================================

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """经典 U-Net 模型，输入 RGB 图像，输出 n_classes 通道的 logits"""
    def __init__(self, n_channels=3, n_classes=8):
        super().__init__()
        # 编码器 (下采样)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # 解码器 (上采样 + 拼接 + DoubleConv)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits

# =====================================================
# 模块 3：损失函数 (loss)
# =====================================================

class DiceLoss(nn.Module):
    """多类别 Dice Loss，对每个类别计算 Dice，取平均"""
    def __init__(self, smooth=1e-5, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        mask = (target != self.ignore_index).unsqueeze(1)
        target = target.clone()
        target[target == self.ignore_index] = 0
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
        target_one_hot = target_one_hot * mask

        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss.mean()

# =====================================================
# 模块 4：训练与评估工具 (train_utils)
# =====================================================

def compute_miou(pred, target, num_classes=8, ignore_index=255):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(torch.tensor(float('nan')))
        else:
            ious.append(intersection / union)
    ious = torch.tensor(ious)
    return ious[~ious.isnan()].mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples

def validate(model, loader, criterion, device, num_classes=8, ignore_index=255):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_miou = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            miou = compute_miou(outputs, labels, num_classes, ignore_index)
            total_miou += miou * images.size(0)
    return total_loss / total_samples, total_miou / total_samples