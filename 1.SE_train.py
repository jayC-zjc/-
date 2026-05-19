import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
from torchvision import transforms
from torchvision.datasets import Flowers102
import torchvision.models as models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import wandb

import module

# 超参数配置
config = {
    "optimizer": "Adam", # 可选 "Adam" 或 "SGD"
    "fc_lr": 1e-4,
    "backbone_lr": 1e-4,
    "epochs_head": 5,
    "epochs_ft": 20,
    "batch_size": 32,
    "model": "se_resnet18"
}
config["run_name"] = f"SE_{config['optimizer']}_fc{config['fc_lr']}_bb{config['backbone_lr']}"
config["save_dir"] = f"outputs/{config['run_name']}"

def run_se_resnet18(config, run_name=None):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    config = config.copy()
    if run_name is not None:
        config["run_name"] = run_name
    elif "run_name" not in config:
        config["run_name"] = f"SE_{config['optimizer']}_fc{config['fc_lr']}_bb{config['backbone_lr']}"
    config["save_dir"] = f"outputs/{config['run_name']}"

    wandb.init(project="flower102", name=config["run_name"])
    wandb.config.update(config)

    # 创建保存路径
    os.makedirs(config["save_dir"], exist_ok=True)
    best_path = os.path.join(config["save_dir"], "best.pth")
    best_val_acc = -float("inf")
    best_epoch = -1
    best_train_loss = None
    best_train_acc = None
    best_val_loss = None
    head_train_losses = []
    head_val_losses = []
    head_epoch_indices = []
    head_train_accs = []
    head_val_accs = []
    ft_train_losses = []
    ft_val_losses = []
    ft_epoch_indices = []
    ft_train_accs = []
    ft_val_accs = []

    # 模型与数据集合加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    train_loader, val_loader, test_loader = module.get_dataloaders(batch_size=config["batch_size"])
    model = module.build_se_resnet18(pretrained=True).to(device)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 仅解冻 fc 层
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=config["fc_lr"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=config["fc_lr"], momentum=0.9)
    else:
        raise ValueError("只能选择 Adam 或 SGD 作为优化器")

    epochs_head = config["epochs_head"]
    for epoch in range(epochs_head):
        train_loss, train_acc = module.train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = module.validate(model, val_loader, criterion, device)
        epoch_idx = epoch + 1
        head_epoch_indices.append(epoch_idx)
        head_train_losses.append(train_loss)
        head_val_losses.append(val_loss)
        head_train_accs.append(train_acc)
        head_val_accs.append(val_acc)
        print(
            f"Head Epoch {epoch_idx}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "phase": "head",
            "epoch": epoch_idx
        }, step=epoch_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = epoch_idx
            torch.save({
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "state_dict": model.state_dict()
            }, best_path)
            print(f"Saved best model (val_acc={best_val_acc:.4f}) to {best_path}")

    # 解冻全部
    for param in model.parameters():
        param.requires_grad = True

    # 差分学习率：fc 层稍大，backbone 很小
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': config["fc_lr"]},
        {'params': (p for n, p in model.named_parameters() if 'fc' not in n), 'lr': config["backbone_lr"]}
        ])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD([
        {'params': model.fc.parameters(), 'lr': config["fc_lr"]},
        {'params': (p for n, p in model.named_parameters() if 'fc' not in n), 'lr': config["backbone_lr"]}
        ], momentum=0.9)

    for epoch in range(config["epochs_ft"]):
        train_loss, train_acc = module.train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = module.validate(model, val_loader, criterion, device)

        epoch_idx = epochs_head + epoch + 1
        ft_epoch_indices.append(epoch_idx)
        ft_train_losses.append(train_loss)
        ft_val_losses.append(val_loss)
        ft_train_accs.append(train_acc)
        ft_val_accs.append(val_acc)
        print(
            f"FT Epoch {epoch_idx}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "phase": "ft",
            "epoch": epoch_idx
        }, step=epoch_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = epoch_idx
            torch.save({
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "state_dict": model.state_dict()
            }, best_path)
            print(f"Saved best model (val_acc={best_val_acc:.4f}) to {best_path}")

    # 训练结束：加载最优模型 -> 测试集评估 -> 把测试结果一并保存到 best.pth
    test_loss = None
    test_acc = None
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        test_loss, test_acc = module.validate(model, test_loader, criterion, device)
        checkpoint.update({
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        torch.save(checkpoint, best_path)
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        print(f"Best model test results saved: Test Acc={test_acc:.4f}")
    else:
        print("Warning: best model not found, skip test evaluation.")

    wandb.finish()

    return {
        "run_name": config["run_name"],
        "optimizer": config["optimizer"],
        "batch_size": config["batch_size"],
        "fc_lr": config["fc_lr"],
        "backbone_lr": config["backbone_lr"],
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "train_acc": best_train_acc,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    }


if __name__ == "__main__":
    run_se_resnet18(config=config)