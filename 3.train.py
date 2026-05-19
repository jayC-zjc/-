# ========== train_3.py ==========
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
from module_3 import get_dataloaders, UNet, DiceLoss, train_one_epoch, validate
import swanlab

if __name__ == "__main__":
    # ------------------ 配置参数 ------------------
    config = {
        "img_dir": r"C:\Users\USER\Desktop\task3\iccv09Data\iccv09Data\images",
        "label_dir": r"C:\Users\USER\Desktop\task3\iccv09Data\iccv09Data",
        "batch_size": 8,
        "img_size": (256, 256),
        "lr": 1e-3,
        "epochs": 80,
        "loss_type": "combined",
        "num_classes": 8,
        "ignore_index": 255,
        "weight_decay": 1e-4,
        "step_size": 30,
        "gamma": 0.5,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    swanlab.init(
        project="stanford_seg",
        experiment_name=f"unet_{config['loss_type']}",
        config=config
    )

    # ------------------ 数据加载 ------------------
    train_loader, val_loader = get_dataloaders(
        config["img_dir"], config["label_dir"],
        batch_size=config["batch_size"], size=config["img_size"]
    )

    # ------------------ 模型、损失、优化器 ------------------
    model = UNet(n_channels=3, n_classes=config["num_classes"]).to(device)

    if config["loss_type"] == "ce":
        criterion = nn.CrossEntropyLoss(ignore_index=config["ignore_index"])
    elif config["loss_type"] == "dice":
        criterion = DiceLoss(ignore_index=config["ignore_index"])
    else:
        ce = nn.CrossEntropyLoss(ignore_index=config["ignore_index"])
        dice = DiceLoss(ignore_index=config["ignore_index"])
        
        class CombinedLoss(nn.Module):
            def __init__(self, ce, dice):
                super().__init__()
                self.ce = ce
                self.dice = dice
            def forward(self, pred, target):
                return self.ce(pred, target) + self.dice(pred, target)
        
        criterion = CombinedLoss(ce, dice)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # ------------------ 训练循环 ------------------
    best_miou = 0.0
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou = validate(model, val_loader, criterion, device, num_classes=config["num_classes"], ignore_index=config["ignore_index"])
        scheduler.step()

        swanlab.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mIoU": val_miou,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_unet.pth")
            print("  -> Saved best model")

    swanlab.finish()