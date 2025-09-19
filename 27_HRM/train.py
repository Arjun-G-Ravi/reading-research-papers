import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from hrm_cifar.model import HRMClassifier
from hrm_cifar.utils import accuracy_topk, seed_all, AverageMeter


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=1.0):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        acc1, acc5 = accuracy_topk(logits, targets, topk=(1, 5))
        loss_meter.update(loss.item(), imgs.size(0))
        top1_meter.update(acc1.item(), imgs.size(0))
        top5_meter.update(acc5.item(), imgs.size(0))

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{top1_meter.avg:.2f}", acc5=f"{top5_meter.avg:.2f}")

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    pbar = tqdm(loader, desc="Eval", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, targets)

        acc1, acc5 = accuracy_topk(logits, targets, topk=(1, 5))
        loss_meter.update(loss.item(), imgs.size(0))
        top1_meter.update(acc1.item(), imgs.size(0))
        top5_meter.update(acc5.item(), imgs.size(0))

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{top1_meter.avg:.2f}", acc5=f"{top5_meter.avg:.2f}")

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Simple HRM-style CIFAR-10 Trainer")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--macro-steps", type=int, default=3)
    parser.add_argument("--micro-steps", type=int, default=2)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--ctrl-dim", type=int, default=256)
    parser.add_argument("--mem-dim", type=int, default=256)
    parser.add_argument("--cnn-width", type=int, default=64)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = HRMClassifier(
        num_classes=10,
        token_dim=args.token_dim,
        ctrl_dim=args.ctrl_dim,
        mem_dim=args.mem_dim,
        macro_steps=args.macro_steps,
        micro_steps=args.micro_steps,
        cnn_width=args.cnn_width,
        cnn_depth=args.cnn_depth,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = None if args.no_amp or device.type != "cuda" else torch.cuda.amp.GradScaler()

    best_acc1 = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = Path(args.save_dir) / "hrm_cifar_best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc1, val_acc5 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"[E{epoch}] train loss {train_loss:.4f} acc@1 {train_acc1:.2f} | "
              f"val loss {val_loss:.4f} acc@1 {val_acc1:.2f}")

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_acc1": best_acc1,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved new best checkpoint to {ckpt_path} (acc@1={best_acc1:.2f})")

    print(f"Best val acc@1: {best_acc1:.2f}")


if __name__ == "__main__":
    main()