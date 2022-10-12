# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:33
@Author  : NDWX
@File    : opt_loader.py
@Software: PyCharm
"""
import torch
from pytorch_toolbelt import losses as L
from .dataset import get_dataloader
from change_detection_pytorch.losses import FocalLoss, DiceLoss


# 定义优化器及损失函数
def load_opt(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=int(1e-5))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=0.1)
    DiceLoss_fn = DiceLoss(mode='multiclass')
    FocalLoss_fn = FocalLoss(mode='multiclass', gamma=2, alpha=0.25)
    loss_fn = L.JointLoss(first=DiceLoss_fn, second=FocalLoss_fn,
                          first_weight=0.5, second_weight=0.5).cuda()
    return optimizer, scheduler, loss_fn


# 生成dataloader
def build_dataloader(train_path, val_path, batch_size, use_style_transfer):
    train_loader = get_dataloader(train_path[0], train_path[1], train_path[2],
                                  "train", batch_size, shuffle=True, num_workers=8, drop_last=True,
                                  use_style_transfer=use_style_transfer)
    valid_loader = get_dataloader(val_path[0], val_path[1], val_path[2],
                                  "val", batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  use_style_transfer=use_style_transfer)
    return train_loader, valid_loader


# 生成测试dataloader
def build_test_dataloader(test_path, batch_size):
    test_dataloader = get_dataloader(test_path[0], test_path[1], test_path[2],
                                     "test", batch_size, shuffle=False, num_workers=8, drop_last=False,
                                     use_style_transfer=False)
    return test_dataloader

