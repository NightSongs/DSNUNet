# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:44
@Author  : NDWX
@File    : train.py
@Software: PyCharm
"""
import glob
import random
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

from models.dsnunet import DSNUNet
from utils.data_process import cal_val_f1
from utils.opt_loader import load_opt, build_dataloader
from utils.trick import random_scale

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 加载模型
def load_model(DEVICE, deep_supervision=False, pretrain=False, pretrain_model_path=None):
    model = DSNUNet(rgb_init=32, sar_init=8, rgb_ch=3, sar_ch=2, out_ch=2, deep_supervision=deep_supervision)
    model.to(DEVICE)
    if pretrain:
        model.load_state_dict(torch.load(pretrain_model_path))
    return model


# 训练函数
def train(num_epochs, optimizer, scheduler, loss_fn, train_loader, valid_loader, model, save_path,
          multi_scale=False, deep_supervision=False):
    epochs = num_epochs + 1
    header = r'Epoch/EpochNum | TrainLoss | ValidF1 | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.3f} | {:9.3f} | {:9.2f}'
    print(header)
    # 记录当前验证集最优mIoU,以判定是否保存当前模型
    best_f1 = 0
    best_f1_epoch = 0
    train_loss_epochs, val_f1_epochs, lr_epochs = [], [], []
    for epoch in range(1, epochs):
        model.train()
        losses = []
        start_time = time.time()
        for batch_index, (x1, x2, y) in enumerate(tqdm(train_loader)):
            accumulation_steps = 16 / x1.shape[0]
            x1, x2, y = x1.float(), x2.float(), y.long()
            if multi_scale:
                scale = random.uniform(0.7, 1.3)
                x1, x2, y = random_scale(x1, x2, y, x1.shape[2:], (scale, scale))
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            if deep_supervision:
                loss = 0
                outputs = model(x1, x2)
                for output in outputs:
                    loss += loss_fn(output, y)
                loss /= len(outputs)
            else:
                output = model(x1, x2)
                loss = loss_fn(output, y)
            loss = loss / accumulation_steps
            loss.backward()
            if ((batch_index + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
        scheduler.step()
        val_f1 = cal_val_f1(model, valid_loader, deep_supervision=deep_supervision)
        train_loss_epochs.append(np.array(losses).mean())
        val_f1_epochs.append(np.mean(val_f1))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        print(raw_line.format(epoch, num_epochs, np.array(losses).mean(),
                              np.mean(val_f1),
                              (time.time() - start_time) / 60 ** 1), end="")
        if best_f1 < val_f1:
            best_f1 = val_f1
            best_f1_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print("  valid F1 is improved. the model is saved.")
        else:
            print("")
    return train_loss_epochs, val_f1_epochs, lr_epochs


if __name__ == '__main__':
    random_seed = 1021
    num_epochs = 100
    batch_size = 8
    channels = 5
    lr = 1e-3
    multi_scale = False
    use_style_transfer = False
    use_k_fold = False
    deep_supervision = False
    setup_seed(random_seed)
    train_dataset = [sorted(glob.glob("../data/train/A/*.tif")), sorted(glob.glob("../data/train/B/*.tif")),
                     sorted(glob.glob("../data/train/OUT/*.tif"))]
    val_dataset = [sorted(glob.glob("../data/val/A/*.tif")), sorted(glob.glob("../data/val/B/*.tif")),
                   sorted(glob.glob("../data/val/OUT/*.tif"))]
    model_save_path = "../user_data/model_data/change_detection.pth"
    train_loader, valid_loader = build_dataloader(train_dataset, val_dataset, int(batch_size), use_style_transfer)
    model = load_model(DEVICE, deep_supervision=deep_supervision)
    optimizer, scheduler, loss_fn = load_opt(model, lr)
    train_loss_epochs, val_mIoU_epochs, lr_epochs = train(num_epochs, optimizer, scheduler, loss_fn,
                                                          train_loader, valid_loader, model, model_save_path,
                                                          multi_scale=multi_scale, deep_supervision=deep_supervision)
