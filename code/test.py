# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/2/002 10:56
@Author  : NDWX
@File    : test.py
@Software: PyCharm
"""
import glob
import os
import random

from models.unet.dsnunet import DSNUNet
import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.opt_loader import build_test_dataloader

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
def load_model(DEVICE, classes, deep_supervision, model_path):
    model = DSNUNet(rgb_init=32, sar_init=8, rgb_ch=3, sar_ch=2, out_ch=classes, deep_supervision=deep_supervision)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    return model


# 预测并计算验证集F1
def predict(test_dataloader, model, output_path, use_TTA=False, deep_supervision=False):
    TP_sum, FN_sum, FP_sum = [], [], []
    with torch.no_grad():
        for x1, x2, image_A_path, y in tqdm(test_dataloader):
            if use_TTA:
                from utils.trick import TTA
                output = TTA(x1, x2, model, deep_supervision=deep_supervision)
            else:
                x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
                if deep_supervision:
                    output = model(x1, x2)[-1].cpu().data.numpy()
                else:
                    output = model(x1, x2).cpu().data.numpy()
            output_ = output.argmax(1)
            y = y.data.numpy()
            TP = ((output_ == 1) & (y == 1)).sum()
            FN = ((output_ == 0) & (y == 1)).sum()
            FP = ((output_ == 1) & (y == 0)).sum()
            TP_sum.append(TP)
            FN_sum.append(FN)
            FP_sum.append(FP)
            for i in range(output.shape[0]):
                pred = output[i].argmax(0) * 255
                pred = pred.squeeze()
                save_path = os.path.join(output_path, os.path.split(image_A_path[i])[1])
                cv2.imwrite(save_path, pred.astype(np.uint8))
        p = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FP_sum) + 0.000001)
        r = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FN_sum) + 0.000001)
        val_f1 = 2 * r * p / (r + p + 0.000001)
        return val_f1, p, r


if __name__ == '__main__':
    setup_seed(1021)
    batch_size = 8
    channels = 5
    use_TTA = True
    deep_supervision = False
    output_dir = "../user_data/res"
    test_dataset = [sorted(glob.glob("../data/test/A/*.tif")), sorted(glob.glob("../data/test/B/*.tif")),
                    sorted(glob.glob("../data/test/OUT/*.tif"))]
    model = load_model(DEVICE, 2, deep_supervision,
                       model_path="../user_data/model_data/change_detection_(32_8).pth")
    test_dataloader = build_test_dataloader(test_dataset, batch_size)
    val_f1, p, r = predict(test_dataloader, model, output_dir, use_TTA, deep_supervision=deep_supervision)
    print("test precision:", round(np.mean(p) * 100, 2))
    print("test recall:", round(np.mean(r) * 100, 2))
    print("test F1-Score:", round(np.mean(val_f1) * 100, 2))
