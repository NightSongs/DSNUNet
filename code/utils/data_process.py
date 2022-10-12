# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:15
@Author  : NDWX
@File    : dataProcess.py
@Software: PyCharm
"""
import cv2
import gdal
import numpy as np
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#  读取图像像素矩阵
def imgread(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    data[np.isnan(data)] = 0
    if (len(data.shape) == 3):
        nir, r, g, vv, vh = data[0], data[1], data[2], data[4], data[5]
        data = np.stack((nir, r, g, vv * 100, vh * 100), axis=0) / 10000.0
        # data = np.stack((nir, r, g), axis=0) / 10000.0 # RGB baseline
        data = data.swapaxes(1, 0).swapaxes(1, 2)  # (C,H,W)->(H,W,C)
    return data


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集Iou
def cal_val_iou(model, loader, deep_supervision=False):
    val_iou = []
    model.eval()
    for imageA, imageB, target in loader:
        imageA, imageB, target = imageA.float(), imageB.float(), target.long()
        imageA, imageB, target = imageA.to(DEVICE), imageB.to(DEVICE), target.to(DEVICE)
        if deep_supervision:
            output = model(imageA, imageB)[-1]
            output = output.argmax(1)
        else:
            output = model(imageA, imageB)
            output = output.argmax(1)
        iou = cal_iou(output, target)
        val_iou.append(iou)
    return val_iou


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集f1
def cal_val_f1(model, loader, deep_supervision=False):
    TP_sum, FN_sum, FP_sum = [], [], []
    model.eval()
    for imageA, imageB, target in loader:
        imageA, imageB, target = imageA.float(), imageB.float(), target.long()
        imageA, imageB, target = imageA.to(DEVICE), imageB.to(DEVICE), target.to(DEVICE)
        if deep_supervision:
            output = model(imageA, imageB)[-1]
            output = output.argmax(1)
        else:
            output = model(imageA, imageB)
            output = output.argmax(1)
        TP, FN, FP = cal_f1(output, target)
        TP_sum.append(TP)
        FN_sum.append(FN)
        FP_sum.append(FP)

    p = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FP_sum) + 0.000001)
    r = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FN_sum) + 0.000001)
    val_f1 = 2 * r * p / (r + p + 0.000001)
    return val_f1


# 计算IoU
def cal_iou(pred, mask, c=2):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p * t).sum()
        #  0.0001防止除零
        iou = 2 * overlap / (uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)


# 计算F1-Score
def cal_f1(pred, mask):
    TP = ((pred == 1) & (mask == 1)).cpu().sum()
    FN = ((pred == 0) & (mask == 1)).cpu().sum()
    FP = ((pred == 1) & (mask == 0)).cpu().sum()
    return TP, FN, FP


# opencv数据转gdal
def OpencvData2GdalData(OpencvImg_data):
    # 若为二维，格式相同
    if (len(OpencvImg_data.shape) == 2):
        GdalImg_data = OpencvImg_data
    else:
        if 'int8' in OpencvImg_data.dtype.name:
            GdalImg_data = np.zeros((OpencvImg_data.shape[2], OpencvImg_data.shape[0], OpencvImg_data.shape[1]),
                                    np.uint8)
        elif 'int16' in OpencvImg_data.dtype.name:
            GdalImg_data = np.zeros((OpencvImg_data.shape[2], OpencvImg_data.shape[0], OpencvImg_data.shape[1]),
                                    np.uint16)
        else:
            GdalImg_data = np.zeros((OpencvImg_data.shape[2], OpencvImg_data.shape[0], OpencvImg_data.shape[1]),
                                    np.float32)
        for i in range(OpencvImg_data.shape[2]):
            # 注意，opencv为BGR
            data = OpencvImg_data[:, :, OpencvImg_data.shape[2] - i - 1]
            data = np.reshape(data, (OpencvImg_data.shape[0], OpencvImg_data.shape[1]))
            GdalImg_data[i] = data
    return GdalImg_data


# gdal数据转opencv
def GdalData2OpencvData(GdalImg_data):
    if 'int8' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint8)
    elif 'int16' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint16)
    else:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.float32)
    for i in range(GdalImg_data.shape[0]):
        OpencvImg_data[:, :, i] = GdalImg_data[GdalImg_data.shape[0] - i - 1, :, :]
    return OpencvImg_data


#  随机旋转(有一定的信息丢失，但没有缩放操作)
def rotate(imageA, imageB, label, angle, center=None, scale=1.0):
    imageA = GdalData2OpencvData(imageA.swapaxes(2, 1).swapaxes(0, 1))
    imageB = GdalData2OpencvData(imageB.swapaxes(2, 1).swapaxes(0, 1))
    (h, w) = imageA.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    resultA = cv2.warpAffine(imageA, M, (w, h))
    resultB = cv2.warpAffine(imageB, M, (w, h))
    resultLabel = cv2.warpAffine(label, M, (w, h))
    resultA = OpencvData2GdalData(resultA).swapaxes(1, 0).swapaxes(1, 2)
    resultB = OpencvData2GdalData(resultB).swapaxes(1, 0).swapaxes(1, 2)
    return resultA, resultB, resultLabel


#  对角镜像
def dia(imageA, imageB, label):
    imageA = np.flip(np.flip(imageA, axis=0), axis=1)
    imageB = np.flip(np.flip(imageB, axis=0), axis=1)
    label = np.flip(np.flip(label, axis=0), axis=1)
    return imageA, imageB, label


#  水平镜像
def hor(imageA, imageB, label):
    imageA = np.flip(imageA, axis=1)
    imageB = np.flip(imageB, axis=1)
    label = np.flip(label, axis=1)
    return imageA, imageB, label


#  垂直镜像
def ver(imageA, imageB, label):
    imageA = np.flip(imageA, axis=0)
    imageB = np.flip(imageB, axis=0)
    label = np.flip(label, axis=0)
    return imageA, imageB, label
