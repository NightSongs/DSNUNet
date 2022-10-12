# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:21
@Author  : NDWX
@File    : dataset.py
@Software: PyCharm
"""
import random
import torch.utils.data as D
from torchvision import transforms as T

from .data_process import *
from .trick import style_transfer


#  随机数据增强
def DataAugmentation(imageA, imageB, label, mode):
    augmentationList = ["hor", "ver", "dia", "rotate"]
    if (mode == "train"):
        aug = random.choice(augmentationList)
        if aug == "hor":
            #  图像水平翻转
            imageA, imageB, label = hor(imageA, imageB, label)
        elif aug == "ver":
            #  图像垂直翻转
            imageA, imageB, label = ver(imageA, imageB, label)
        elif aug == "dia":
            #  图像对角翻转
            imageA, imageB, label = dia(imageA, imageB, label)
        elif aug == "rotate":
            #  RandomFixRotate
            angle = random.choice([90, -90, 270, -270])
            imageA, imageB, label = rotate(imageA, imageB, label, angle)
    if (mode == "val"):
        pass
    imageA = np.ascontiguousarray(imageA)
    imageB = np.ascontiguousarray(imageB)
    label = np.ascontiguousarray(label)
    return imageA, imageB, label


# 构建dataset
class change_dataset(D.Dataset):
    def __init__(self, image_A_paths, image_B_paths, label_paths, mode, use_style_transfer):
        self.image_A_paths = image_A_paths
        self.image_B_paths = image_B_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_A_paths)
        assert len(image_A_paths) == len(image_B_paths), '前后时相影像数量不匹配'
        self.use_style_transfer = use_style_transfer
        self.as_tensor = T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        imageA = imgread(self.image_A_paths[index])
        imageB = imgread(self.image_B_paths[index])
        if self.mode == "train":
            if self.use_style_transfer:
                if np.random.random() < 0.5:
                    imageA = style_transfer(imageA, imageB)
                else:
                    imageB = style_transfer(imageB, imageA)
            label = cv2.imread(self.label_paths[index], -1) / 255
            imageA, imageB, label = DataAugmentation(imageA, imageB, label, self.mode)
            return self.as_tensor(imageA), self.as_tensor(imageB), label
        elif self.mode == "val":
            label = cv2.imread(self.label_paths[index], -1) / 255
            imageA, imageB, label = DataAugmentation(imageA, imageB, label, self.mode)
            return self.as_tensor(imageA), self.as_tensor(imageB), label
        elif self.mode == "test":
            if not self.label_paths is None:
                label = cv2.imread(self.label_paths[index], -1) / 255
                image_A_array = np.ascontiguousarray(imageA)
                image_B_array = np.ascontiguousarray(imageB)
                return self.as_tensor(image_A_array), self.as_tensor(image_B_array), self.image_A_paths[index], label
            else:
                image_A_array = np.ascontiguousarray(imageA)
                image_B_array = np.ascontiguousarray(imageB)
                return self.as_tensor(image_A_array), self.as_tensor(image_B_array), self.image_A_paths[index]

    def __len__(self):
        return self.len


# 构建数据加载器
def get_dataloader(image_A_paths, image_B_paths, label_paths, mode, batch_size,
                   shuffle, num_workers, drop_last, use_style_transfer):
    dataset = change_dataset(image_A_paths, image_B_paths, label_paths, mode, use_style_transfer)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader
