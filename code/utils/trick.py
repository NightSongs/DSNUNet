# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:24
@Author  : NDWX
@File    : trick.py
@Software: PyCharm
"""
import torch
import numpy as np
from sklearn.model_selection import KFold
import torchvision.transforms.functional as vF
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


#  K折训练
def k_fold(dataset, random_seed, num=5):
    x1_path, x2_path, y_path = np.array(dataset[0]), np.array(dataset[1]), np.array(dataset[2])
    folds = KFold(n_splits=num, shuffle=True, random_state=random_seed).split(range(len(x1_path)),
                                                                              range(len(y_path)))
    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_dataset = [x1_path[trn_idx], x2_path[trn_idx], y_path[trn_idx]]
        val_dataset = [x1_path[val_idx], x2_path[val_idx], y_path[val_idx]]
        yield train_dataset, val_dataset


#  warm up
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


#  多尺度训练
def random_scale(x1, x2, y, img_scale, ratio_range, div=32):
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = [int(img_scale[0] * ratio), int(img_scale[1] * ratio)]
        return scale

    scale = random_sample_ratio(img_scale, ratio_range)
    scale = [(s // div) * div for s in scale]
    x1 = vF.resize(x1, scale)
    x2 = vF.resize(x2, scale)
    y = vF.resize(y, scale)

    return x1, x2, y


#  测试时增强
def TTA(x1, x2, model, DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu', deep_supervision=False):
    x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
    x1_flip1 = torch.flip(x1, [2]).to(DEVICE)
    x1_flip2 = torch.flip(x1, [3]).to(DEVICE)
    x2_flip1 = torch.flip(x2, [2]).to(DEVICE)
    x2_flip2 = torch.flip(x2, [3]).to(DEVICE)
    output = []
    if deep_supervision:
        output1 = model(x1, x2)[-1].cpu().data.numpy()
        output2 = torch.flip(model(x1_flip1, x2_flip1)[-1], [2]).cpu().data.numpy()
        output3 = torch.flip(model(x1_flip2, x2_flip2)[-1], [3]).cpu().data.numpy()
    else:
        output1 = model(x1, x2).cpu().data.numpy()
        output2 = torch.flip(model(x1_flip1, x2_flip1), [2]).cpu().data.numpy()
        output3 = torch.flip(model(x1_flip2, x2_flip2), [3]).cpu().data.numpy()
    output.append(output1)
    output.append(output2)
    output.append(output3)
    output = np.mean(np.array(output), axis=0)
    return output


#  快速傅里叶变换->风格统一
def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:, :, i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:, :, i])
        target_image_fshift = np.fft.fftshift(target_image_f)

        change_length = 1
        source_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
        int(h / 2) - change_length:int(h / 2) + change_length] = \
            target_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length]

        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)

        source_image_if[source_image_if > 255] = np.max(source_image[:, :, i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)
    out = out.astype(np.uint8)
    return out
