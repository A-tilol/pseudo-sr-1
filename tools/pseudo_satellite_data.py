import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as nnF
import torch

import albumentations as album
import albumentations.pytorch

import glob


class SatData(Dataset):
    def __init__(self,
                 data_lr,
                 data_hr=None,
                 b_train=True,
                 rgb=True,
                 img_range=1,
                 shuffle=True,
                 z_size=(8, 8),
                 scale=4):
        self.ready_hr = data_hr is not None
        if data_hr is not None:
            self.hr_files = glob.glob(os.path.join(data_hr, "*.tif"),
                                      recursive=True)
            self.hr_files.sort()
        self.lr_files = glob.glob(os.path.join(data_lr, "*.tif"),
                                  recursive=True)
        self.lr_files.sort()
        if shuffle:
            if data_hr is not None:
                np.random.shuffle(self.hr_files)
            np.random.shuffle(self.lr_files)
        self.training = b_train
        self.rgb = rgb
        self.z_size = z_size
        self.scale = scale
        self.img_min_max = (0, img_range)
        if self.training:
            self.preproc = album.Compose([
                album.HorizontalFlip(p=0.5),
                album.RandomRotate90(p=1),
                album.pytorch.ToTensor()
            ])
        else:
            self.preproc = album.Compose([album.pytorch.ToTensor()])

    def __len__(self):
        if self.ready_hr:
            return min([len(self.lr_files), len(self.hr_files)])
        else:
            return len(self.lr_files)

    def __getitem__(self, index):
        data = dict()
        if np.prod(self.z_size) > 0:
            data["z"] = torch.randn(1, *self.z_size, dtype=torch.float32)

        lr_idx = index % len(self.lr_files)
        lr = cv2.imread(self.lr_files[lr_idx])
        if self.rgb:
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        data["lr_path"] = self.lr_files[lr_idx]

        hr_idx = index % len(self.hr_files)
        hr = cv2.imread(self.hr_files[hr_idx])
        if self.rgb:
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        data["hr_path"] = self.hr_files[hr_idx]

        aug_imgs = self.preproc(image=lr, hr=hr)
        data["lr"] = aug_imgs["image"] * self.img_min_max[1]
        data["hr"] = aug_imgs["hr"] * self.img_min_max[1]
        print(aug_imgs["hr"] * self.img_min_max[1])

        data["hr_down"] = nnF.interpolate(
            data["hr"].unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False).clamp(
                min=self.img_min_max[0], max=self.img_min_max[1]).squeeze(0)
        return data

    def get_noises(self, n):
        return torch.randn(n, 1, *self.z_size, dtype=torch.float32)

    def permute_data(self):
        if self.ready_hr:
            np.random.shuffle(self.hr_files)
        np.random.shuffle(self.lr_files)


if __name__ == "__main__":
    high_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")
    low_folder = os.path.join(os.environ["DATA_TRAIN"], "LOW/wider_lnew")
    test_folder = os.path.join(os.environ["DATA_TEST"])
    img_range = 1
    data = SatData(low_folder, high_folder, img_range=img_range)
    for i in range(len(data)):
        d = data[i]
        for elem in d:
            if elem in ['z', 'lr_path', 'hr_path']: continue
            img = np.around((d[elem].numpy().transpose(1, 2, 0) / img_range) *
                            255.0).astype(np.uint8)
            cv2.imshow(elem, img[:, :, ::-1])
        cv2.waitKey()
    print("fin.")
