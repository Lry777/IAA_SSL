import os.path
from pathlib import Path
from typing import Tuple, List
import torchvision.transforms as T
import scipy.io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import default_loader



class AVADataset(Dataset):
    def __init__(self, path_to_csv: Path, images_path: Path, train, transform):

        self.transform = transform
        if train == 'train':
            self.df = pd.read_csv(os.path.join(path_to_csv, "train.csv"))
        elif train == 'lb_0.01':
            self.df = pd.read_csv(os.path.join(path_to_csv, "mix_lb_cl_0.01.csv"))
        elif train == 'lb_0.1':
            self.df = pd.read_csv(os.path.join(path_to_csv, "mix_lb_cl_0.1.csv"))
        elif train == 'lb_0.2':
            self.df = pd.read_csv(os.path.join(path_to_csv, "mix_lb_cl_0.2.csv"))
        elif train == 'lb_0.5':
            self.df = pd.read_csv(os.path.join(path_to_csv, "mix_lb_cl_0.5.csv"))
        elif train == 'test':
            self.df = pd.read_csv(os.path.join(path_to_csv, "test.csv"))
        else:
            print('ava load woring transform')

        self.train = train
        self.images_path = images_path
    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
        row = self.df.iloc[item]

        image_id = int(row["image_id"])
        # print(image_id)
        # image_path = self.images_path / f"{image_id}.jpg"
        image_path = os.path.join(self.images_path, str(image_id)+'.jpg')
        # image = default_loader(image_path)
        image = Image.open(image_path).convert('RGB')
        # im = cv2.imread(image_path)
        # print('yuan:', im.shape)
        x = self.transform(image)

        # y = row[1:11].values.astype("float32")
        # c = row[11]-1
        # p = y / y.sum()

        if self.train == 'test':
            y = row[1:11].values.astype("float32")
            p = y / y.sum()
            return image_id, x, p
        elif self.train == 'ft' or self.train == 'lb':
            y = row[1:11].values.astype("float32")
            c = row[-1] - 1
            p = y / y.sum()
            return x, p, c
        else:
            y = row[1:11].values.astype("float32")
            p = y / y.sum()
        return x, p, p


class AADB_dataset(torch.utils.data.Dataset):


    def __init__(
        self,
        path_to_csv,
        path_to_images,
        split,
        transforms
    ):
        self.path_to_images = path_to_images
        self.transforms = transforms
        self.split = split
        if split == 'train':
            self.df = pd.read_csv(os.path.join(path_to_csv, "train.csv"))
        elif split == 'lb_0.01':
            self.df = pd.read_csv(os.path.join(path_to_csv, "lb_0.01.csv"))
        elif split == 'lb_0.1':
            self.df = pd.read_csv(os.path.join(path_to_csv, "lb_0.1.csv"))
        elif split == 'lb_0.2':
            self.df = pd.read_csv(os.path.join(path_to_csv, "lb_0.2.csv"))
        elif split == 'lb_0.5':
            self.df = pd.read_csv(os.path.join(path_to_csv, "lb_0.5.csv"))
        elif split == 'test':
            self.df = pd.read_csv(os.path.join(path_to_csv, "test.csv"))
        else:
            print('aadb load woring transform')

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int):
        row = self.df.iloc[item]
        image_id = row["image_id"]
        # print(image_id)
        # image_path = self.images_path / f"{image_id}.jpg"
        image_path = os.path.join(self.path_to_images, image_id)
        # image = default_loader(image_path)
        image = Image.open(image_path).convert('RGB')
        # im = cv2.imread(image_path)
        # print('yuan:', im.shape)
        x = self.transforms(image)

        if self.split == 'test':
            # y = row[1:13].values.astype("float32")
            y = row[1:13].values.astype("float32")
            return image_id, x, y
        else:
            # y = row[1:13].values.astype("float32")
            y = row[1:13].values.astype("float32")
            return x, y, y
