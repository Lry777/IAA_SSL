import os
import torch
import numpy as np
from torchvision import transforms
import json
from sklearn.model_selection import train_test_split
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.datasets.folder import default_loader
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import os.path
from pathlib import Path
from typing import Tuple, List
import torchvision.transforms as T
import scipy.io
import torch
from torch.utils.data import Dataset
from PIL import Image

def load_transforms(
    input_shape: Tuple[int, int] = (256, 256),
) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape),
        T.ToTensor()
    ])

class AADB_dataset(torch.utils.data.Dataset):

    attributes = [
        "score",
        "balancing_elements",
        "color_harmony",
        "content",
        "depth_of_field",
        "light",
        "motion_blur",
        "object",
        "repetition",
        "rule_of_thirds",
        "symmetry",
        "vivid_color"
    ]

    splits = {
        "train": {"idx": 0, "file": "imgListTrainRegression_score.txt"},
        "test": {"idx": 1, "file": "imgListTestNewRegression_score.txt"},
        "val": {"idx": 2, "file": "imgListValidationRegression_score.txt"}
    }

    labels_file = "attMat.mat"

    def __init__(
        self,
        image_dir: str = "data/aadb/images",
        labels_dir: str = "data/aadb/labels",
        split: str = "train",
        transforms: T.Compose = load_transforms()
    ):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.files, self.labels, self.scores = self.load_split(split)

    def load_split(self, split: str) -> Tuple[List[str], np.ndarray]:
        # Load labels
        assert split in ["train", "val", "test"]
        labels_path = os.path.join(self.labels_dir, self.labels_file)
        labels = scipy.io.loadmat(labels_path)["dataset"]

        labels = labels[0][self.splits[split]["idx"]]

        # Load file paths
        files_path = os.path.join(self.labels_dir, self.splits[split]["file"])

        with open(files_path, "r") as f:
            files_all = f.read().strip().splitlines()
            files = [f.split()[0] for f in files_all]
            scores = [f.split()[1] for f in files_all]

            files = [os.path.join(self.image_dir, f) for f in files]

        return files, labels, scores

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = Image.open(self.files[idx]).convert("RGB")
        x = self.transforms(x)
        y = torch.from_numpy(self.labels[idx])
        print(len(x[0][1]), len(x[1][1]), y, self.scores[idx])
        return x, y

splits = {
        "train": {"idx": 0, "file": "imgListTrainRegression_score.txt"},
        "test": {"idx": 1, "file": "imgListTestNewRegression_score.txt"},
        "val": {"idx": 2, "file": "imgListValidationRegression_score.txt"}
    }

def clearn_all_aadb_data(data_path, split='train'):
    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels/attMat.mat')
    labels_score_path = os.path.join(data_path, 'labels', splits[split]['file'])

    train_data = []
    labels = scipy.io.loadmat(labels_path)["dataset"]
    labels = labels[0][splits[split]['idx']]

    with open(labels_score_path, "r") as f:
        files_all = f.read().strip().splitlines()
        files = [f.split()[0] for f in files_all]
        scores = [f.split()[1] for f in files_all]

    assert len(files)==len(labels), '11'
    for i in range(len(files)):
        # print(labels[i][0], scores[i])
        assert float(labels[i][0])==float(scores[i]), '22'
        temp = list(labels[i])
        temp.insert(0,files[i])
        train_data.append(temp)
    print(len(train_data))
    columns = ['image_id', 'quality', 'cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8', 'cl9', 'cl10', 'cl11']
    train_data = pd.DataFrame(train_data)
    train_data.columns = columns

    return train_data




if __name__ == "__main__":
    data_path = '/home/xiexie/data/AADB'
    # 整理出测试集和训练集
    df_train = clearn_all_aadb_data(data_path, split='train')
    df_train.to_csv(os.path.join(data_path,'train.csv'), index=False)

    df_test = clearn_all_aadb_data(data_path, split='test')
    df_test.to_csv(os.path.join(data_path,'test.csv'), index=False)

    # 从训练集中划分有标签和无标签
    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    df_lb, df_ulb = train_test_split(df_train, train_size=0.5)
    df_lb.to_csv(os.path.join(data_path, 'lb_0.5.csv'), index=False)
    #
    # path_train_alllabels = glob.glob(os.path.join(data_path, 'labels', 'imgListTrain*.txt'))
    # print(len(path_train_alllabels))