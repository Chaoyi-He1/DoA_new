import math
import os
import random
import shutil
from pathlib import Path
import pickle
from typing import Tuple
import cv2
import pandas as pd
import numpy as np
import torch
from skimage.transform import resize
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm


def read_data_pickle(data_path, data_size):
    data = np.fromfile(data_path, dtype=np.float32).reshape(6144, 512)
    data = np.asarray(data, dtype=float)
    data = np.reshape(data, newshape=(-1, data_size, data_size)) if len(data.shape) != 3 else data
    assert not (np.isnan(data)).any()
    return data


class LoadDataAndLabels(Dataset):
    def __init__(self, folder_path: str = './path/train', cache: bool = False,
                 train: bool = True, data_size: int = 512, rank: int = -1) -> None:
        super().__init__()
        self.data_folder_path = os.path.join(folder_path, 'images')
        self.label_folder_path = os.path.join(folder_path, 'labels')
        data_files = os.listdir(self.data_folder_path)
        data_files.sort()
        self.label_files = [os.path.join(self.label_folder_path, f.replace('.bin', '.txt')) for f in data_files]
        self.data_files = [os.path.join(self.data_folder_path, f) for f in data_files]
        self.cache = cache
        self.train = train
        self.data_size = data_size
        self.rank = rank
        self.data = []
        self.label = []
        
        if self.cache:
            self.cache_data()
        
    def cache_data(self):
        print('Caching data...')
        if self.rank in [-1, 0]:
            pbar = tqdm(range(len(self.data_files)))
        else:
            pbar = range(len(self.data_files))
            
        for i in pbar:
            data_path = self.data_files[i]
            label_path = self.label_files[i]
            
            with open(label_path, 'r') as f:
                label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            assert label.shape[1] == 7, "> 5 label columns: %s" % label_path
            assert (label >= 0).all(), "negative labels: %s" % label_path
            assert (label[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % label_path
            
            data = read_data_pickle(data_path, self.data_size)
            
            self.data.append(data)
            self.label.append(label)
        
        print('Done!')
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache:
            data = self.data[index]
            label = self.label[index]
        else:
            data_path = self.data_files[index]
            label_path = self.label_files[index]
            
            with open(label_path, 'r') as f:
                label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            assert label.shape[1] == 7, "> 5 label columns: %s" % label_path
            assert (label >= 0).all(), "negative labels: %s" % label_path
            assert (label[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % label_path
            
            data = read_data_pickle(data_path, self.data_size)
        
        data = torch.from_numpy(data).unsqueeze(0)
        label = torch.from_numpy(label)
        
        return data, label

    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = np.stack(data, axis=0)
        label = np.stack(label, axis=0)
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        return data, label
    