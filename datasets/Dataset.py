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


def read_img_pickle(img_path, img_size):
    # with open(img_path, 'rb') as fo:
    #     # img = pickle.load(fo, encoding='bytes')
    #     img = fo.read()
    img = np.fromfile(img_path, dtype=np.float32).reshape(6144, 512)
    img = np.asarray(img, dtype=float)
    img = np.reshape(img, newshape=(-1, img_size, img_size)) if len(img.shape) != 3 else img
    assert not (np.isnan(img)).any()
    return img


class LoadDataAndLabels(Dataset):
    def __init__(self) -> None:
        super().__init__()
    