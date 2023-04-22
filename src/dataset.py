import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Dict 


class CarsDataset(Dataset):

    def __init__(self, csv_file:str, root_dir:str, transform=None, custom_crop=False)->Dict:
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.custom_crop = custom_crop

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.image[idx])
        img = Image.open(img_name).convert("RGB")
        label = self.dataframe["class"][idx]
        left, top, right, bottom = self.dataframe.x1[idx], self.dataframe.y1[idx], self.dataframe.x2[idx], self.dataframe.y2[idx]
        
        if self.custom_crop:
            img = self._custom_crop(img, left, top, right, bottom)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(int(label))

    def _custom_crop(self, img, left, top, right, bottom):   
        width = right-left
        height = bottom-top
        img = TF.crop(img, top, left, height, width)

        return img