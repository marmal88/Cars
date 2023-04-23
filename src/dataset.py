import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Dict 


class CarsDataset(Dataset):

    def __init__(self, csv_file:str, root_dir:str, transform=None, custom_crop=False)->Dict:
        """ Initializes the CarsDataset with the necessary variables
        Args:
            csv_file (str): Path to a CSV file containing the image paths and labels.
            root_dir (str): Root directory where the images are stored.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. Default: None.
            custom_crop (bool, optional): Whether to crop the images based on bounding box
                coordinates specified in the CSV file. Default: False.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.custom_crop = custom_crop

    def __len__(self):
        """ Returns the number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx:str):
        """ Loads and preprocesses the image and label at the specified index.
            Provides an avenue to perform transformations on the dataset.
        Args:
            idx (int): The index of the sample to load.
        Returns:
            tuple: A tuple containing the preprocessed image tensor and its corresponding label tensor.
        """
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

    def _custom_crop(self, img:Image, left:str, top:str, right:str, bottom:str)->Image:
        """ Crops the image based on bounding box coordinates.
        Args:
            img (PIL.Image): The image to crop.
            left (int): The left coordinate of the bounding box.
            top (int): The top coordinate of the bounding box.
            right (int): The right coordinate of the bounding box.
            bottom (int): The bottom coordinate of the bounding box.
        Returns:
            PIL.Image: The cropped image.
        """
        width = right-left
        height = bottom-top
        img = TF.crop(img, top, left, height, width)

        return img