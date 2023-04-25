import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

from src.config.load_config import read_yaml_file

class ClassifierModel(nn.Module):

    def __init__(self, num_class:int):
        """ Initializes the ClassifierModel instance
            Added a dropout to the last linear layer and amended out_features to num_class
        Args:
            num_class (int): The number of classes in the classification problem.
        """        
        super().__init__()
        self.cfg = read_yaml_file()

        if self.cfg["training"]["model_name"] == "resnet101":
            model_func = resnet101
            weight = ResNet101_Weights.IMAGENET1K_V2
        elif self.cfg["training"]["model_name"] == "resnet50":
            model_func = resnet50
            weight = ResNet50_Weights.IMAGENET1K_V2
        else:
            print(f"Model not found, please ensure model loaded in model.py file")

        if self.cfg["inference"]["run_inference"]==True:
            self.model = model_func()
        else:
            self.model = model_func(weights=weight)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_class))

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x = self.model(x)
        return x

