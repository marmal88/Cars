import torchvision
import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


class ClassifierModel(nn.Module):

    def __init__(self, num_class:int):
        """ Initializes the ClassifierModel instance
            The super here inherits the functions from the base torch nn.Module, allowing 
            us to create layers and convolutions.
            Added a dropout to the last linear layer and amended out_features to num_class
        Args:
            num_class (int): The number of classes in the classification problem.
        """        
        super().__init__()
        self.model = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_class))

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x = self.model(x)
        return x