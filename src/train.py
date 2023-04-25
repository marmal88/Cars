import os
import pandas as pd
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm 
import io
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.config.load_config import read_yaml_file
from src.dataset import CarsDataset
from src.model import ClassifierModel


cfg = read_yaml_file()

# Loading of training dataset
train_df_path = cfg["training"]["train_df_path"]
train_df = pd.read_csv(train_df_path) 
class_names = [x for x in train_df["class_names"].unique()]
num_class = train_df["class"].nunique()

# Set inital parameters of the training loop
epochs = cfg["training"]["epochs"]
batch_size = cfg["training"]["batch_size"]
custom_crop = cfg["training"]["custom_crop"]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomGrayscale(p=0.1),
                T.Resize([224,224]),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
train_num = int(round(0.8*len(train_df)))
valid_num = int(round(0.2*len(train_df)))

# Create dataset for training run
path_to_data = os.path.join(os.getcwd(), "data/car_ims")
dataset = CarsDataset(csv_file=train_df_path, root_dir=path_to_data, transform=transform, custom_crop=custom_crop)

# Split dataset into training and validation runs 
train_set, valid_set = torch.utils.data.random_split(dataset, [train_num, valid_num])

# Load the datasets into train and validation loaders respectively
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, shuffle=True)

# Instantiate Model
model = ClassifierModel(num_class)
if torch.cuda.is_available():
    model.to("cuda")
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001) 

# Instantiate tensorboard for metrics tracking
writer = SummaryWriter()

max_valid_acc=0.0
for e in range(epochs):
    
    # Training Loop
    train_loss = 0.0
    model.train()
    for data, labels in tqdm(train_loader):
        if torch.cuda.is_available():
            data, labels = data.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        target = model(data)
        loss = loss_fn(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    training_loss = train_loss/len(train_loader)
    writer.add_scalar("Loss/Train", training_loss, e)
    
    print(f'Epoch {e+1} | Training Loss: {training_loss:.4f}')

    # Validation Loop
    valid_loss = 0.0
    valid_correct = 0
    model.eval()    
    for batch_idx, (data, labels) in enumerate(tqdm(valid_loader)):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = model(data)
        loss = loss_fn(target,labels)
        valid_loss = loss.item() * data.size(0)

        # Storing of grid images in tensorboard,
        # Commented out for faster training times
        # grid = make_grid(data)
        # writer.add_image("images", grid)
        # writer.add_graph(model, data)

        _, predicted = torch.max(target.detach(), 1)
        valid_correct += (predicted == labels).sum().item()

    valid_loss = valid_loss/len(valid_loader)
    valid_acc = (valid_correct / len(valid_loader.dataset)) * 100
    writer.add_scalar("Correct/Val", valid_correct ,e)
    writer.add_scalar("Loss/Val", valid_loss, e)
    writer.add_scalar("Acc/Val", valid_acc, e)

    print(f'Epoch {e} | Validation Loss: {valid_loss:.4f} | Validation Accuracy: {valid_acc:.2f}%')
    
    # Save out the models only if its accuracy exceed config amount 
    # and does better than the highest accuracy so far
    save_above = cfg["training"]["save_above"]
    if (valid_acc>save_above) and (valid_acc>max_valid_acc):
        max_valid_acc = valid_acc
        
        model_name = cfg["training"]["model_name"]
        if custom_crop:
            name = f"{model_name}_{valid_acc:.2f}_crop"
        else:
            name = f"{model_name}_{valid_acc:.2f}_nocrop"
        
        model_path = os.path.join(os.getcwd(), f"models/{name}.pth") 
        torch.save(model.state_dict(), model_path)
    
        # Log out each layer of the model per epoch
        for name, weight in model.named_parameters():
            writer.add_histogram(name,weight, e)
            writer.add_histogram(f'{name}.grad',weight.grad, e)

writer.close()

