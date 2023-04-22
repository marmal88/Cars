import os
import pandas as pd
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

from src.config.load_config import read_yaml_file
from src.dataset import CarsDataset
from src.model import ClassifierModel


cfg = read_yaml_file()

train_df_path = cfg["training"]["train_df_path"]
train_df = pd.read_csv(train_df_path) 

num_class = train_df["class"].nunique()
epochs = 30
batch_size = 37
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = T.Compose([
                T.Resize([224,224]),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])

custom_crop = cfg["training"]["custom_crop"]

train_num = int(round(0.8*len(train_df)))
valid_num = int(round(0.2*len(train_df)))

path_to_data = os.path.join(os.getcwd(), "data/car_ims")
dataset = CarsDataset(csv_file=train_df_path, root_dir=path_to_data, transform=transform, custom_crop=custom_crop)

train_set, valid_set = torch.utils.data.random_split(dataset, [train_num, valid_num])

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2, shuffle=True)


model = ClassifierModel(num_class)
if torch.cuda.is_available():
    model.to("cuda")
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001) 

writer = SummaryWriter()

for e in range(epochs):
    train_loss = 0.0
    for data, labels in tqdm(train_loader):
        if torch.cuda.is_available():
            data, labels = data.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        target = model(data)
        loss = loss_fn(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {e+1} | Training Loss: {train_loss / len(train_loader)}')

    valid_loss = 0.0
    valid_correct = 0
    model.eval()    
    for data, labels in tqdm(valid_loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = model(data)
        loss = loss_fn(target,labels)
        valid_loss = loss.item() * data.size(0)

        _, predicted = torch.max(target.detach(), 1)
        valid_correct += (predicted == labels).sum().item()

    valid_loss = valid_loss/len(valid_loader)
    valid_acc = (valid_correct / len(valid_loader.dataset)) * 100
    writer.add_scalar("Loss/Val", valid_loss, e)
    writer.add_scalar("Acc/Val", valid_acc, e)

    print(f'Epoch {e+1} | Training Loss: {train_loss:.6f} | Validation Loss: {valid_loss:.6f} | Validation Accuracy: {valid_acc:.2f}%')
    
    if custom_crop:
        name = f"car_model_{valid_acc:.2f}_crop"
    else:
        name = f"car_model_{valid_acc:.2f}_nocrop"
    torch.save(model.state_dict(), f"/content/drive/MyDrive/cars/models/{name}.pth")

writer.close()