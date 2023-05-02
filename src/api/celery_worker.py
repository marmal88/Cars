import os
import io
import cv2
import base64
from celery import Celery
from dotenv import load_dotenv
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.config.load_config import read_yaml_file
from src.model import ClassifierModel

load_dotenv(".env")
BROKER_URL = os.environ['CELERY_BROKER_URL']
RESULT_BACKEND = os.environ['CELERY_RESULT_BACKEND']

celery = Celery(
            main="celery_app",
            broker=BROKER_URL,
            backend=RESULT_BACKEND,
        )

# Loading of Config
cfg = read_yaml_file()

# Loading of background information
train_df_path = cfg["training"]["train_df_path"]
train_df = pd.read_csv(train_df_path) 
num_class = train_df["class"].nunique()
class_names = [x for x in train_df["class_names"].unique()]

# Loading of the Model
model_path = cfg["inference"]["model_path"]
model = ClassifierModel(num_class)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
if torch.cuda.is_available():
    model.to("cuda")
model.eval()

# Message Queue
@celery.task
def celery_ping2(name):
    if name.lower()=="false":
        return "I didnt get your name"
    return f"Hello {name}"

@celery.task
def celery_infer2(image_base64):
    # try:
    image_file = base64.b64decode(image_base64)
    image_data = io.BytesIO(image_file)
    image_data.seek(0)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_data).convert('RGB')
    print(img.size)
    print(transform(img).size)

    image_tensor = transform(img).unsqueeze(0)
    print(image_tensor.size)
    batch_tensor = image_tensor
    print(type(batch_tensor))
    output = model(image_tensor)
    print("station3")
    _, predicted = torch.max(output.detach(), 1)
    print("station4")
    classification = class_names[predicted.item()]
    print("station5")

    return classification
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return None

