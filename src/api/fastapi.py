import torch
import pandas as pd
import io
import cv2
from PIL import Image
import base64
import fastapi 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as T
from typing import Dict

from src.api.celery_worker import celery_infer2, celery_ping2
from src.config.load_config import read_yaml_file
from src.model import ClassifierModel

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

app = FastAPI(
    title="Cars Classification API",
    description="Returns 2 endpoints /ping and /infer on port 4000",
)

@app.get("/ping",  status_code=fastapi.status.HTTP_200_OK)
async def root()->Dict:
    """ 
    Returns:
        Dict: Response in JSON format
    """    
    return JSONResponse({"message": "pong"})

@app.get("/ping2",  status_code=fastapi.status.HTTP_200_OK)
async def root(name=False)->Dict:
    message = celery_ping2.delay(name)

    return JSONResponse({"message": message.get()})


@app.post("/infer2",  status_code=fastapi.status.HTTP_200_OK)
async def root(image: UploadFile = File())->Dict:
    if not image:
        raise HTTPException(status_code=fastapi.status.HTTP_400_BAD_REQUEST, detail="Image file not found")
    
    image_bytes = await image.read()
    # _, buffer = cv2.imencode(".jpg", image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    classification = celery_infer2.delay(image_base64)

    return JSONResponse({"class": classification.get()})


@app.post("/infer",  status_code=fastapi.status.HTTP_200_OK)
async def root(image: UploadFile = File())->Dict:
    """ Inference Endpoint 
        Takes in an image and returns the classification of the image.
    Args:
        image (UploadFile, optional): Image in file format to be inferenced. Defaults to File().
    Raises:
        HTTPException: Raises a 400 Error if there is no image file.
    Returns:
        Dict: Response in JSON format with classification.
    """    
    if not image:
        raise HTTPException(status_code=fastapi.status.HTTP_400_BAD_REQUEST, detail="Image file not found")
    
    image_file = await image.read()
    image_data = io.BytesIO(image_file)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(Image.open(image_data).convert('RGB')).unsqueeze(0)
    output = model(image_tensor)
    _, predicted = torch.max(output.detach(), 1)
    classification = class_names[predicted.item()]

    return JSONResponse({"class": classification})
