import io
import json
import torch
import base64
import BytesIO
from PIL import Image
import fastapi 
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torchvision.models as models

from src.config.load_config import read_yaml_file


cfg = read_yaml_file()

app = FastAPI(
    title="Cars Classification API",
    description="Returns 2 endpoints ping and infer",
)

@app.get("/ping",  status_code=fastapi.status.HTTP_200_OK)
async def root():
    return {"message": "PONG"}

@app.post("/infer",  status_code=fastapi.status.HTTP_200_OK)
async def root(image: UploadFile = File()):
    if not image:
        raise HTTPException(status_code=fastapi.status.HTTP_400_BAD_REQUEST, detail="Image file not found")
    
    image_file = await image.read()
    image_data = BytesIO(image_file)
    
    model_path = "models/model.pth"
    model = models.resnet101()
    model.load_state_dict(torch.load(cfg.model_path, map_location=torch.device('cpu')))
    model.eval()

    classification = model(image_data)

    return {"class": classification}