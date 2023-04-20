import io
import json
import torch
import base64
from PIL import Image
from fastapi import FastAPI

app = FastAPI(
    title="Cars Classification API",
    description="Returns 2 endpoints ping and infer",
)


@app.get("/ping")
async def root():
    return {"message": "PONG"}

@app.post("/infer")
async def root(json_file):

    with open(json_file, "rb") as file:
        raw = file.read()
        data = json.loads(raw)
        for v in data.values():
            img = base64.b64decode(v)
            img = Image.open(io.BytesIO(img))

    model = torch.load("models/")
    


    year = pred.rsplit(' ',1)[-1]
    # # realized one cannot naively state that the 2nd last word is the that some of the makes 
    ls_makes = ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe', 'Wagon', 'Van', 'Minivan']
    make = pred.rsplit(' ',2)[-2] if x.rsplit(' ',2)[-2] in ls_makes else np.nan
    model = " ".join(pred.rsplit(' ',2)[:-2])

    key = 'class' and value = "Make, Model, Year”
    return {f'class': "{make}, {model}, {year}"}