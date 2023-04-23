# Cars

## Table of Contents
- [Cars](#cars)
  - [Table of Contents](#table-of-contents)
  - [1. Folder Structure](#1-folder-structure)
  - [2. Application Setup](#2-application-setup)
  - [3. Explain the data pre-processing steps](#3-explain-the-data-pre-processing-steps)
    - [3.1 Assumptions](#31-assumptions)
    - [3.2 Pre-processing steps](#32-pre-processing-steps)
    - [3.3 Configurations](#33-configurations)
  - [4. Model Selection](#4-model-selection)
  - [5. Training and Validation results](#5-training-and-validation-results)

---
## 1. Folder Structure
```bash
.
├── Cars.ipynb                 # Artefact #1 (sample of training run, actual was done using .py files)
├── requirements.txt
├── README.MD
├── data
│   ├── annotations  
│   └── car_ims      
├── docker                     
│   ├── docker-compose.yml         
│   └── inference.Dockerfile   # Artefact #3 Dockerfile
├── runs                       # location of tensorboard run files
├── models                     # location of .pth files
└── src              
    ├── api  
    │   └── fastapi            # Artefact #2 fastapi endpoint
    ├── config
    │   ├── config.yml
    │   └── load_config.py
    ├── dataset.py
    ├── model.py
    └── train.py
```
---

## 2. Application Setup

Application has been containerized using docker compose for ease of deployment.

1. To run the fastapi server, please use the command:
    ```bash
    docker compose -f docker/docker-compose.yml up -d
    ```

2. Once running you can go to the UI using your local browser at `http://localhost:4000/docs`

3. To stop the fastAPI server, please use the command:
    ```bash
    docker compose -f docker/docker-compose.yml down
    ```

4. To remove existing docker images
    ```bash
    docker image rm fastapi-server-cars:0.1.0
    ```

---

## 3. Explain the data pre-processing steps

### 3.1 Assumptions 
- Images are not flipped and in the correct orientation
- Bounding boxes provided are all correct with no errors

### 3.2 Pre-processing steps
- As there was a bounding box present for each image
- Created a toggle to allow for training of images either cropped to bounding box or not
  - Generally found that cropping the image allowed for better inference 

### 3.3 Configurations
- Config file provided to allow for customization of the training.
- Batch size selected to be maximum that could be fit into vram memory.
- No hyperparameter tuning was used.

---
## 4. Model Selection

The model selected was a standard **resnet**.

Considerations for model selection
- Able to be retrained within timeline (~1 day) with fairly accurate results  
  - Already pretrained on imagenet
  - Performs well across various machine learning classification tasks
- Reasonable model architecture 
  - Model comes with different depth and weights
    - Generally found that the larger model (ResNet152) was able to deliver better accuracy given same parameters
  - Model has reasonable depth and size to be able to get baseline classification
  - Model makes use of skip connections to allow gradients to flow directly accross network

---
## 5. Training and Validation results 

During the training and validation phases the model training loss, validation loss and validation accuracy were measured using tensorboard. 

These outputs were logged using tensorboard. Tensorboard can be run through:
```bash
tensorboard --logdir=runs --port 6006
```
Once running the port can be accessed at `localhost:6006`.

---