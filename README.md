# Cars

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [1. Folder Structure](#1-folder-structure)
  - [2. Application Setup](#2-application-setup)
  - [3. Explain the data pre-processing steps](#3-explain-the-data-pre-processing-steps)
    - [3.1 Assumptions](#31-assumptions)
    - [3.2 Pre-processing steps](#32-pre-processing-steps)
    - [3.3 Other Considerations](#33-other-considerations)
  - [4. Model Selection](#4-model-selection)
  - [5. Training and Validation results](#5-training-and-validation-results)

---
## 1. Folder Structure
```bash
.
├── 
├── requirements.txt
├── README.MD
├── data
│   ├── annotations  # image metadata/csv files
│   └── car_ims      # image dataset
├── docker           # docker files
├── models           # location of pth files
└── src              # fastapi code
```
---

## 2. Application Setup

Application has been containerized using docker compose for ease of deployment.

1. To run the fastapi server, please use the command:
    ```bash
    docker compose -f docker/docker-compose.yml up   
    ```

2. Once running you can go to the UI using your local browser at `http://localhost:4000/docs`

3. To stop the fastAPI server, please use the command:
    ```bash
    docker compose -f docker/docker-compose.yml down
    ```

To remove existing docker

---

## 3. Explain the data pre-processing steps

### 3.1 Assumptions 
- Images are not flipped and in the correct orientation
- Bounding boxes provided are all correct with no errors

### 3.2 Pre-processing steps
- As there was a bounding box present for each image
- Created a toggle to allow for training of images either cropped to bounding box or not

### 3.3 Other Considerations
- Batch size selected to be maximum that could be fit into vram memory
- 

---

## 4. Model Selection

The model selected was a is a **resnet101**.

Considerations for model selection
- Able to be retrained within timeline (~1 day) with fairly accurate results  
  - Already pretrained on imagenet
  - Performs well across various machine learning classification tasks
- Reasonable model architecture 
  - Model has reasonable depth and size to be able to get baseline classification
  - Model makes use of skip connections to allow gradients to flow directly accross network

---
## 5. Training and Validation results 
(confusion matrix, graph output
etc.)