from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("api_telhad/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # salvar imagem temporária
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # rodar modelo
    results = model(file.filename)

    # pegar nomes das classes
    names = model.names

    detections = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            detections.append(class_name)

    # contar total
    total_telhados = len(detections)

    # contar por tipo
    contagem = {}
    for classe in detections:
        if classe in contagem:
            contagem[classe] += 1
        else:
            contagem[classe] = 1

    # apagar imagem temporária
    os.remove(file.filename)

    return {
        "total_telhados": total_telhados,
        "por_material": contagem
    }