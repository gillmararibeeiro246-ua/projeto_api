from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid

app = FastAPI()

# 🔥 LIBERAR CORS (para funcionar no Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 CAMINHO DO MODELO NO RENDER
model = YOLO("api_telhad/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # criar nome único para evitar conflito
    temp_filename = f"{uuid.uuid4()}.jpg"

    try:
        # salvar imagem temporária
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # rodar modelo
        results = model(temp_filename)

        names = model.names
        detections = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                detections.append(class_name)

        total_telhados = len(detections)

        contagem = {}
        for classe in detections:
            contagem[classe] = contagem.get(classe, 0) + 1

        return {
            "total_telhados": total_telhados,
            "por_material": contagem
        }

    finally:
        # apagar imagem temporária se existir
        if os.path.exists(temp_filename):
            os.remove(temp_filename)