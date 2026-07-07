from fastapi import *
from fastapi.responses import FileResponse

import shutil
from fastapi.middleware.cors import CORSMiddleware
from src.predict import predict_image

from src.report_generator import (
    generate_report
)

from api.database import *
import os

os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")

async def predict(
    patient_name:str,
    file:UploadFile
):

    path = (
        f"uploads/{file.filename}"
    )

    with open(path,"wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_image(path)

    db = SessionLocal()

    scan = Scan(
        patient_name=patient_name,
        prediction=result["prediction"],
        confidence=result["confidence"],
        image_path=path,
        heatmap_path=result["heatmap"]
    )

    db.add(scan)

    db.commit()

    report_path = (
        f"reports/{scan.id}.pdf"
    )

    generate_report(
        patient_name,
        result["prediction"],
        result["confidence"],
        report_path
    )

    return {

        "id":scan.id,

        "prediction":
        result["prediction"],

        "confidence":
        result["confidence"],

        "heatmap":
        result["heatmap"],

        "report_url":
        f"/report/{scan.id}"
    }

@app.get("/history")

def history():

    db = SessionLocal()

    scans = db.query(
        Scan
    ).all()

    return scans

@app.get("/report/{scan_id}")

def report(scan_id:int):

    return FileResponse(
        f"reports/{scan_id}.pdf"
    )

from fastapi.responses import FileResponse

@app.get("/heatmap/{filename}")
def heatmap(filename:str):

    return FileResponse(
        f"heatmaps/{filename}"
    )