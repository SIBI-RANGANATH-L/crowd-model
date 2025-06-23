from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from model import detect_people
from firebase_init import db
from datetime import datetime
import asyncio
from typing import List

app = FastAPI()

# Allow CORS from frontend origin (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log people count to Firestore
def log_people_count(count: int):
    db.collection('crowd_logs').add({
        'timestamp': datetime.utcnow(),
        'people_count': count
    })

@app.get("/")
async def root():
    return {"message": "YOLOv8 Realtime API"}

# Endpoint to get recent crowd data for chart
@app.get("/crowd-data")
async def get_crowd_data():
    docs = db.collection('crowd_logs').order_by('timestamp').limit(50).stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        data.append({
            "timestamp": d['timestamp'].isoformat(),
            "people_count": d['people_count']
        })
    return data

# WebSocket for streaming base64 frames from frontend
@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # data is base64 encoded frame: "data:image/jpeg;base64,...."
            header, base64_str = data.split(",", 1)
            img_bytes = base64.b64decode(base64_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            coords, count = detect_people(frame)
            log_people_count(count)

            # Send back detections as JSON
            await websocket.send_json({"people_count": count, "coords": coords})
    except WebSocketDisconnect:
        print("Client disconnected")
