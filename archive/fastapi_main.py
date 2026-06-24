import os
import sys
import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add parent directory to path to import parkinsons_detection.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parkinsons_detection import LivePredictor

app = FastAPI(title="Parkinson's Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            # Assumes model artifacts are in the same parent directory where parkinsons_detection.py is
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rf_model.pkl')
            selector_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'selected_features.pkl')
            scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scaler.pkl')
            
            # Predictor handles its own loading, but we can pass paths if we modify it
            # For now, we assume it looks in CWD, so we might need to change dir or adjust LivePredictor
            predictor = LivePredictor()
        except Exception as e:
            print(f"Error initializing predictor: {e}")
    return predictor

# Static Data (Ported from Streamlit app)
ALGORITHM_RESULTS = {
    "Gray Wolf Optimization": {
        "accuracy": 0.76,
        "precision_healthy": 0.71,
        "recall_healthy": 0.92,
        "precision_pd": 0.88,
        "recall_pd": 0.58,
        "features_selected": 13,
        "selected_features": [
            "MFCC1_mean", "MFCC2_mean", "MFCC3_mean", "MFCC4_mean", 
            "MFCC6_mean", "MFCC8_mean", "MFCC1_std", "MFCC2_std", 
            "MFCC3_std", "MFCC7_std", "MFCC10_std", "MFCC12_std", "Chroma8"
        ],
        "feature_importance": [
            {"name": "MFCC2_mean", "value": 0.1790}, {"name": "MFCC10_std", "value": 0.1048}, 
            {"name": "MFCC2_std", "value": 0.0961}, {"name": "MFCC1_std", "value": 0.0937}, 
            {"name": "MFCC12_std", "value": 0.0915}, {"name": "MFCC3_std", "value": 0.0741},
            {"name": "MFCC4_mean", "value": 0.0570}, {"name": "MFCC7_std", "value": 0.0535}, 
            {"name": "MFCC1_mean", "value": 0.0541}, {"name": "MFCC8_mean", "value": 0.0438}, 
            {"name": "MFCC3_mean", "value": 0.0475}, {"name": "MFCC6_mean", "value": 0.0351},
            {"name": "Chroma8", "value": 0.0699}
        ]
    },
    "Artificial Bee Colony": {
        "accuracy": 0.72,
        "precision_healthy": 0.69,
        "recall_healthy": 0.85,
        "precision_pd": 0.82,
        "recall_pd": 0.50,
        "features_selected": 15,
        "selected_features": [
            "MFCC1_mean", "MFCC2_mean", "MFCC3_mean", "MFCC5_mean",
            "MFCC6_mean", "MFCC7_mean", "MFCC1_std", "MFCC2_std",
            "MFCC4_std", "MFCC7_std", "MFCC9_std", "MFCC10_std",
            "MFCC11_std", "Chroma7", "Chroma8"
        ],
        "feature_importance": [
            {"name": "MFCC2_mean", "value": 0.1620}, {"name": "MFCC10_std", "value": 0.1105}, 
            {"name": "MFCC2_std", "value": 0.0891}, {"name": "MFCC1_std", "value": 0.0877}, 
            {"name": "MFCC11_std", "value": 0.0815}, {"name": "MFCC4_std", "value": 0.0721},
            {"name": "MFCC3_mean", "value": 0.0650}, {"name": "MFCC7_std", "value": 0.0585}, 
            {"name": "MFCC1_mean", "value": 0.0521}, {"name": "MFCC7_mean", "value": 0.0498}, 
            {"name": "MFCC9_std", "value": 0.0475}, {"name": "MFCC5_mean", "value": 0.0391},
            {"name": "Chroma8", "value": 0.0542}, {"name": "Chroma7", "value": 0.0415}, 
            {"name": "MFCC6_mean", "value": 0.0293}
        ]
    },
    "Particle Swarm Optimization": {
        "accuracy": 0.74,
        "precision_healthy": 0.70,
        "recall_healthy": 0.88,
        "precision_pd": 0.85,
        "recall_pd": 0.53,
        "features_selected": 14,
        "selected_features": [
            "MFCC1_mean", "MFCC2_mean", "MFCC3_mean", "MFCC4_mean",
            "MFCC6_mean", "MFCC8_mean", "MFCC1_std", "MFCC2_std",
            "MFCC3_std", "MFCC6_std", "MFCC10_std", "MFCC12_std",
            "Chroma6", "Chroma8"
        ],
        "feature_importance": [
            {"name": "MFCC2_mean", "value": 0.1710}, {"name": "MFCC10_std", "value": 0.1078}, 
            {"name": "MFCC2_std", "value": 0.0921}, {"name": "MFCC1_std", "value": 0.0907}, 
            {"name": "MFCC12_std", "value": 0.0865}, {"name": "MFCC3_std", "value": 0.0711},
            {"name": "MFCC4_mean", "value": 0.0560}, {"name": "MFCC6_std", "value": 0.0545}, 
            {"name": "MFCC1_mean", "value": 0.0511}, {"name": "MFCC8_mean", "value": 0.0428}, 
            {"name": "MFCC3_mean", "value": 0.0455}, {"name": "MFCC6_mean", "value": 0.0361},
            {"name": "Chroma8", "value": 0.0682}, {"name": "Chroma6", "value": 0.0625}
        ]
    }
}

ALL_FEATURES = [
    'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'MFCC4_mean', 'MFCC5_mean', 'MFCC6_mean', 
    'MFCC7_mean', 'MFCC8_mean', 'MFCC9_mean', 'MFCC10_mean', 'MFCC11_mean', 'MFCC12_mean', 'MFCC13_mean',
    'MFCC1_std', 'MFCC2_std', 'MFCC3_std', 'MFCC4_std', 'MFCC5_std', 'MFCC6_std',
    'MFCC7_std', 'MFCC8_std', 'MFCC9_std', 'MFCC10_std', 'MFCC11_std', 'MFCC12_std', 'MFCC13_std',
    'Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth',
    'ZCR', 'RMS', 'Spectral_Contrast', 'Spectral_Flatness', 'Tempo',
    'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6',
    'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12'
]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Parkinson's Detection API"}

@app.get("/api/results")
async def get_results():
    return ALGORITHM_RESULTS

@app.get("/api/features")
async def get_features():
    return {"features": ALL_FEATURES}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize predictor
    predictor_instance = get_predictor()
    if predictor_instance is None or predictor_instance.model is None:
        await websocket.send_json({"error": "Model not loaded on server"})
        await websocket.close()
        return

    try:
        while True:
            # Receive audio data as binary (assuming float32 or int16 from frontend)
            data = await websocket.receive_bytes()
            
            # Convert received bytes to numpy array
            # Assuming frontend sends float32 mono audio chunks after normalization
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Add to predictor buffer
            predictor_instance.process_audio_chunk(audio_chunk)
            
            # Get prediction
            result = predictor_instance.predict_live()
            
            if result:
                # Send result back to frontend
                await websocket.send_json(result)
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        finally:
            await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
