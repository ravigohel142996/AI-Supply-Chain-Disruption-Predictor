"""FastAPI REST API for supply chain disruption prediction."""
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.utils.logger import log
from src.utils.config import config
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor, FeatureEngineering
from src.models.trainer import ModelTrainer
from src.models.predictor import RiskPredictor


# Initialize FastAPI app
app = FastAPI(
    title="AI Supply Chain Disruption Predictor API",
    description="REST API for predicting supply chain disruptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_trainer = None
predictor = None


# Pydantic models for request/response
class FeatureInput(BaseModel):
    """Input features for prediction."""
    order_value: float = Field(..., description="Order value in dollars")
    shipping_distance: float = Field(..., description="Shipping distance in km")
    lead_time: float = Field(..., description="Lead time in days")
    supplier_reliability_score: float = Field(..., ge=0, le=1, description="Supplier reliability (0-1)")
    inventory_level: float = Field(..., description="Current inventory level")
    demand_forecast: float = Field(..., description="Demand forecast")
    weather_risk_index: float = Field(..., ge=0, le=1, description="Weather risk index (0-1)")
    shipping_mode: str = Field(..., description="Shipping mode")
    supplier_region: str = Field(..., description="Supplier region")
    product_category: str = Field(..., description="Product category")
    season: str = Field(..., description="Season")
    carrier: str = Field(..., description="Carrier name")


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int
    delay_probability: float
    risk_level: int
    risk_category: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_count: int
    high_risk_count: int
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool


# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "AI Supply Chain Disruption Predictor API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/predict",
            "/batch-predict",
            "/model/info"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_trainer is not None and model_trainer.model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: FeatureInput):
    """
    Make a single prediction.
    
    Args:
        features: Input features
        
    Returns:
        Prediction with risk assessment
    """
    global predictor
    
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = features.dict()
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        result = predictor.predict_with_risk(df)
        
        return PredictionResponse(
            prediction=int(result['prediction'].iloc[0]),
            delay_probability=float(result['delay_probability'].iloc[0]),
            risk_level=int(result['risk_level'].iloc[0]),
            risk_category=str(result['risk_category'].iloc[0])
        )
    
    except Exception as e:
        log.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Make batch predictions from uploaded CSV file.
    
    Args:
        file: CSV file with features
        
    Returns:
        Batch predictions
    """
    global predictor
    
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Load data
        data_ingestion = DataIngestion()
        df = data_ingestion.upload_handler(file)
        
        # Make predictions
        results = predictor.predict_with_risk(df)
        
        # Convert to response format
        predictions = []
        for _, row in results.iterrows():
            predictions.append(PredictionResponse(
                prediction=int(row['prediction']),
                delay_probability=float(row['delay_probability']),
                risk_level=int(row['risk_level']),
                risk_category=str(row['risk_category'])
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        high_risk_count = sum(1 for p in predictions if p.risk_category in ['High', 'Critical'])
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            high_risk_count=high_risk_count,
            processing_time=processing_time
        )
    
    except Exception as e:
        log.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=dict)
async def model_info():
    """Get information about the loaded model."""
    global model_trainer
    
    if model_trainer is None or model_trainer.model is None:
        return {"status": "no_model_loaded"}
    
    return {
        "model_type": model_trainer.model_type,
        "metrics": model_trainer.metrics,
        "feature_names": model_trainer.feature_names
    }


@app.post("/model/load")
async def load_model(model_path: str):
    """
    Load a trained model.
    
    Args:
        model_path: Path to model file
    """
    global model_trainer, predictor
    
    try:
        model_trainer = ModelTrainer()
        model_trainer.load_model(model_path)
        predictor = RiskPredictor(model_trainer.model)
        
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    
    except Exception as e:
        log.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Setup logger
    from src.utils.logger import app_logger
    app_logger.setup()
    
    # Run server
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload
    )
