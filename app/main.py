# 1. Let's get the file path of the root first and then get to the model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from . import inference_finbert, inference_tfidf


# We need FastAPI application
app = FastAPI(
    title="Financial Intelligence System API",
    version="2.0"
)

class ModelType(Enum):
    FINBERT = "finbert"
    TFIDF = "tfidf"

# Model Handlers
MODEL_HANDLERS = {
    ModelType.FINBERT: inference_finbert,
    ModelType.TFIDF: inference_tfidf
}

# Model output classes
CLASSES = ("positive", "negative", "neutral")


# --------- API Endpoints ---------------
# To get the status of the API
@app.get("/health")
def health_check():
    return{
        "status": "ok",
        "models": {
            f"{model_type.value}_loaded": handler.is_model_loaded()
            for model_type, handler in MODEL_HANDLERS.items()
        }
        
    }
 
# To get the model info
@app.get("/model-info")
def model_info():
    return{
        "default_model": ModelType.FINBERT.value,
        "classes" : CLASSES,
        "models": {
            model_type.value:{
                "name":  handler.MODEL_INFO["name"],
                "version": handler.MODEL_INFO["version"],
                "loaded": handler.is_model_loaded()
            }
            for model_type, handler in MODEL_HANDLERS.items()
        }

    }

# To predict a text, we need the text from the user -> so we need to validate the data before predicting
class PredictRequest(BaseModel):
    text: str = Field(...,min_length=1,max_length=2000, description="Financial news headline")
    model: ModelType = ModelType.FINBERT

class PredictResponse(BaseModel):
    sentiment:str
    model_name:str
    model_version:str


# Post request for predict
@app.post("/predict", response_model=PredictResponse)
def predict(request:PredictRequest):
    # strip starting and trailing spaces
    text = request.text.strip()
    # empty text validation
    if not text:
        raise HTTPException(status_code=400,
                            detail="Input text cannot contain empty string")
    # handler holds the model which is selected
    handler = MODEL_HANDLERS[request.model]
    model_prediction = handler.predict(text)["sentiment"]
         
 
    
    return{
        "sentiment": model_prediction,
        "model_name": handler.MODEL_INFO["name"],
        "model_version": handler.MODEL_INFO["version"]
    }


# Now, we need to do the same for Batch of texts

class BatchPredictRequest(BaseModel):
    texts:list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of Financial news headline"
    )
    model: ModelType = ModelType.FINBERT

class BatchPredictResponse(BaseModel):
    predictions: list[str]
    model_name: str
    model_version: str

# Post request for BatchPrediction
@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(request:BatchPredictRequest):
    cleaned_texts = [text.strip() for text in request.texts]
    # Check if there are any empty texts in the list, if it even has one raise an error
    if any (not text for text in cleaned_texts):
        raise HTTPException(
            status_code=400,
            detail="Input texts cannot be empty strings"
        )
    handler = MODEL_HANDLERS[request.model]
    model_predictions = handler.predict_batch(cleaned_texts)["sentiment"]
    
 

    return{
        "predictions": model_predictions,
        "model_name": handler.MODEL_INFO["name"],
        "model_version": handler.MODEL_INFO["version"]
    }




