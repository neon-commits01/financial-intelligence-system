import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import joblib

load_dotenv()
model_id= os.getenv("TFIDF_MODEL_ID")
if not model_id:
    raise RuntimeError("TFIDF_MODEL_ID environment variable is not set")


model_file = hf_hub_download(
    repo_id=model_id,
    filename="tfidf_linear_svm.joblib"
)

# Let's load the TFIDF model
model = joblib.load(model_file)

# Model metadata
MODEL_INFO = {
    "name": "TF-IDF",
    "version": "v1",
    "classes": ("positive", "negative", "neutral")
}
 
# a quick check - did the model load?
def is_model_loaded():
    return model is not None



def predict(text):
    sentiment = model.predict([text])[0]
    return {
        "sentiment" : sentiment
    }

def predict_batch(texts):
    sentiments = model.predict(texts).tolist()
    return {
        "sentiment": sentiments
    }


