#from pathlib import Path
 
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch  


#PROJECT_ROOT = Path(__file__).resolve().parents[1]
# model_path = os.getenv("FINBERT_MODEL_PATH")
# if not model_path:
#     raise RuntimeError("FINBERT_MODEL_PATH environment variable is not set")
# MODEL_PATH = Path(model_path)
load_dotenv()
model_id = os.getenv("FINBERT_MODEL_ID")
if not model_id:
    raise RuntimeError("FINBERT_MODEL_ID environment variable is not set")


# Since the argmax gives us the index of the highest probab, we need to map to the label
index2label = {
        0: "positive",
        1: "negative",
        2: "neutral"
}

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_id)
 # changes the model's behaviour to eval mode, but it still computes the gradients
model.eval()

# Model metadata
MODEL_INFO = {
    "name": "Fine-tuned FinBERT",
    "version": "v2",
    "classes": ("positive", "negative", "neutral")
}

# let's check if the model loaded well
def is_model_loaded():
    return model is not None





def predict(text):
   
    # here, the values of each key should be converted to the tensors
    encoded_input = tokenizer(text, truncation=True, return_tensors="pt")

    # tells PyTorch to not track gradient-related computation
    with torch.inference_mode():
        # convert the key-value pairs into named params and pass them
        outputs = model(**encoded_input)
    # converting the logist to probabilities
    probabs = torch.softmax(outputs.logits, dim=-1)
    # now we shall pick the highest one, coz that is the model prediction -> returns index
    index = torch.argmax(probabs, dim=-1).item()
    sentiment = index2label[index]
    confidence = probabs[0][index].item()
    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
        
def predict_batch(texts):
    encoded_batch_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.inference_mode():
      output = model(**encoded_batch_input)
    probabs = torch.softmax(output.logits,dim=-1)
    indices = torch.argmax(probabs,dim=-1).tolist()
    sentiments = [index2label[index] for index in indices]
    confidence_scores = []
    for i in range(len(indices)):
        score = probabs[i][indices[i]].item()
        confidence_scores.append(score)
    

    return {
        "sentiment": sentiments,
        "confidence": confidence_scores
    }
    
