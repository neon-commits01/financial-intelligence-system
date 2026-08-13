# Fin Intelligence System v1

A financial news sentiment classification API built to compare a traditional NLP baseline with a domain-specific Transformer model.

The system takes a financial headline and predicts:

- `positive`
- `negative`
- `neutral`

**Live API:**  
https://financial-intelligence-system-871346793913.asia-south1.run.app/docs

You can test the deployed API directly through the Swagger UI — no setup required.

---

## Why this project?

Financial news contains a lot of domain-specific language, so sentiment classification is not always as simple as looking for positive or negative words.

I started with a classical NLP baseline using **TF-IDF + Linear SVM**, then upgraded the system with **FinBERT** to see how much a financial-domain Transformer could improve the results.

The project therefore has two model versions:

- **V1:** TF-IDF + Linear SVM
- **V2:** Fine-tuned FinBERT

The goal was not just to train a model, but to take it through the complete workflow from experimentation to a working API and cloud deployment.

---

## V1 → V2

### V1 — TF-IDF baseline

TF-IDF was used to convert financial headlines into numerical features, followed by a Linear SVM classifier.

It provides a useful baseline because it is lightweight, fast, and effective for traditional text classification.

**V1 results:**

- Accuracy: **74.41%**
- Macro-F1: **70.77%**

### V2 — Fine-tuned FinBERT

FinBERT is a BERT-based model designed for financial language.

Instead of relying only on a general-purpose language model, I fine-tuned FinBERT specifically for the three-class financial sentiment task.

**V2 results:**

- Accuracy: **88.94%**
- Macro-F1: **88.31%**

### Comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Linear SVM (V1) | 74.41% | 70.77% |
| FinBERT — zero-shot | 88.10% | 87.23% |
| Fine-tuned FinBERT (V2) | **88.94%** | **88.31%** |

Compared with the TF-IDF baseline, the fine-tuned FinBERT model improved:

- Accuracy by **14.53 percentage points**
- Macro-F1 by **17.54 percentage points**

The project also includes error analysis to understand where the models succeed and where they still make mistakes.

---

## Dataset

The project uses the **Financial PhraseBank** dataset for financial sentiment classification.

The target classes are:

```text
positive
negative
neutral
```

The same task was used to evaluate the classical baseline and the FinBERT models, making the comparison meaningful.

---

## Architecture

```text
                    Client
                      │
                     HTTPS
                      ▼
              Google Cloud Run
                      │
                   FastAPI
                  /       \
                 ▼         ▼
             TF-IDF     FinBERT
               V1          V2
                 \         /
                  ▼       ▼
                    Response
```

The models are hosted separately on Hugging Face rather than being stored in the Git repository.

```text
GitHub
  │
  │ source code
  ▼
Cloud Build
  │
  ▼
Docker Image
  │
  ▼
Google Cloud Run
  │
  ├── TF-IDF V1
  │      ↓
  │   Hugging Face
  │
  └── FinBERT V2
         ↓
      Hugging Face
```

---

## API

### Try it yourself

**Swagger UI:**  
https://financial-intelligence-system-871346793913.asia-south1.run.app/docs

Open the link, choose an endpoint, click **Try it out**, enter a headline, and click **Execute**.

### Single prediction

```http
POST /predict
```

Example:

```json
{
  "text": "The company reported stronger-than-expected quarterly earnings."
}
```

The API returns the predicted sentiment and model information.

### Batch prediction

```http
POST /predict-batch
```

Example:

```json
{
  "texts": [
    "The company reported record quarterly revenue.",
    "The bank announced a significant decline in profits.",
    "The company maintained its outlook for the year."
  ]
}
```

This allows multiple headlines to be classified in one request.

### Model information

```http
GET /model-info
```

Returns information about the models currently exposed by the API.

---

## Models

The trained models are hosted on Hugging Face:

**TF-IDF V1**

`neonbit01/tfidf-financial-sentiment-v1`

**Fine-tuned FinBERT V2**

`neonbit01/finbert-finetuned-v2`

The application loads the models using environment variables rather than hard-coding model locations.

```text
FINBERT_MODEL_ID=neonbit01/finbert-finetuned-v2
TFIDF_MODEL_ID=neonbit01/tfidf-financial-sentiment-v1
```

---

## Running locally

### Clone the repository

```bash
git clone https://github.com/neon-commits01/financial-intelligence-system.git
cd financial-intelligence-system
```

### Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file:

```text
FINBERT_MODEL_ID=neonbit01/finbert-finetuned-v2
TFIDF_MODEL_ID=neonbit01/tfidf-financial-sentiment-v1
```

Do not commit `.env` to GitHub.

### Run the API

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Docker

The application is containerized using Docker.

Build:

```bash
docker build -t fin-intelligence-system .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 fin-intelligence-system
```

The same containerized application is used for the Cloud Run deployment.

---

## Project Structure

```text
fintel-system-v1/
│
├── app/
│   ├── main.py
│   ├── inference_finbert.py
│   └── inference_tfidf.py
│
├── notebooks/
│   ├── 03_baseline_models.ipynb
│   ├── 04_finbert_inference_baseline.ipynb
│   ├── 05_finbert_error_analysis.ipynb
│   └── 06_finbert_finetuning.ipynb
│
├── reports/
│   └── model_comparison_v1_vs_v2.csv
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

The notebooks document the progression from the baseline experiments through FinBERT inference, error analysis, and fine-tuning.

---

## What this project covers

This project covers all of these:

- NLP preprocessing
- TF-IDF
- Linear SVM
- Transformer inference
- FinBERT fine-tuning
- Model evaluation
- Error analysis
- FastAPI
- Docker
- Hugging Face model hosting
- Git/GitHub
- Google Cloud Build
- Google Artifact Registry
- Google Cloud Run

The main takeaway was learning how to move from an ML experiment to an actual service that can be accessed over the internet.

---

## Status

**Deployed and working.**

The current production API runs on Google Cloud Run and exposes both the V1 TF-IDF baseline and V2 fine-tuned FinBERT model.
