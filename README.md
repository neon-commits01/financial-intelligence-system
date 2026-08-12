# Financial Intelligence System

A production-style financial news sentiment analysis API built with
**FastAPI**, **TF-IDF**, and a **fine-tuned FinBERT model**.

The project takes financial news headlines as input and classifies their
sentiment as:

-   `positive`
-   `negative`
-   `neutral`

The API exposes both individual and batch prediction endpoints and is
deployed publicly using **Google Cloud Run**.

------------------------------------------------------------------------

## Why this project?

Financial news moves quickly, and manually reviewing large numbers of
headlines is not practical.

This project explores a simple but useful question:

> Can we automatically triage financial news headlines by sentiment,
> while keeping a lightweight traditional ML baseline alongside a
> domain-specific transformer model?

The project therefore uses two models:

1.  **TF-IDF + Linear SVM** as the traditional NLP baseline.
2.  **Fine-tuned FinBERT** as the domain-specific transformer model.

This makes the project useful not only as an API, but also as a
comparison between classical NLP and modern transformer-based NLP.

------------------------------------------------------------------------

## Architecture

``` text
                    ┌─────────────────────┐
                    │       Client        │
                    │ Browser / Postman   │
                    │ Python / Application│
                    └──────────┬──────────┘
                               │ HTTPS
                               ▼
                    ┌─────────────────────┐
                    │     Google Cloud    │
                    │       Run           │
                    │                     │
                    │      FastAPI        │
                    │         │           │
                    │    ┌────┴────┐      │
                    │    ▼         ▼      │
                    │ TF-IDF    FinBERT   │
                    │  Model      Model   │
                    └─────────┬───────────┘
                              │
                              ▼
                           Response
```

### Deployment pipeline

``` text
GitHub
   │
   │ push to main
   ▼
Cloud Build
   │
   │ builds Docker image
   ▼
Artifact Registry
   │
   │ container image
   ▼
Cloud Run
   │
   ▼
Public HTTPS API
```

Cloud Build is configured to automatically build and deploy the
application when changes are pushed to the repository.

------------------------------------------------------------------------

## Models

### 1. TF-IDF baseline --- v1

The baseline represents each headline using **TF-IDF** features and
performs classification using a linear Support Vector Machine.

Why keep a traditional model?

-   Fast inference
-   Small and relatively lightweight
-   Easy to understand and debug
-   Strong baseline for sparse text classification
-   Useful for comparing classical NLP against transformers

**Model name:** `TF-IDF`\
**Version:** `v1`

------------------------------------------------------------------------

### 2. Fine-tuned FinBERT --- v2

FinBERT is a BERT-based model designed for financial language.

The model was fine-tuned for the project's three-class sentiment
classification task.

It is expected to perform better when financial context matters because
words and phrases can have different meanings in financial news than in
general language.

**Model name:** `Fine-tuned FinBERT`\
**Version:** `v2`

The fine-tuned FinBERT model achieved approximately **88% macro-F1**
during evaluation.

------------------------------------------------------------------------

## Dataset

The project uses the **Financial PhraseBank** dataset for financial
sentiment classification.

The dataset contains financial statements and news-style phrases
labelled as:

-   Positive
-   Negative
-   Neutral

The data was used to train and evaluate the project's sentiment models.

------------------------------------------------------------------------

## API

### Base URL

The deployed API is available at:

`https://financial-intelligence-system-871346793913.asia-south1.run.app`

### API documentation

FastAPI automatically provides interactive API documentation.

-   `/docs` --- Swagger UI
-   `/redoc` --- ReDoc documentation

### Model information

``` http
GET /model-info
```

Returns information about the models currently exposed by the API.

------------------------------------------------------------------------

### Single prediction

``` http
POST /predict
```

Classifies a single financial headline.

Example request:

``` json
{
  "text": "The company reported stronger-than-expected quarterly earnings."
}
```

Example response:

``` json
{
  "prediction": "positive"
}
```

------------------------------------------------------------------------

### Batch prediction

``` http
POST /predict-batch
```

Classifies multiple headlines in one request.

Example request:

``` json
{
  "texts": [
    "The company reported record quarterly revenue.",
    "The bank announced a significant decline in profits.",
    "The company maintained its outlook for the year."
  ]
}
```

Example response:

``` json
{
  "predictions": [
    "positive",
    "negative",
    "neutral"
  ]
}
```

The API also exposes model/version information for batch predictions.

------------------------------------------------------------------------

## Local Development

### 1. Clone the repository

``` bash
git clone https://github.com/neon-commits01/financial-intelligence-system.git
cd financial-intelligence-system
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file containing the environment variables required
by the application.

Do **not** commit `.env` to GitHub.

The application reads configuration using environment variables rather
than hard-coding deployment-specific values.

### 5. Run the API

``` bash
uvicorn app.main:app --reload
```

The API will be available locally through the address printed by
Uvicorn.

Interactive documentation will be available at:

``` text
/docs
```

------------------------------------------------------------------------

## Docker

The application is containerized using Docker.

### Build the image

``` bash
docker build -t financial-intelligence-system .
```

### Run the container

``` bash
docker run --env-file .env -p 8000:8000 financial-intelligence-system
```

The Docker image contains the application and its Python dependencies,
allowing the same application environment to be reproduced across
machines.

------------------------------------------------------------------------

## Google Cloud Deployment

The production API is deployed using **Google Cloud Run**.

The deployment flow is:

``` text
GitHub
   ↓
Cloud Build
   ↓
Docker image
   ↓
Artifact Registry
   ↓
Cloud Run
```

### Why Cloud Run?

Cloud Run was chosen because it provides:

-   Managed container execution
-   HTTPS endpoint
-   Automatic scaling
-   Scale-to-zero behavior
-   No need to manage a virtual machine
-   Simple deployment from a Git repository

The service uses **request-based billing** and can scale down when there
is no traffic.

------------------------------------------------------------------------

## Environment Variables

Environment-specific configuration is kept outside the source code.

For example, the application reads the FinBERT model configuration using
an environment variable:

``` text
FINBERT_MODEL_ID
```

The actual values should be supplied through the deployment environment
and should not be committed to the repository.

For local development, these values can be placed in `.env`.

For cloud deployment, configure them through the Cloud Run service
configuration.

------------------------------------------------------------------------

## Project Structure

``` text
financial-intelligence-system/
│
├── app/
│   ├── main.py
│   ├── inference_finbert.py
│   └── inference_tfidf.py
│
├── models/
│   └── ...
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

The exact contents may evolve as the project develops.

------------------------------------------------------------------------

## Testing

The API can be tested using:

-   FastAPI Swagger UI
-   Postman
-   `curl`
-   Python HTTP clients
-   Any application capable of making HTTPS requests

Both individual and batch prediction endpoints can be used to compare
model behaviour.

------------------------------------------------------------------------

## What I learned from the project

This project was built to understand the complete path from an ML model
to a usable cloud API.

Key concepts covered include:

-   Text preprocessing
-   TF-IDF vectorization
-   Linear SVM classification
-   Transformer-based NLP
-   FinBERT
-   Model inference
-   FastAPI
-   REST APIs
-   Environment variables
-   Docker images and containers
-   Dockerfile-based builds
-   Cloud Build
-   Artifact Registry
-   Google Cloud Run
-   Continuous deployment from GitHub
-   Cloud scaling and request-based billing

The main goal was not just to train a model, but to understand how an ML
model becomes an actual service that another application can call.

------------------------------------------------------------------------

## Future Improvements

Potential next steps include:

-   Add more detailed API validation
-   Add automated tests
-   Add monitoring and structured logging
-   Add model performance monitoring
-   Improve error handling
-   Add more financial NLP tasks
-   Experiment with additional transformer models
-   Add authentication if the API becomes a private service
-   Improve CI/CD checks before deployment

------------------------------------------------------------------------

## Tech Stack

  Component             Technology
  --------------------- --------------------------
  Language              Python
  API                   FastAPI
  Traditional NLP       TF-IDF
  Baseline classifier   Linear SVM
  Transformer           FinBERT
  Containerization      Docker
  Source control        Git / GitHub
  Build / CI/CD         Google Cloud Build
  Container registry    Google Artifact Registry
  Deployment            Google Cloud Run

------------------------------------------------------------------------

## Status

**Deployed and working.**

The API is publicly accessible through Google Cloud Run and currently
supports both single-headline and batch sentiment prediction.

------------------------------------------------------------------------

## License

This project is intended primarily as a learning and portfolio project.
