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

Financial news is full of context-specific language, so sentiment classification is not always as simple as looking for positive or negative words.

I started with a classical NLP baseline using **TF-IDF + Linear SVM**, then upgraded the system with **FinBERT** to see how much a financial-domain Transformer could improve the results.

The project therefore has two model versions:

- **V1:** TF-IDF + Linear SVM
- **V2:** Fine-tuned FinBERT

The main goal was not just to train a better model, but to take the model all the way from experimentation to a working API and cloud deployment.

---

## V1 → V2

### V1 — TF-IDF baseline

TF-IDF was used to convert financial headlines into numerical features, followed by a Linear SVM classifier.

It provides a useful baseline because it is:

- lightweight
- fast
- easy to interpret
- strong for traditional text classification

**V1 results:**

- Accuracy: **74.41%**
- Macro-F1: **70.77%**

### V2 — Fine-tuned FinBERT

FinBERT is a BERT-based model designed for financial language.

Instead of using a general-purpose language model, I fine-tuned FinBERT specifically for the three-class financial sentiment task.

This allows the model to learn patterns from the financial dataset rather than relying only on its original pre-trained knowledge.

**V2 results:**

- Accuracy: **88.94%**
- Macro-F1: **88.31%**

### Comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Linear SVM (V1) | 74.41% | 70.77% |
| FinBERT — zero-shot | 88.10% | 87.23% |
| Fine-tuned FinBERT (V2) | **88.94%** | **88.31%** |

Fine-tuned FinBERT is the strongest model in this experiment.

Compared with the TF-IDF baseline, V2 improved:

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
