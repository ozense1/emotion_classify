# Emotion Classification with Classical, Deep Learning, and Transformer Models

This project uses the [Emotion Dataset](https://www.kaggle.com/datasets/abdallahwagih/emotion-dataset) from Kaggle, published by Abdallah Wagih.  
This repository contains an end-to-end workflow for **text emotion classification** on short comments. 
The target is to predict one of three emotions:

- `anger`
- `fear`
- `joy`

The project explores:

- **Classical ML models** with TF-IDF features
- **Deep learning models** based on pretrained Word2Vec embeddings
- **Transformer-based models** (BERT, DistilBERT, RoBERTa, ALBERT, MiniLM)
- **Model interpretability** using SHAP for the best classical model

---

# Model Performance Summary 
This section compares all trained models across **classical**, **deep learning**, and **transformer-based** approaches, using **F1-macro** as the main evaluation metric.

## Classical ML Models (TF-IDF)

| Model                 | F1-Macro |
|----------------------|----------|
| **Logistic Regression** | **0.9376** |
| Linear SVM           | 0.9359 |
| Random Forest        | 0.9283 |
| Naive Bayes          | 0.9210 |
| XGBoost              | 0.9199 |

Best classical model: **Logistic Regression (0.9376)**  
Consistent, strong baseline with SHAP interpretability.

---

## Deep Learning Models (Word2Vec + PyTorch)

| Model            | F1-Macro |
|------------------|----------|
| Attention-BiLSTM (tuned) | **0.9605** |
| Attention-BiLSTM | 0.9571 |
| BiLSTM           | 0.9302 |
| Kim-CNN          | 0.9288 |
| LSTM             | 0.1679 |
| GRU              | 0.1641 |

Best deep learning model: **Tuned Attention-BiLSTM (0.9605)**  
LSTM and GRU underperform classical and transformer methods due to dataset size and static Word2Vec embeddings.

---

## Transformer Models (HuggingFace)

| Model            | F1-Macro |
|------------------|----------|
| **RoBERTa-base**   | **0.9722** |
| BERT-base        | 0.9697 |
| DistilBERT       | 0.9688 |
| MiniLM           | 0.9672 |
| ALBERT           | 0.9663 |

Best transformer: **RoBERTa-base (0.9722)**  
This is the top-performing model across all experiments.

---
##  How to Run the Project

### 1. **Clone the Repo & Prepare the Data**
- Place **`Emotion_classify_Data.csv`** in the project root.
- *(Optional but recommended)* Run **`0_preprocessing_eda.ipynb`** to inspect data quality and generate:
  - `cleaned_emotion_classify_data.csv`
  - `output_suspicious_words.csv`

### 2. **Manual Cleaning (optional)**
- Open **`output_suspicious_words.csv`** and inspect suspicious or unknown tokens.
- Fix issues (typos, mislabeled entries, slang normalization, etc.) directly in the dataset.
- Save the final cleaned dataset as **`manual_cleaned_emotion_classify_data.csv`**, which must include:
  - `Comment_clean`
  - `Emotion`

### 3. **Run Classical ML Models**
- Open **`1_classical_model.ipynb`**.
- Ensure **`manual_cleaned_emotion_classify_data.csv`** is available in the root.
- Run the notebook to:
  - Train and evaluate classical ML models (LogReg, SVM, RF, NB, XGBoost)
  - Generate `classical_model_macro_results.csv`
  - Explore SHAP feature explanations for Logistic Regression

### 4. **Run Deep Learning Models**
- Download **GoogleNews Word2Vec** and update the `w2v_path` variable.
- Open **`2_deep_learning_model.ipynb`**.
- Run the notebook to:
  - Train LSTM, BiLSTM, CNN-BiLSTM, and Attention-BiLSTM models
  - Save results to `deep_learning_model_results.csv`*

### 5. **Run Transformer Models**
- Open **`3_transformer.ipynb`**.
- Ensure the following are installed:
  - `transformers`
  - `torch`
  - GPU support *(recommended but optional)*
- Run the notebook to fine-tune:
  - **BERT-base**, **DistilBERT**, **RoBERTa-base**, **ALBERT**, **MiniLM**
- Save results to **`transformer_model_results.csv`**

---


## Future Work

### **1. Expand the Set of Emotions**
The current project focuses on three emotions: **joy**, **fear**, and **anger**.  
However, human emotions are far more nuanced.  Expanding the emotional categories would allow the model to capture richer emotional expression—especially useful for applications like mental-health monitoring, chatbots, and social-media analysis.

### **2. Handling Code-Switching & Multilingual Input**
Social-media users frequently mix languages (code-switching). Currently, the preprocessing pipeline treats non-English tokens as **typos** and discards them.
Future work could include:
- Multilingual transformers (e.g., XLM-R, mBERT)
- Language-aware tokenization
- Retaining and embedding non-English tokens
- Detecting and modeling mixed-language patterns

Improving multilingual handling makes the system more inclusive and far more representative of real conversational data.

---
