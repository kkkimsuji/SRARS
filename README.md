# SRARS: Utilizing Advanced Transformers for Summarized Review-Aware Recommender System

[![Paper](https://img.shields.io/badge/IEEE_Access-Paper-blue)](https://doi.org/10.1109/ACCESS.2025.3598937)
[![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2025.3598937-red)](https://doi.org/10.1109/ACCESS.2025.3598937)


This repository contains the official implementation of the following paper:
> **Kim, S.**, Jin, L., Li, Q., Li, X., Yoon, C., & Kim, J. (2025). Utilizing Advanced Transformers for Summarized Review-Aware Recommender System. IEEE Access. 

## Overview

This repository provides the official implementation of **SRARS (Summarized Review-Aware Recommender System)**, a review-aware recommendation model based on summarized review information. SRARS applies **BART-based abstractive summarization** to generate concise review representations, models high-order user-item interactions via an **outer product**, and refines them using **multi-head self-attention**. Experiments on three Amazon review datasets demonstrate that SRARS outperforms strong baselines in terms of **MAE** and **RMSE**.

## Environment & Requirements

This project is implemented in **Python 3.8+**. To ensure reproducibility, please install the specific versions of the libraries listed below.

### 1. Key Dependencies
| Category | Library | Minimum Version | Description |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | `TensorFlow` | `2.10.0` | Main model architecture & training |
| **NLP** | `Transformers` | `4.25.0` | BART-based feature extraction |
| **NLP Backend** | `PyTorch` | `1.12.0` | Backend for HuggingFace Transformers |
| **Analysis** | `Pandas` | `1.5.0` | Data manipulation and storage |
| **Matrix** | `NumPy` | `1.23.0` | Efficient numerical operations |
| **ML Tools** | `scikit-learn` | `1.1.0` | Data splitting and metrics |

### 2. Utility Libraries
- `PyYAML` (>=6.0): For parsing the `config.yaml` file.
- `PyArrow` (>=10.0.0): For high-performance Parquet file handling.
- `tqdm` (>=4.64.0): For real-time progress bars during embedding extraction.

### 3. Installation
We recommend using a virtual environment. You can install all dependencies at once using the following command:

```bash
pip install -r requirements.txt
```
## Repository Structure

The repository is organized as follows to ensure a clear workflow from data preprocessing to model evaluation:

```text
SRARS/
├── data/
│   ├── raw/                # Original datasets (e.g., SampleData.json.gz)
│   └── processed/          # Preprocessed files (pickles, parquets, embeddings)
├── model/
│   ├── proposed.py         # SRARS model architecture (Self-Attention)
│   └── srars_best_model.h5 # Saved best model weights after training
├── src/
│   ├── bart.py             # Feature extraction using BART encoder
│   ├── config.yaml         # Centralized hyperparameters and path settings
│   ├── data_loader.py      # Data cleaning and review-set generation
│   └── trainer.py          # Training pipeline (split, train, evaluate)
├── main.py                 # Main entry point to run the full pipeline
├── requirements.txt        # List of Python dependencies
├── .gitignore              # Git ignore configuration
└── README.md               # Project documentation and results
```
## Model Description

**SRARS (Summarized Review-Aware Recommender System)** is composed of three modules: **review feature extraction**, **interaction learning**, and **rating prediction**. 

First, user and item review sets are summarized with pretrained BART to obtain compact feature representations. These summarized vectors are then projected through an MLP. 

Next, SRARS models user-item relationships using an outer product, which captures high-order interactions across embedding dimensions. A multi-head self-attention layer is applied to the resulting interaction map to highlight informative signals and suppress less useful patterns. 

Finally, the refined interaction representation is passed through an MLP to predict the final rating score. The model is optimized with MSE loss using the Adam optimizer.

<img width="820" height="454" alt="image" src="https://github.com/user-attachments/assets/f9b2308a-311b-4b82-9c7e-49ba77a4fd97" />

## Experimental Results

The table below summarizes the performance comparison between **SRARS** and baseline models on three Amazon review datasets.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">Musical Instruments</th>
      <th colspan="2">Industrial and Scientific</th>
      <th colspan="2">Video Games</th>
    </tr>
    <tr>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NCF</td>
      <td>0.835</td>
      <td>1.102</td>
      <td>0.828</td>
      <td>1.116</td>
      <td>0.955</td>
      <td>1.203</td>
    </tr>
    <tr>
      <td>D-attn</td>
      <td>0.797</td>
      <td>0.998</td>
      <td>0.769</td>
      <td>0.992</td>
      <td>0.867</td>
      <td>1.078</td>
    </tr>
    <tr>
      <td>DeepCoNN</td>
      <td>0.730</td>
      <td>1.004</td>
      <td>0.716</td>
      <td>1.012</td>
      <td>0.797</td>
      <td>1.066</td>
    </tr>
    <tr>
      <td>NARRE</td>
      <td>0.716</td>
      <td>0.997</td>
      <td>0.692</td>
      <td>0.986</td>
      <td>0.782</td>
      <td>1.043</td>
    </tr>
    <tr>
      <td>AENAR</td>
      <td>0.703</td>
      <td>1.017</td>
      <td>0.660</td>
      <td>0.996</td>
      <td>0.755</td>
      <td>1.058</td>
    </tr>
    <tr>
      <td><b>SRARS</b></td>
      <td><b>0.657</b></td>
      <td><b>0.991</b></td>
      <td><b>0.621</b></td>
      <td><b>0.978</b></td>
      <td><b>0.729</b></td>
      <td><b>1.040</b></td>
    </tr>
  </tbody>
</table>

SRARS achieved the best performance across all datasets in terms of both **MAE** and **RMSE**.

