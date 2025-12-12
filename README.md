
# Predictive Bus Stop Ridership Model

This repository contains a **Python re‑implementation** of the analytical workflow originally developed in R for understanding and predicting **low‑ridership bus stops** at TriMet (Portland, OR).

The core script is [`predictive_model_full.py`](predictive_model_full.py).  
It reproduces the entire R Markdown pipeline: principal‑component analysis (PCA), k‑means clustering, network‑centrality joins, a logistic regression classifier, and 5‑fold cross‑validation.

---

## Project Structure

```
.
├── combined_data_no_prop.csv          # Monthly ridership metrics (source data)
├── stops_routes_w_centrality.csv      # Stop‑level network‑centrality measures
├── stops_land_use.csv                 # Land‑use & socioeconomic attributes
├── predictive_model_full.py           # End‑to‑end translated Python workflow
└── README.md                          # You are here
```

> ## Main Dataset
The dataset is too large to upload to GitHub.  
You can download it here: [Download dataset](https://drive.google.com/file/d/1S-j4hxxQO96_JRGX5DBZW0GlsWb0dcWJ/view?usp=sharing)

After downloading, place the dataset in the same folder as the script before running.

---

## What the Model Does

1. **Data Preparation**  
   * Adds a `total_riders` column (`TOTAL_ONS + TOTAL_OFFS`).  
   * Keeps stops with **complete 12‑month history** and replaces structural zeros with ones to avoid log/scale issues.

2. **Dimensionality Reduction (PCA)**  
   * Applies a log‑transform and z‑score to continuous predictors.  
   * Computes principal components and uses the **first 8 PCs** for clustering.

3. **Clustering (k‑means)**  
   * Searches k = 2‑10 via **average silhouette width** to find the optimal number of clusters.  
   * Assigns each stop to a cluster (`cluster`).

4. **Feature Enrichment**  
   * Merges **network‑centrality** metrics (`degree`, `betweenness`, `closeness`) and **land‑use / demographic** variables.  
   * Flags **low‑ridership** stops (bottom 25 % within each cluster).

5. **Modeling**  
   * Fits a **logistic regression** predicting `low_ridership` with:
     * Demographics & land‑use predictors  
     * Network‑centrality measures  
     * Fixed effects for `cluster`, `month`, and optional `ZONEGEN_CL`  
   * Outputs coefficients, diagnostics, and an ROC curve.

6. **Validation**  
   * Performs **5‑fold cross‑validation** at the stop level, reporting mean AUC and accuracy.

---

## Requirements

| Package | Tested Version |
| ------- | -------------- |
| Python  | 3.9+           |
| pandas  | 2.x            |
| numpy   | 1.26           |
| scikit‑learn | 1.4        |
| statsmodels | 0.16        |
| seaborn | 0.13           |
| matplotlib | 3.9          |

Install with:

```bash
pip install -r requirements.txt
```

*(Generate `requirements.txt` via `pip freeze > requirements.txt` after creating your environment.)*

---

## Running the Pipeline

```bash
python predictive_model_full.py
```

The script prints:

* Cumulative variance explained by PCs  
* Optimal *k* by silhouette width  
* Logistic‑regression summary  
* Full‑sample AUC  
* Per‑fold CV AUC & accuracy  

It also shows two plots (scree plot & ROC curve).

---
