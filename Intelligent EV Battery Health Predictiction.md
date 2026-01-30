# Intelligent EV Battery Health Prediction System

## Project Overview
**Strategic Objective:**  
Predictive Maintenance for Electric Vehicle (EV) batteries through **State of Health (SoH) estimation** and **Remaining Useful Life (RUL) forecasting**.

**Core Challenges Addressed:**
- Complex `.mat` file structures  
- High-frequency sensor noise  
- Non-linear battery degradation dynamics  

---

## Project Links
- **Live Dashboard:**  
  [EV Battery Health Prediction App](https://ev-battery-health-prediction-app-v2.streamlit.app/)

- **RAG AI Assistant:**  
  [EV Battery RAG Assistant](https://ev-battery-rag-1023905756100.us-central1.run.app/)

- **Code & Notebooks:**  
  [GitHub Repository](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Capstone_Project/README.md)

- **Project Presentation:**  
  [Gamma Presentation](https://ev-battery-health-predic-z4pcfbk.gamma.site/)

---

## End-to-End Predictive Maintenance Architecture

### Dataset & Coverage
- NASA Prognostics Center of Excellence (PCoE) dataset  
- 2.1M+ sensor measurements  
- 636 charge–discharge cycles

### Data Engineering Layer
- Nested `.mat` parsing using SciPy
- Hierarchical data flattening
- Analytical Pandas DataFrames

### Feature Engineering
- **Cycle-Level Aggregates:** Mean voltage, current, temperature, capacity fade indicators
- **Physics-Informed Features:**
  - Voltage Drop (ΔV)
  - Temperature Rise (ΔT)
  - Internal degradation signals

### Scalable Data Pipeline
- Automated AWS S3 ingestion
- CSV & Parquet storage
- Dockerized ETL container

---

## Predictive Modeling & Learning Strategies

### Baseline Models
- Random Forest
- XGBoost
- Non-linear regression benchmarks

### Deep Learning Models
- LSTM
- GRU (sequential time-series modeling)

### Temporal Learning Design
- Sliding windows  
- Sequence-to-one SoH prediction  
- Multi-cycle temporal context  

### Optimization Parameters
- Time steps: 30  
- Activation: Tanh
- Hidden Units: 128 / 64
- Training Epochs: 100

### Complementary Learning
- K-Means clustering for aging stage classification  
- Q-Learning for charging policy optimization

---

## Model Performance & Comparative Analysis

### Primary Accuracy Target
- SoH prediction within **±5% absolute error on unseen batteries**

### Results Summary

| Model | R² | MAE |
|--------|------|------|
| Random Forest | ~0.94 | 1.70% |
| XGBoost | 0.9276 | 1.76% |
| GRU (Best) | — | 1.50% (B0018) |

### Cross-Battery Validation
- GRU R²:  
  - B0007 → 0.79  
  - B0018 → 0.92  

### Quantified Improvement
- GRU MAE: 2.90%  
- LSTM MAE: 5.73%  
- **~50% error reduction**

---

## Impact, Reliability & Practical Significance
- Exceeded ±5% SoH prediction accuracy target
- Early degradation detection capability
- Reduced risk of unplanned battery failures
- Physics-informed features improved model trust
- SHAP-based interpretability for decision transparency

---

## GenAI-Augmented Intelligence Layer
- RAG architecture deployed on Google Cloud Run  
- Core stack:
  - Vertex AI Search  
  - Gemini 2.5 Flash
  - Dockerized inference services
- Grounded responses with citation-backed reasoning

---

## Deployment, Monitoring & Explainability
- Streamlit dashboard deployment
- Docker containerized services
- Cloud-native hosting
- Real-time SoH & RUL predictions
- CSV upload and AWS S3 integration
- SHAP feature attribution for explainability

---

## Knowledge Systems & Automation

### Reference Materials
- [ML Reference Handbook](https://docs.google.com/document/d/124kgXhx3bXfq7GbmAUBpu5sGk1QKvqUau468rDnHRVk/edit?tab=t.bwyqhzzhi4q9)

### Custom AI Agents
- [AI Agents Repository](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/tree/main/AI)

### Automation Stack
- Jupyter Notebook → Quarto → Gamma  
- Docker-based reporting pipelines  

---

## Final Takeaways
- Physics-guided features + advanced time-series models  
- ~50% MAE reduction vs LSTM on complex degradation patterns  
- Production-grade architecture  
- Interpretable ML with GenAI augmentation  

---

# Other Key Projects

## General Data Analysis
- [Summary](https://engineering-materials-da-wus3wrn.gamma.site/)  
- [README](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Projects/Main_Project_Data_Analysis/README.md)  
- [Notebook](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Projects/Main_Project_Data_Analysis/Engineering_Materials__Project_AravindhG_JupyterNotebook.ipynb)

---

## EV Charging Patterns
- [Summary](https://ev-charging-demand-forec-5hppct2.gamma.site/)  
- [README](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Projects/Main_Project_ML/EV%20Charging%20Patterns/README.md)  
- [Notebook](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Projects/Main_Project_ML/EV%20Charging%20Patterns/EV_Charging_Patterns.ipynb)

---

## Vehicle Energy Dataset (VED) Analysis
- [Summary](https://vehicle-energy-dataset-a-6knezqt.gamma.site/)  
- [README](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Projects/Main_Project_ML/VED%20Analysis/README.md)  
- [Notebook](https://github.com/aravindhkrishnan09/Data-Science-and-Engineering-Analytics/blob/main/Projects/Main_Project_ML/VED%20Analysis/VED_ML.ipynb)

---
