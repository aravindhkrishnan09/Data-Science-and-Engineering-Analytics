# EV Battery Health Prediction - RAG System

https://ev-battery-rag-1023905756100.us-central1.run.app

This directory contains the **Retrieval-Augmented Generation (RAG)** system for the EV Battery Health Prediction project. It serves as an intelligent AI assistant capable of answering complex questions about the project's methodology, datasets, models, and results by grounding its responses in the actual project documentation.

## 🚀 Overview

The RAG system leverages Google Cloud's **Vertex AI Search** as a knowledge base and **Gemini 2.5 Flash** as the reasoning engine. By connecting the LLM to a specific Data Store containing project files (notebooks, reports, datasets), the assistant provides factually accurate information with direct citations.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: Google Gemini 2.5 Flash (via Vertex AI)
- **Retrieval/Grounding**: Vertex AI Search (Data Store)
- **Language**: Python 3.10+
- **Environment Management**: `python-dotenv`

## ✨ Key Features

- **Grounded Q&A**: Answers are generated based on the provided project documents, minimizing hallucinations.
- **Source Attribution**: Provides a "View Sources" section in the UI to show exactly where the information was retrieved from.
- **Chat History**: Maintains session-based conversation history for a seamless user experience.
- **Optimized Parameters**: Configured with `temperature: 0.0` to ensure high factual consistency.

## 📂 File Structure

- `main.py`: The core Streamlit application.
- `main_check.ipynb`: A diagnostic notebook used to verify connectivity with the Google Cloud Discovery Engine (Vertex AI Search) and the specific Data Store ID.

## ⚙️ Setup & Installation

### Prerequisites
1.  **Google Cloud Project**: An active GCP project with Vertex AI and Discovery Engine APIs enabled.
2.  **Vertex AI Search Data Store**: A Data Store populated with the project's documentation (e.g., the `.ipynb` and `.docx` files from the `jupyter_notebooks` folder).
3.  **Authentication**: Ensure your local environment is authenticated with Google Cloud (e.g., via `gcloud auth application-default login`).

### Configuration
Create a `.env` file in this directory with the following keys:
```env
PROJECT_ID=your-google-cloud-project-id
MODEL_REGION=us-central1 # or your preferred region
DATA_STORE_LOCATION=global
DATA_STORE_ID=your-vertex-ai-search-datastore-id
```

PROJECT_ID="data-science-479011"
DATA_STORE_LOCATION="global"
MODEL_REGION="us-central1"
DATA_STORE_ID="capstone-project-ds_1767691287743"

### Running the Application
1.  Navigate to the RAG directory:
    ```bash
    cd Capstone_Project/RAG
    ```
2.  Install dependencies (if not already done in the root):
    ```bash
    pip install streamlit google-cloud-aiplatform python-dotenv
    ```
3.  Launch the app:
    ```bash
    streamlit run main.py
    ```