# Deployment Guide: EV Battery Health RAG System on Google Cloud Run

This guide outlines the steps to containerize and deploy the Streamlit-based RAG application to Google Cloud Run.

## 1. Prerequisites

*   **Google Cloud Project**: An active GCP project.
*   **Vertex AI Search**: A Data Store set up and populated with your project documents.
*   **Google Cloud SDK**: Installed and authenticated on your local machine (`gcloud auth login`).

## 2. Enable Required APIs

Run the following command to enable the necessary services in your project:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com discoveryengine.googleapis.com aiplatform.googleapis.com
```

## 3. Prepare Configuration Files

Ensure the following files exist in your `Capstone_Project/RAG/` directory:

### `requirements.txt`
Contains the Python dependencies.

### `Dockerfile`
Defines the container environment. It is configured to run Streamlit on port 8080.

### `.dockerignore`
Prevents local environment files and caches from being uploaded to the cloud.

## 4. Deploy to Google Cloud Run

Navigate to the `Capstone_Project/RAG/` directory in your terminal and run the following single-line command. 

**Note:** Replace `your-project-id` and `your-datastore-id` with your actual Google Cloud values.

```bash
gcloud run deploy ev-battery-rag --source . --region us-central1 --allow-unauthenticated --set-env-vars "PROJECT_ID=your-project-id,MODEL_REGION=us-central1,DATA_STORE_LOCATION=global,DATA_STORE_ID=your-datastore-id"
```

## 5. Configure Permissions (IAM)

Crucial step: The Cloud Run service needs permission to talk to the AI APIs.

1.  Go to the **IAM & Admin** page in the GCP Console.
2.  Locate the service account used by your Cloud Run service (usually the **Default Compute Service Account** or one ending in `@serverless-robot-prod.iam.gserviceaccount.com`).
3.  Click the edit icon (pencil) for that member and add the following roles:
    *   **Vertex AI User**
    *   **Discovery Engine Viewer**
4.  Save the changes.

## 6. Access the Application

Once the deployment is successful, the terminal will output a **Service URL**. 

To find the URL in the Google Cloud Console:
1. Go to the [Cloud Run console](https://console.cloud.google.com/run).
2. Click on your service name (**ev-battery-rag**).
3. The URL is displayed at the top of the service details page.

## 7. Updating the App

Whenever you make changes to `main.py` or other files, simply run the command in **Step 4** again. Cloud Run will build a new version and update the URL automatically.