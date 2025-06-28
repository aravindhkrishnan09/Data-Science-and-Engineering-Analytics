
# Vehicle Energy Dataset (VED) - Machine Learning Project

This repository contains code and scripts for processing, analyzing, and storing Vehicle Energy Dataset (VED) files using Python, AWS S3, and Google Drive. The project focuses on combining static and dynamic vehicle data and preparing it for machine learning tasks such as regression and clustering.

---

## 📦 Features

- Load and preprocess VED static data (ICE, HEV, PHEV, EV types).
- Sample and concatenate dynamic CSV files from Google Drive or AWS S3.
- Handle missing values and column renaming.
- Merge static datasets and upload processed files to AWS S3 in CSV and Parquet formats.
- Authenticate and access files from both Google Drive and AWS S3 securely.

---

## 📁 Data Sources

### Static Data:
The VED dataset contains dynamic and static data collected over a year from vehicles operating in Ann Arbor, Michigan. It includes time-series data on speed, location, battery parameters, fuel rates, and more. This dataset enables analysis of energy consumption, driving patterns, and the development of machine learning models.
GitHub Repository: https://github.com/gsoh/VED

### Dynamic Data:
- Weekly driving datasets stored in:
  - Google Drive folders
  - AWS S3 bucket: `s3aravindh973515031797`

---

## 🛠️ Technologies Used

- **Python** (Pandas, NumPy, os)
- **AWS S3 (boto3)** for cloud storage
- **Google Drive (PyDrive)** for cloud access
- **dotenv** for secure credentials handling

---

## 📊 Data Processing Workflow

1. **Static Load:**
   - Read Excel files into Pandas DataFrames.
   - Replace `'NO DATA'` with `NaN`.
   - Merge ICE, HEV, PHEV, and EV data into a unified `df_static`.

2. **Dynamic Load:**
   - Sample 50% of weekly driving data CSVs from either local directory, Google Drive, or AWS S3.
   - Concatenate into a unified DataFrame `df_part1`.

3. **Upload to Cloud:**
   - Upload static data to AWS S3 as `.csv` and `.parquet`.
   - Supports uploading dynamic data similarly.

## 🚀 How to Run

1. **Set AWS credentials** in a `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_DEFAULT_REGION=your_region
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy boto3 python-dotenv pydrive
   ```

3. **Run the scripts**:
   - Load and clean static data
   - Authenticate and pull dynamic data from S3/Drive
   - Upload processed files back to cloud

---

## 📧 Contact

Project developed by [Aravindh G]  
For any issues or queries, please open an issue or reach out via GitHub.

---
