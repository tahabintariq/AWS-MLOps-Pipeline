# Automated MLOps Environment and Pipeline on AWS

## Overview

This project demonstrates the implementation of a persistent and automated machine learning environment using AWS infrastructure. It covers the full lifecycle of setting up a cloud-based ML workstation, from disk management and cloud storage integration to automated environment provisioning and a model training pipeline.

---

## Infrastructure Architecture

- **Compute:** AWS EC2 Ubuntu Instance  
- **Storage:** 10GB Secondary EBS Volume formatted as `ext4` and mounted at `/mnt/ml-data`  
- **Persistence:** Volume auto-mount configuration via `/etc/fstab` to ensure data survival across reboots  
- **Cloud Integration:** Amazon S3 bucket with versioning for dataset management and backups  
- **Security:** IAM Instance Profile for secure AWS CLI authentication without the use of permanent access keys  

---

## Component Breakdown

### 1. Environment Automation (`setup_ml_env.sh`)

An idempotent bash script that handles the initial system setup:

- Updates system packages  
- Installs Python3, pip, and virtualenv  
- Creates a Python virtual environment directly on the persistent EBS volume  
- Installs necessary ML libraries (`pandas`, `numpy`, `scikit-learn`, `joblib`)  
- Designed to be re-run safely without causing configuration conflicts  

---

### 2. Data Management

- **Dataset:** UCI Adult Census dataset  
- **S3 Integration:** One-way synchronization between the local EBS volume and S3 cloud storage  
- **Versioning:** S3 versioning enabled to track changes in raw and processed data over time  

---

### 3. Machine Learning Pipeline (`train_pipeline.py`)

A standalone Python script that executes a standardized training workflow:

- **Data Loading:** Reads data from the persistent `/mnt/ml-data` directory  
- **Preprocessing:** Applies Label Encoding to targets and Standard Scaling to numeric features  
- **Model Selection:** Trains two classifiers:
  - Logistic Regression  
  - Random Forest  
- **Automation:** Automatically selects the best-performing model based on accuracy  
- **Output:**  
  - Saves the serialized best model (`.pkl`)  
  - Logs performance metrics with timestamps to the persistent volume  

---

### 4. Operational Automation

- **Cost Control:** Scheduled auto-shutdown mechanism using `cron` to minimize AWS costs  
- **Data Persistence:** Validation that datasets, models, and logs remain accessible after instance restarts  

---

## Directory Structure

```
/mnt/ml-data/
│
├── datasets/      # Raw and processed data files
├── models/        # Serialized model binaries
├── logs/          # Performance metrics and execution logs
└── ml_venv/       # Isolated Python environment
```

---

## Execution Instructions

1. Run the bootstrap script:
   ```bash
   bash setup_ml_env.sh
   ```

2. Activate the environment:
   ```bash
   source /mnt/ml-data/ml_venv/bin/activate
   ```

3. Execute the training pipeline:
   ```bash
   python3 train_pipeline.py
   ```

4. Verify results:
   - `/mnt/ml-data/models/`
   - `/mnt/ml-data/logs/`
