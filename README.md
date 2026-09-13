# 🚀 High-Performance C++ Inference Engine & Automated MLOps Pipeline

## 📖 Overview
This project solves two major problems in enterprise Machine Learning: **Inference Latency** and **Model Degradation (Data Drift)**. 

Instead of wrapping ML models in slow Python web frameworks, this project executes an ONNX-serialized model purely in C++ memory for maximum throughput. Furthermore, it features a self-healing MLOps pipeline that continuously logs production data, detects statistical drift, and automatically triggers GitHub Actions to retrain the model.

## 🏗️ System Architecture
1. **The C++ API (`main.cpp`):** A custom REST API built with `cpp-httplib` and the `ONNX Runtime C++ API`. It receives live HTTP requests, parses URL parameters, executes the AI prediction, and logs telemetry data to a local CSV database.
2. **The ML Bridge (`train.py`):** Trains a Random Forest/XGBoost model in Python and serializes it to the universal `.onnx` format so C++ can load it natively.
3. **The Drift Detective (`monitor.py`):** A Python script that reads the C++ telemetry logs and compares live production distributions against original training baselines.
4. **The CI/CD Automation (`mlops.yml`):** A GitHub Actions workflow that runs the monitor automatically. If data drift is detected, it triggers the training script to generate and deploy a new model.

## 🛠️ Tech Stack
* **C++14 / cpp-httplib:** Web server routing & memory management.
* **Microsoft ONNX Runtime:** Cross-platform ML model execution.
* **Python (Pandas, Scikit-Learn):** Model training and drift detection.
* **GitHub Actions:** CI/CD pipeline automation.
