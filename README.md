\# House Price Prediction (MLOps Project)



\## 📌 Overview



This project predicts house prices using machine learning.

It is built with an end-to-end MLOps pipeline including training, API, and Docker deployment.



\## ⚙️ Technologies Used



\* Python

\* Scikit-learn

\* FastAPI

\* Docker



\## 📊 Dataset



Dataset used from Kaggle: House Prices Dataset



\## 🚀 Features



\* Data preprocessing

\* Machine learning model training

\* REST API for prediction

\* Docker containerization



\## ▶️ How to Run



\### Run locally:



python src/train.py

python -m uvicorn app.main:app --reload



\### Run using Docker:



docker build -t house-price-app .

docker run -p 8000:8000 house-price-app



\## 🌐 API Endpoint



\* POST /predict → returns predicted house price



\## 📌 Author



Saad Pawaskar



