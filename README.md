# 🏠 **California House Price Prediction using XGBoost Regressor**

This project predicts median house values in California districts using the California Housing dataset from Scikit-learn and an XGBoost Regressor. 
The goal is to understand the key factors affecting housing prices and build a high-performance prediction pipeline.


## 🚀 Live Application

👉 Direct View / Try the App:
https://mlhousepricepredictionusingxgboost-regression-mrkukigkjkzsxnyb.streamlit.app/
Built and deployed using Streamlit.

## 📌 Project Overview

In this project, we:

 - Load and preprocess the dataset

-  Handle missing values

- Create derived features (e.g., RoomsPerHousehold, BedroomsPerRoom)

- Analyze relationships between features and the target

- Split the dataset into training and testing sets

- Train an XGBoost Regressor for price prediction

- Evaluate model performance using MAE and R²

- Build a prediction pipeline for scoring new unseen data

- Deploy the model as an interactive Streamlit web application

## 📂 Dataset

- Source: Scikit-learn – California Housing Dataset

- Instances: ~20,640 samples

- Features (8):

 - MedInc

  - HouseAge

  - AveRooms

  - AveBedrms

  - Population

  - AveOccup

  - Latitude

    - Longitude

- Target:

   - MedHouseVal (Median house value in $100,000 units)

## 🛠️ Technologies Used

- Python 3.x

- Pandas / NumPy – Data manipulation & numerical computation

- Matplotlib / Seaborn – Data visualization

- Scikit-learn – Preprocessing, train-test split, evaluation

- XGBoost – Regression model building

- Streamlit – Web application deployment

## 📊 Model Performance
- Metric	Score
- MAE	0.23
- R² Score	0.83

The model demonstrates strong predictive capability with high variance explanation (83%).

## 📈 Visualizations

- Feature distribution plots

- Correlation heatmap

- Predicted vs Actual comparison plot

- Feature importance chart (from XGBoost)

These visualizations help interpret the drivers behind housing prices.

## 🧭 Workflow
House Price Data
       ↓
        
Data Preprocessing

        ↓
        
Exploratory Data Analysis

        ↓
        
Train-Test Split

        ↓
        
XGBoost Regressor

        ↓
        
Evaluation (MAE, R²)

        ↓
        
Visualization

        ↓
        
Streamlit Web App Deployment

## 💻 Streamlit Web Application

The project includes a fully styled dark-themed interactive app with:

- Real-time price prediction

- Dynamic market tier classification (Affordable / Mid-Range / Premium)

- Model performance metrics display

- Clean Golden Amber UI design

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV or Optuna

- Try ensemble stacking for improved accuracy

- Expand deployment with Docker containerization
