🏠 **California House Price Prediction using XGBoost Regressor**
This project predicts median house values in California districts using the California Housing dataset from Scikit-learn and an XGBoost Regressor. 
The goal is to understand the key factors affecting housing prices and build a high-performance prediction pipeline.

📌 Project Overview
In this project, we:

- Load and preprocess the dataset (handle missing values, feature scaling if needed, create new features like RoomsPerHousehold, BedroomsPerRoom).

- Analyze the data to understand relationships between features and the target.

- Split the dataset into training and testing sets for model evaluation.

- Train an XGBoost Regressor to predict house prices.

- Evaluate model performance using metrics like MAE,and R².

- Build a prediction pipeline for scoring new, unseen data.

📂 Dataset

- Source: Scikit-learn California Housing Dataset

- Instances: ~20,640 samples

- Features (8): MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

Target: MedHouseVal (Median house value in $100,000 units)

🛠️ Technologies Used

- Python 3.x

- Pandas / NumPy – data manipulation and analysis

- Matplotlib / Seaborn – data visualization

- Scikit-learn – data preprocessing, train-test split, evaluation metrics

- XGBoost – regression model building and tuning

📊 Model Performance Metric Score:

- MAE: 0.23

- R² Score: 0.83

📈 Visualizations

Feature distributions to understand variable spread.

Correlation heatmap to identify feature relationships.

Predicted vs. Actual plot to assess prediction accuracy.

Feature importance chart from XGBoost to interpret key drivers of price.

🧭 Workflow
**House Price Data → Data Preprocessing → Data Analysis → Train-Test Split → XGBoost Regressor → Evaluation → Visualization**

🔮 Future Improvements

- Perform hyperparameter tuning using GridSearchCV/Optuna for optimal performance.

- Add geospatial features (distance to ocean, amenities, schools).

- Try ensemble methods (stacking multiple models) for further accuracy gains.

- Deploy as a Streamlit or Gradio web app for easy public access.
