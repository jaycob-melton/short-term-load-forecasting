# Short-Term Load Forecasting with Weather & Time Features

This project builds a short-term electricity **load forecasting model** using publicly available system demand data and weather forecasts. Models used are Linear Regression, Ridge Regression, Random Forests, XGBoost, and LSTM Neural Networks.

## Project Goal
Predict **hourly load** using historical load, weather variables (temperature, cloud cover), and calendar features (day of week, holiday, etc.).

---

## Dependencies
- Python (3.8)
- Jupyter
- pandas, numpy, scikit-learn, xgboost, pytorch
- ospd load and weather data
- Visualizations with matplotlib/seaborn

---

## Folder Structure
load-forecasting-mini-project/ 

├── data/ # Raw + cleaned datasets, add time_series_60min.csv and weather_data.csv from https://open-power-system-data.org/

├── notebooks/ # EDA and modeling notebooks 

├── src/ # Data loaders, feature engineering, modeling 

├── env.yml # Conda environment 

├── README.md
