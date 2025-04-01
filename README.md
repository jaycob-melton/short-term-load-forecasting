# Short-Term Load Forecasting with Weather & Time Features

This project builds a short-term electricity **load forecasting model** using publicly available system demand data and weather forecasts.

## Project Goal
Predict **next-day hourly load** using historical load, weather variables (temperature, cloud cover), and calendar features (day of week, holiday, etc.).

---

## Skills Highlighted
- Time series feature engineering (lags, calendar effects)
- Model training & evaluation (baseline, linear regression, random forest, XGBoost)
- Weather API integration with `meteostat`
- Error analysis with MAE, RMSE, residual plots
- Scalable pipeline design using modular Python scripts
- Presentation-ready analysis notebook and slide deck

---

## Tech Stack
- Python (3.10)
- JupyterLab
- pandas, scikit-learn, xgboost
- meteostat for weather data
- Visualizations with matplotlib/seaborn

---

## 📁 Folder Structure
load-forecasting-mini-project/ 
├── data/ # Raw + cleaned datasets 
├── notebooks/ # EDA and modeling notebooks 
├── src/ # Data loaders, feature engineering, modeling 
├── reports/ # PDF summary for interview 
├── env.yml # Conda environment 
├── README.md