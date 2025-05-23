# 🛒 Grocery Sales Time Series Analysis and Forecasting

This project builds a machine learning model to forecast daily item sales across stores in the **Guayas region** of Ecuador using real-world data from [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) on Kaggle. The goal was to predict future demand and optimize inventory, using a blend of classical and modern machine learning techniques.

> ⏳ **Project Duration:** 4 weeks  
> 🎯 **Forecast Period:** January – March 2014  
> 🧪 **Model Type:** XGBoost Regressor  
> 🏆 **Best Model Performance:**  
> • MAE: **0.0562**  
> • MAPE: **1.66%**  
> • R² Score: **0.8379**

---

## 📁 Project Structure

```bash
.
├── model-data.csv
├── time-series-forecast-eda.ipynb          # Data exploration, cleaning & initial feature engineering 
├── time-series-forecast-naive-ml.ipynb     # Baseline & naive forecast models 
├── time-series-forecast-classical-ml.ipynb # ARIMA model 
├── time-series-forecast-xgboost.ipynb      # Final model development and evaluation 
├── xgh_model.pkl                           # Trained XGBoost model (saved with Pickle) 
└── README.md
```
---

## 📦 Data Sources

From the [Kaggle competition dataset](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data):

| File | Description |
|------|-------------|
| `train.csv` | Daily unit sales per item per store |
| `items.csv` | Metadata on item family, class, and perishability |
| `stores.csv` | Store metadata: city, state, cluster |
| `transactions.csv` | Daily transactions count per store |
| `oil.csv` | Daily oil prices (Ecuador is oil-dependent) |
| `holidays_events.csv` | Local/national holidays & their characteristics |
| `onpromotion` | Binary flag indicating promotional activity per item/store/day |

---

## 📊 EDA & Feature Engineering

Key steps included:

- **Filtering to Guayas region** and selecting top 3 product families: `GROCERY I`, `BEVERAGES`, and `CLEANING`
- Handling **missing dates** and filling zero sales
- Transforming `unit_sales` to deal with extreme outliers
- Generating time-based features (day of week, month, holidays)
- Adding lag-based and rolling mean features
- Merging datasets (oil prices, holiday events, promotions, etc.)

---

## 🧠 Models Compared

| Model              | R² Score | MAPE    | Notes                            |
|-------------------|----------|---------|----------------------------------|
| Naive Baseline     | -0.4914  | 8.09%   | Predicts same as previous day    |
| Holt-Winters       | -0.2834  | 7.05%   | Predicts same as previous day    |
| ARIMA              | ~0.22    | ~4.70%  | Limited performance              |
| XGBoost (default)  | 0.7934   | 2.08%   | Strong improvement over baseline |
| **XGBoost (tuned)**| **0.8379** | **1.66%** | Best performing model            |

---

## 📈 Results: Forecast vs Actual

![Forecast Plot](https://github.com/DanMontHell/Time-Series-Forecast-Masterschool/blob/main/time_series_prediction.png)

---

## 🔍 Feature Importance

The most predictive features according to the tuned XGBoost model:

![Feature Importance](https://github.com/DanMontHell/Time-Series-Forecast-Masterschool/blob/main/feature_importance.png)

---

## 🧰 Tools & Skills

**Languages & Libraries**  
`Python` · `pandas` · `NumPy`  
`scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn`

**Core Skills Demonstrated**  
- 📊 **Time Series Analysis**: trends, lags, rolling features  
- 🔍 **EDA & Feature Engineering**: merging datasets, handling NaNs, outlier removal  
- 🧠 **Machine Learning**: regression, model tuning, error evaluation  
- 🧪 **Model Evaluation**: MAPE, MAE, R² with visual and numeric tracking  
- 💾 **Model Persistence**: Saved with `pickle` for reuse  
- 📁 **Project Modularity**: Jupyter notebooks + utility script  
- 🧹 **Efficient Workflow**: filtered 2M-row subset for reproducibility

---

## ✅ Next Steps

- Add a web interface using Streamlit for interactive forecasts
- Explore recurrent models like LSTM or Prophet for longer-term horizon
- Improve hyperparameter tuning with cross-validation and Bayesian search
- Expand to include more item families or regions

---

## 📌 Summary

This project demonstrates a full time series forecasting pipeline from raw data to deployment-ready model, applying best practices in feature engineering, model selection, and interpretability.

If you’d like to collaborate, have feedback, or want to see more of my work, feel free to connect on [LinkedIn](https://www.linkedin.com/in/your-link-here/) or explore other [projects on GitHub](https://github.com/DanMontHell).
