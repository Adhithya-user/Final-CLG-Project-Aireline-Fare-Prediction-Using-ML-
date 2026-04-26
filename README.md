# ✈️ Airline Fare Prediction — ML Project
### II-Year M.Sc. Data Science | Annamalai University | R. Adhithya | 2025-26

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask app
python app.py

# 3. Open browser
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
flight_project/
├── app.py                                  ← Flask web application
├── Adhithya_project_Fare_of_Airlines.ipynb ← Jupyter notebook (complete ML pipeline)
├── Data_Train.xlsx                         ← Training dataset (10,683 records)
├── requirements.txt                        ← Python dependencies
├── README.md
├── model/
│   ├── random_forest_model.pkl             ← Random Forest (best model)
│   ├── decision_tree_model.pkl             ← Decision Tree
│   ├── xgboost_model.pkl                   ← XGBoost
│   ├── gradient_boost_model.pkl            ← Gradient Boosting
│   ├── best_model.pkl                      ← Best model alias (Random Forest)
│   └── feature_columns.pkl                 ← Feature column names
└── templates/
    └── index.html                          ← Web UI template
```

---

## 📊 Model Performance

| Algorithm | R² Score | MAE (₹) | RMSE (₹) | MAPE (%) |
|-----------|----------|---------|---------|---------|
| 🌲 **Random Forest** | **81.5%** | ₹1,251 | ₹1,931 | 14.0% |
| 🌿 Decision Tree | 79.1% | ₹1,253 | ₹2,053 | 13.8% |
| ⚡ XGBoost | 83.8% | ₹1,176 | ₹1,805 | 13.2% |
| 🚀 Gradient Boosting | 83.6% | ₹1,230 | ₹1,819 | 14.0% |

---

## 🌐 REST API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "IndiGo",
    "source": "Delhi",
    "destination": "Cochin",
    "stops": "1 stop",
    "dep_hour": 9, "dep_min": 0,
    "arr_hour": 13, "arr_min": 30,
    "dur_hours": 4, "dur_mins": 30,
    "journey_day": 15, "journey_month": 5
  }'
```

---

## ✅ Features (18 input variables)

- `Airline` — Target-guided ordinal encoded (0-11)
- `Destination` — Target-guided ordinal encoded (0-4)
- `Total_Stops` — Label encoded (0=non-stop, 1=1 stop, ...)
- `Journey_day`, `Journey_month` — Extracted from date
- `Dep_Time_hour`, `Dep_Time_minute` — Extracted from departure time
- `Arrival_Time_hour`, `Arrival_Time_minute` — Extracted from arrival time
- `Duration_hours`, `Duration_mins` — Parsed from duration string
- `Source_Banglore`, `Source_Kolkata`, `Source_Delhi`, `Source_Chennai`, `Source_Mumbai` — One-hot encoded
