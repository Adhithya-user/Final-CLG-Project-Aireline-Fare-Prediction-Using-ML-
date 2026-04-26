"""
✈️  Airline Fare Prediction — Flask Web App
Run:   python app.py
Open:  http://127.0.0.1:5000
"""

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib, os, json

app = Flask(__name__)

# ── Load all 4 models ─────────────────────────────────────────────────────────
MODEL_DIR = 'model'

models = {
    'Random Forest':     joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl')),
    'Decision Tree':     joblib.load(os.path.join(MODEL_DIR, 'decision_tree_model.pkl')),
    'XGBoost':           joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl')),
    'Gradient Boosting': joblib.load(os.path.join(MODEL_DIR, 'gradient_boost_model.pkl')),
}
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

# ── Encoding maps — MUST match notebook pipeline exactly ──────────────────────
# From:  airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index
AIRLINE_ENC = {
    'Trujet': 0, 'SpiceJet': 1, 'Air Asia': 2, 'IndiGo': 3,
    'GoAir': 4, 'Vistara': 5, 'Vistara Premium economy': 6,
    'Air India': 7, 'Multiple carriers': 8,
    'Multiple carriers Premium economy': 9,
    'Jet Airways': 10, 'Jet Airways Business': 11,
}

# From:  dest = data.groupby(['Destination'])['Price'].mean().sort_values().index
# Note: 'New Delhi' was replaced with 'Delhi' before encoding
DEST_ENC = {
    'Kolkata': 0, 'Hyderabad': 1, 'Delhi': 2,
    'Banglore': 3, 'Cochin': 4, 'New Delhi': 2,
}

SOURCES = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
STOPS   = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

# ── Performance metrics (from training) ───────────────────────────────────────
METRICS = {
    'Random Forest':     {'r2': 81.5, 'mae': 1251, 'rmse': 1931, 'mape': 14.0},
    'Decision Tree':     {'r2': 79.1, 'mae': 1253, 'rmse': 2053, 'mape': 13.8},
    'XGBoost':           {'r2': 83.8, 'mae': 1176, 'rmse': 1805, 'mape': 13.2},
    'Gradient Boosting': {'r2': 83.6, 'mae': 1230, 'rmse': 1819, 'mape': 14.0},
}
BEST_MODEL  = max(METRICS, key=lambda k: METRICS[k]['r2'])
BEST_TRAIN_R2 = 95.2   # Train R² of Random Forest
BEST_TEST_R2  = max(m['r2'] for m in METRICS.values())


def build_features(d: dict) -> pd.DataFrame:
    """Convert form/JSON input → model-ready DataFrame using exact notebook encoding."""
    airline = d.get('airline', 'IndiGo')
    source  = d.get('source',  'Delhi')
    dest    = d.get('destination', 'Cochin')
    stops_str = d.get('stops', 'non-stop')

    dep_h  = int(d.get('dep_hour',   9))
    dep_m  = int(d.get('dep_min',    0))
    arr_h  = int(d.get('arr_hour',  11))
    arr_m  = int(d.get('arr_min',   30))
    dur_h  = int(d.get('dur_hours',  2))
    dur_m  = int(d.get('dur_mins',  30))
    j_day  = int(d.get('journey_day',   15))
    j_month= int(d.get('journey_month',  5))

    stops_int = STOPS.get(stops_str, 0) if isinstance(stops_str, str) else int(stops_str)

    row = {
        'Airline'            : AIRLINE_ENC.get(airline, 3),
        'Destination'        : DEST_ENC.get(dest, 2),
        'Total_Stops'        : stops_int,
        'Journey_day'        : j_day,
        'Journey_month'      : j_month,
        'Dep_Time_hour'      : dep_h,
        'Dep_Time_minute'    : dep_m,
        'Arrival_Time_hour'  : arr_h,
        'Arrival_Time_minute': arr_m,
        'Duration_hours'     : dur_h,
        'Duration_mins'      : dur_m,
        'Duration_hour'      : dur_h,    # duplicate column from notebook
        'Duration_minute'    : dur_m,    # duplicate column from notebook
    }
    # One-hot Source columns
    for s in SOURCES:
        row[f'Source_{s}'] = 1 if source == s else 0

    df = pd.DataFrame([row])
    # Align to exact training columns (fill any missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template(
        'index.html',
        airlines=list(AIRLINE_ENC.keys()),
        sources=SOURCES,
        destinations=[k for k in DEST_ENC if k != 'New Delhi'],
        stops=list(STOPS.keys()),
        metrics=METRICS,
        best_model=BEST_MODEL,
        best_train_r2=BEST_TRAIN_R2,
        best_test_r2=BEST_TEST_R2,
        dataset_size=10682,
        feature_count=len(feature_cols),
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data   = request.form.to_dict()
        feats  = build_features(data)
        chosen = data.get('model_choice', 'Random Forest')
        model  = models.get(chosen, models['Random Forest'])

        prediction = round(float(model.predict(feats)[0]), 2)

        # All 4 predictions
        all_preds = {
            name: round(float(m.predict(feats)[0]), 2)
            for name, m in models.items()
        }
        ensemble = round(float(np.mean(list(all_preds.values()))), 2)

        return render_template(
            'index.html',
            airlines=list(AIRLINE_ENC.keys()),
            sources=SOURCES,
            destinations=[k for k in DEST_ENC if k != 'New Delhi'],
            stops=list(STOPS.keys()),
            metrics=METRICS,
            best_model=BEST_MODEL,
            best_train_r2=BEST_TRAIN_R2,
            best_test_r2=BEST_TEST_R2,
            dataset_size=10682,
            feature_count=len(feature_cols),
            prediction=prediction,
            chosen_model=chosen,
            all_preds=all_preds,
            ensemble=ensemble,
            form_data=data,
        )
    except Exception as e:
        return render_template(
            'index.html',
            airlines=list(AIRLINE_ENC.keys()),
            sources=SOURCES,
            destinations=[k for k in DEST_ENC if k != 'New Delhi'],
            stops=list(STOPS.keys()),
            metrics=METRICS,
            best_model=BEST_MODEL,
            best_train_r2=BEST_TRAIN_R2,
            best_test_r2=BEST_TEST_R2,
            dataset_size=10682,
            feature_count=len(feature_cols),
            error=str(e),
        )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON REST API — accepts JSON, returns all 4 model predictions."""
    try:
        data  = request.get_json() or {}
        feats = build_features(data)
        preds = {n: round(float(m.predict(feats)[0]), 2) for n, m in models.items()}
        return jsonify({
            'predictions'  : preds,
            'ensemble_avg' : round(float(np.mean(list(preds.values()))), 2),
            'currency'     : 'INR',
            'best_model'   : BEST_MODEL,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print('=' * 58)
    print('  ✈️  Airline Fare Prediction — ML Web App')
    print('  Open:  http://127.0.0.1:5000')
    print('  API:   POST http://127.0.0.1:5000/api/predict')
    print('=' * 58)
    app.run(debug=True, host='0.0.0.0', port=5000)
