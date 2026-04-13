# 🔥 EMBER — Alberta Wildfire Risk Assessor

A machine learning-powered chat interface for predicting large wildfire
escalation risk in Alberta's Forest Protection Area.

## About

This tool uses SHAP feature weights derived from Random Forest, Logistic
Regression, XGBoost, and Isolation Forest models trained on **26,551
wildfire incidents** from the Alberta Forest Protection Area (2006–2024).

## How it works

Enter fire conditions at the time of initial assessment — size, spread rate,
wind speed, temperature, humidity, ignition cause, and forest region — and
the tool returns an **escalation risk score** with the **key driving
factors**.

## Model performance

| Model | Metric | Score |
|---|---|---|
| Random Forest | AUROC | 0.957 ± 0.008 |
| Logistic Regression | AUPRC | 0.584 ± 0.041 |
| XGBoost | AUPRC | 0.567 ± 0.042 |
| Isolation Forest | AUPRC | 0.264 (8× above random baseline of 0.033) |

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## Deployment

The app includes a `Procfile` for one-click deployment on Heroku, Railway,
or Render:

```bash
gunicorn app:app
```

## Data source

Alberta Forestry and Parks — Open Government Licence Alberta  
<https://open.alberta.ca/opendata/wildfire-data>

## Research

IE 565 — University of Tennessee, Knoxville  
Emma Wilhoit · Dr. Maryam Mofrad (PhD) · Dr. Anahita Khojandi (PhD)
