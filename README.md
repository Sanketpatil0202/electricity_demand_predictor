# ⚡ Electricity Demand Predictor

A simple end-to-end machine learning project: a **Random Forest Regressor**
trained on weather and time-of-day data, served through an interactive
**Streamlit** web app.

🔗 Try it live: *https://59zvxdjqgem6ipkzbfzrdx.streamlit.app/*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)

## 🧠 About the project

This project predicts electricity demand from four features:

| Feature | Description |
|---|---|
| `Temperature` | Normalized temperature reading |
| `Humidity` | Normalized humidity reading |
| `WindSpeed` | Normalized wind speed reading |
| `HourOfDay` | Normalized hour of day |

The model is a `RandomForestRegressor` from scikit-learn wrapped in a
`Pipeline`, trained with an 80/20 train-test split. It was originally
prototyped in a Jupyter notebook (`ElectricityDemand_RandomForest_regressor.ipynb`)
and refactored here into a reusable script + Streamlit app so anyone can
run it themselves.

## 📂 Project structure

```
electricity-demand-predictor/
├── app.py                     # Streamlit app
├── train_model.py             # Trains and saves the model
├── generate_sample_data.py    # Creates a sample dataset (swap for your own)
├── model/
│   └── electricity_demand_model.pkl
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/electricity-demand-predictor.git
cd electricity-demand-predictor

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Generate a sample dataset and retrain the model
python generate_sample_data.py
python train_model.py

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> A pre-trained model (`model/electricity_demand_model.pkl`) is already
> included, trained on a synthetic sample dataset, so the app works out
> of the box even if you skip steps 4.

## 🔁 Using your own dataset

Replace `electricity.csv` with your own data (same column names:
`Temperature, Humidity, WindSpeed, HourOfDay, ElectricityDemand`), then run:

```bash
python train_model.py --data your_file.csv
```

## 📊 Model performance

On the sample dataset (your numbers will vary with real data):

| Metric | Value |
|---|---|
| R² | ~0.87 |
| MAE | ~0.065 |
| RMSE | ~0.076 |

## 🛠️ Built with

- [scikit-learn](https://scikit-learn.org/) – RandomForestRegressor
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/)
- [Streamlit](https://streamlit.io/) – interactive web app
- [joblib](https://joblib.readthedocs.io/) – model serialization

## 📄 License

MIT — free to use, modify, and share.

---

*Originally developed as a Jupyter notebook exercise in regression modeling,
refactored into a shareable interactive demo.*
