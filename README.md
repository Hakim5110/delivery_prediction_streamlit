# Delivery Outcome Predictor — Streamlit

A small Streamlit interface for the supplied `DecisionTreeClassifier` model.

## Files

- `app.py` — Streamlit interface and prediction logic
- `decision_tree_model.sav` — trained model artifact
- `requirements.txt` — Python dependencies

## Run locally

Use Python 3.11 for a straightforward deployment environment.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud using GitHub

1. Create a new GitHub repository, for example `delivery-outcome-predictor`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `decision_tree_model.sav` to the repository root.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit installs the packages from `requirements.txt`.

## Important model notes

The provided model is a scikit-learn `DecisionTreeClassifier` with 11 input features and class labels `0` and `1`. The meaning of those classes, feature units, and numeric encodings for categorical-looking fields were not included with the model. The app intentionally displays raw class labels and asks for the original numeric encodings.

Before using it for real decisions, confirm:
- What class `0` and class `1` mean.
- The units and valid ranges of all features.
- How `Traffic_Congestion`, `Weather_Condition`, and `Delivery_Slot` were encoded during training.
- That the training environment and preprocessing match this deployed model.

Only deploy a model artifact you trust. Pickle files can execute code when loaded; do not load untrusted `.sav` files.
