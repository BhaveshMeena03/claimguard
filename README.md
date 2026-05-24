# ClaimGuard

A small full stack project that predicts whether a car insurance claim is likely to be fraud.
You fill in the claim details on a web form and it shows a fraud probability and a risk level
(Low / Medium / High).

I built this to learn how to take a machine learning model out of a notebook and actually use
it in a real app with a backend API and a frontend.

## How it works

- A **Random Forest** model is trained on a public auto insurance claims dataset (1000 claims).
- A **FastAPI** backend loads the trained model and has a `/predict` endpoint.
- A **Next.js** frontend has the form and calls the API to show the result.

```
frontend (Next.js form)  ->  backend (FastAPI /predict)  ->  model.pkl
```

## Tech used

- Python, scikit-learn, pandas (model)
- FastAPI (backend API)
- Next.js, React, TypeScript, Tailwind CSS (frontend)

## Running it locally

You need two terminals (one for the backend, one for the frontend).

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python train_model.py        # trains the model and creates model.pkl (already included)
uvicorn main:app --reload    # runs on http://localhost:8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev                  # runs on http://localhost:3000
```

Then open http://localhost:3000 and fill in the form.

## Model performance

On the test set (20% of the data):

- Accuracy: about 81%
- ROC AUC: about 0.79

It is better at spotting normal claims than fraud ones (fraud recall is around 0.6). The dataset
is small and imbalanced (only ~25% of claims are fraud) so the predictions are not perfect. The
most useful feature was `incident_severity` (major damage claims are more likely to be fraud).

## What I learned

- How to save a trained sklearn pipeline with joblib and load it inside an API
- How to send JSON from a React form to a FastAPI endpoint (and dealing with CORS)
- Why you have to convert the form number inputs back to numbers before sending them
- That a lot of the work is just matching the data shape the model expects

## Things I want to improve later

- Show which features pushed the prediction up or down (feature importance / SHAP)
- Add more input validation on the form
- Deploy it so it has a live link (backend on Render, frontend on Vercel)
- Try other models (XGBoost) and handle the class imbalance better

## Dataset

Auto insurance claims dataset (`backend/insurance_claims.csv`), 1000 rows, target column
`fraud_reported`.
