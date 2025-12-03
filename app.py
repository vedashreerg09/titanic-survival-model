from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("titanic_model.joblib")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict")
def predict_survival(Pclass: int, Sex: str, Age: float, SibSp: int,
                     Parch: int, Fare: float, Embarked: str):

    # Create input dataframe
    data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    })

    # Predict
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "survived_prediction": int(prediction),
        "survival_probability": float(probability)
    }
