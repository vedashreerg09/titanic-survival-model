from fastapi import FastAPI
import joblib
import pandas as pd

# Load trained model
model = joblib.load("titanic_model.joblib")

# Create FastAPI instance
app = FastAPI()   # <- this must exist

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running"}

@app.post("/predict")
def predict_survival(Pclass: int, Sex: str, Age: float, SibSp: int,
                     Parch: int, Fare: float, Embarked: str):

    data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Embarked": [Embarked]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

