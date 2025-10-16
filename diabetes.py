from dataclasses import dataclass,asdict
from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = FastAPI()

model = joblib.load("gradient_boost.joblib")
scaler = joblib.load("scaler.joblib")

class Diabete(BaseModel):
    pregnancies : int
    glucose	: int
    blood_pressure : int
    skin_thickness	: int
    insulin	: int
    BMI	: float
    diabetes_pedigree_function: float
    age: int

@app.post("/predict")
def diabete_predict(data : Diabete):
    X = [[
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.BMI,
        data.diabetes_pedigree_function, data.age
    ]]
    
    # Prédiction
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    # Extraction du modèle final
    gb_model = model.best_estimator_.named_steps['gb']
    
    # Assurance (probabilité max)
    assurance = round(100 * gb_model.predict_proba(X_scaled).max(), 2)
    
    # Importance des variables
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    importances = pd.DataFrame(
        data=gb_model.feature_importances_,
        index=columns,
        columns=["Importances"]
    )
    
    imp = importances.sort_values('Importances', ascending=False).head(3)
    
    # Construction du retour JSON
    top_features = [
        {"name": name, "importance": round(value, 2)}
        for name, value in zip(imp.index, imp["Importances"])
    ]
    
    return {
        "resultats" : int(y_pred[0]),
        "assurance": float(assurance),
        "probabilities": {
            "no_diabetes": float(round(100-assurance,2)),
            "diabetes": float(assurance)
        },
        "top_features": top_features,
    }