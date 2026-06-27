from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import create_model 
import joblib 
import pandas as pd 
import uvicorn  
from typing import Dict, Any 
 
model = joblib.load("model/churn_model.pkl") 
features = model.feature_names_in_ 
  
input_fields = {f: (float, ...) for f in features} 
ChurnInput = create_model('ChurnInput', **input_fields) 

def get_feature_types() : 
    x_train = pd.read_csv("data/train.csv")[features]  
    return {f: x_train[f].dtype for f in features}  

feature_types = get_feature_types() 

def map_dtype(dtype) : 
    if "float" in str(dtype): return float 
    if "int" in str(dtype): return int 
    return str 

input_fields = {f: (map_dtype(feature_types[f]), ...) for f in features} 
ChurnInput = create_model('ChurnInput', **input_fields) 

app = FastAPI() 

app.add_middleware( 
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
) 

@app.get("/features") 
def get_features() : 
    return {"features": features.tolist()} 
 
@app.post("/predict") 
async def predict(data : Dict[str, Any]) : 
    try : 
        input_df = pd.DataFrame([data.dict()]) 
        input_df = input_df[features] 
        
        prediction = int(model.predict(input_df)[0]) 
        probability = float(model.predict_proba(input_df)[0][1]) 
        
        return {"churn_prediction": prediction, "churn_probability": probability} 
    except Exception as e : 
        raise HTTPException(status_code=400, detail=str(e)) 
    
@app.get("/metadata") 
def get_metadata() : 
    ct = model.named_steps['preprocess'] 
    ohe = ct.named_transformers_['cat'] 
    try :     
        metadata = {} 
        for name, cat in zip(ohe.feature_names_in_, ohe.categories_) : 
            metadata[name] = cat.tolist() 
        return metadata 
    except KeyError : 
        return {} 