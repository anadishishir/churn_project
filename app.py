from fastapi import FastAPI, HTTPException 
from pydantic import create_model 
import joblib 
import pandas as pd 
import uvicorn  
from typing import Dict, Any 
 
model = joblib.load("model/churn_model.pkl") 
features = model.feature_names_in_ 
  
input_fields = {f: (float, ...) for f in features} 
ChurnInput = create_model('ChurnInput', **input_fields) 

app = FastAPI() 

@app.get("/features") 
def get_features() : 
    return {"features": features.tolist()} 
 
@app.post("/predict") 
async def predict(data : Dict[str, Any]) : 
    try : 
        input_df = pd.DataFrame([data]) 
        input_df = input_df[features]   
        
        prediction = int(model.predict(input_df)[0]) 
        probability = float(model.predict_proba(input_df)[0][1]) 
        
        return {"churn_prediction" : prediction, "churn_probability" : probability} 
    except Exception as e : 
        raise HTTPException(status_code=400, detail=str(e)) 

@app.get("/metadata") 
def get_metadata() : 
    ct = model.named_steps['preprocess'] 
    ohe = ct.named_transformers_['cat'] 

    metadata = {} 
    for name, cat in zip(ohe.feature_name_in_, ohe.categories_) : 
        metadata[name] = cat.tolist() 
    return metadata 