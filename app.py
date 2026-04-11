from flask import Flask, request, jsonify 
import joblib 
import pandas as pd 

app = Flask(__name__) 
model = joblib.load("churn_model.pkl") 

@app.route("/predict",
           methods=["POST"]) 
def predict(): 

    data = request.get_json() 

    input_df = pd.DataFrame([data]) 
    prediction = model.predict(input_df)[0] 
    probability = model.predict_proba(input_df)[0][1] 

    return jsonify({"churn_prediction":int(prediction),
                    "churn_probability":float(probability)
                    }) 

if __name__ == "__main__":
    app.run(debug=True)    