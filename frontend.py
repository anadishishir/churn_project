import streamlit as st 
import requests 

st.set_page_config(page_title="Customer Retention Analytics System", layout="wide") 
st.title("Customer Retention Analytics System") 

@st.cache_data 
def get_app_data() : 
    try : 
        feat_res = requests.get("http://127.0.0.1:8000/features") 
        features = feat_res.json().get("features", []) 

        meta_res = requests.get("http://127.0.0.1:8000/metadata") 
        metadata = meta_res.json() 

        return features, metadata 
    
    except Exception as e :  
        return None, None  

features, metadata = get_app_data()  

if not features : 
    st.error("Backend is unreachable. Ensure 'python app.py' is running on port 8000.") 
    st.stop() 

with st.form("customer_retention_analytics_form") : 
    st.subheader("Customer Information") 
    user_inputs = {} 

    cols = st.columns(3) 
    for i, feature in enumerate(features) : 
        with cols[i % 3] : 
            if feature in metadata : 
                user_inputs[feature] = st.selectbox(feature, metadata[feature]) 
            else : 
                user_inputs[feature] = st.number_input(feature, value = 0.0, format = "%f") 
    
    submitted = st.form_submit_button("Predict Customer Retention Probability") 

if submitted : 
    try : 
        res = requests.post("http://127.0.0.1:8000/predict", json=user_inputs) 
        if res.status_code == 200 : 
            data = res.json() 
            prediction = "Retained" if data['churn_prediction'] == 1 else "Not Retained" 
            st.success(f"### Customer Retention Prediction : {prediction}") 
            st.write(f"### Model Confidence : {data['churn_probability']:.2%}") 
        else : 
            st.error(f"Error : {res.json().get('detail', 'Unkonwn error')}") 
    except Exception as e : 
        st.error(f"Could not connect to backend : {e}") 