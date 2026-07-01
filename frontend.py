import streamlit as st 
import requests 

API_URL = "https://customer-retention-analytics-system-api.onrender.com" 

st.set_page_config( 
    page_title="Customer Retention Analytics System", 
    page_icon="📊", 
    layout="wide" 
) 

st.title("📊 Customer Retention Analytics System") 
st.markdown( 
    "Predict whether a customer is likely to stay or churn using a trained Machine Learning model." 
) 


@st.cache_data 
def get_app_data() : 
    feat_res = requests.get(f"{API_URL}/features", timeout=30) 
    feat_res.raise_for_status() 

    meta_res = requests.get(f"{API_URL}/metadata", timeout=30) 
    meta_res.raise_for_status() 

    return feat_res.json()["features"], meta_res.json() 


try : 
    features, metadata = get_app_data() 
except Exception as e : 
    st.error(f"Unable to connect to backend.\n\n{e}") 
    st.stop()  

customer_info = [ 
    "gender", 
    "SeniorCitizen", 
    "Partner", 
    "Dependents", 
    "tenure" 
] 

services = [ 
    "PhoneService", 
    "MultipleLines", 
    "InternetService", 
    "OnlineSecurity", 
    "OnlineBackup", 
    "DeviceProtection", 
    "TechSupport", 
    "StreamingTV", 
    "StreamingMovies" 
] 

billing = [ 
    "Contract", 
    "PaperlessBilling", 
    "PaymentMethod", 
    "MonthlyCharges", 
    "TotalCharges" 
] 


def render_section(title, section_features, user_inputs) : 
    st.subheader(title) 

    cols = st.columns(3) 

    for i, feature in enumerate(section_features) : 
        with cols[i % 3] : 

            if feature in metadata : 
                user_inputs[feature] = st.selectbox( 
                    feature, 
                    metadata[feature] 
                ) 

            else : 
                if feature in ["SeniorCitizen", "tenure"] : 
                    user_inputs[feature] = st.number_input( 
                        feature, 
                        step=1, 
                        value=0 
                    ) 
                else : 
                    user_inputs[feature] = st.number_input( 
                        feature, 
                        value=0.0 
                    ) 


with st.form("prediction_form") : 

    user_inputs = {} 

    render_section( 
        "👤 Customer Information", 
        customer_info, 
        user_inputs 
    ) 

    st.divider() 

    render_section( 
        "🌐 Service Details", 
        services, 
        user_inputs 
    ) 

    st.divider() 

    render_section( 
        "💳 Billing Information", 
        billing, 
        user_inputs 
    ) 

    st.divider() 

    submitted = st.form_submit_button( 
        "🔍 Predict Customer Retention", 
        use_container_width=True 
    ) 
 

if submitted : 

    with st.spinner("Predicting...") : 

        try : 

            response = requests.post( 
                f"{API_URL}/predict", 
                json=user_inputs, 
                timeout=60 
            ) 

            response.raise_for_status() 

            result = response.json() 

            probability = result["churn_probability"] 

            if result["churn_prediction"] == 1 : 
                st.success("✅ Customer is likely to be retained.") 
            else : 
                st.error("⚠️ Customer is likely to churn.") 

            col1, col2 = st.columns(2) 

            with col1 : 
                st.metric( 
                    "Retention Probability", 
                    f"{probability:.2%}" 
                ) 

            with col2 : 
                st.metric( 
                    "Churn Probability", 
                    f"{1-probability:.2%}" 
                ) 

        except Exception as e : 
            st.exception(e) 