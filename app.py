import streamlit as st
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from streamlit_shap import st_shap
import google.generativeai as genai

# --- 1. Page Configuration ---
st.set_page_config(page_title="Liver Disease AI Assistant", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button {
        color: white; background-color: #007bff; border-radius: 8px;
        height: 3.2em; width: 100%; font-size: 18px; font-weight: 600; border: none;
    }
    div.stButton > button:hover { background-color: #0056b3; }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SETUP GEMINI API ---
# ⚠️ REPLACE WITH YOUR ACTUAL API KEY
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ Google API Key not found. Please set it in secrets.toml")
    st.stop()

# --- 3. Load Data & Train Model ---
@st.cache_data
def load_data_and_train():
    try:
        df = pd.read_csv('indian_liver_patient.csv')
    except FileNotFoundError:
        return None, None, None, None

    # Preprocessing
    df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0}) 

    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, df, acc, X_train

model, df, accuracy, X_train = load_data_and_train()

if model is None:
    st.error("⚠️ Error: 'indian_liver_patient.csv' not found.")
    st.stop()

# --- 4. Sidebar Input ---
st.sidebar.header("📋 Medical Staff Information")

# NEW: Doctor Name Input
doctor_name = st.sidebar.text_input("Attending Physician", value="Dr. Stephen Strange")

st.sidebar.markdown("---")
st.sidebar.header("📝 Patient Information")
patient_name = st.sidebar.text_input("Patient Name", value="John Doe")

st.sidebar.subheader("Clinical Data")

def user_input_features():
    age = st.sidebar.slider('Age', 1, 90, 45)
    gender = st.sidebar.radio('Gender', ('Male', 'Female'), horizontal=True)
    
    st.sidebar.caption("Bilirubin Levels (mg/dL)")
    total_bilirubin = st.sidebar.slider('Total Bilirubin', 0.1, 30.0, 1.0)
    direct_bilirubin = st.sidebar.slider('Direct Bilirubin', 0.1, 15.0, 0.4)
    
    st.sidebar.caption("Liver Enzymes (U/L)")
    alp = st.sidebar.slider('Alkaline Phosphotase (ALP)', 50, 2000, 200)
    alt = st.sidebar.slider('Alamine Aminotransferase (ALT)', 10, 1500, 30)
    ast = st.sidebar.slider('Aspartate Aminotransferase (AST)', 10, 2000, 40)
    
    st.sidebar.caption("Proteins (g/dL)")
    proteins = st.sidebar.slider('Total Proteins', 2.0, 10.0, 6.5)
    albumin = st.sidebar.slider('Albumin', 1.0, 6.0, 3.0)
    ag_ratio = st.sidebar.slider('Albumin/Globulin Ratio', 0.1, 3.0, 1.0)

    data = {
        'Age': age, 'Gender': 1 if gender == 'Male' else 0,
        'Total_Bilirubin': total_bilirubin, 'Direct_Bilirubin': direct_bilirubin,
        'Alkaline_Phosphotase': alp, 'Alamine_Aminotransferase': alt,
        'Aspartate_Aminotransferase': ast, 'Total_Protiens': proteins,
        'Albumin': albumin, 'Albumin_and_Globulin_Ratio': ag_ratio
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 5. Helper Function: Call Gemini ---
def consult_ai_doctor(prediction_text, patient_data, risk_factors, p_name, doc_name):
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except:
        return "❌ API Key configuration failed."

    data_str = "\n".join([f"- {col}: {val}" for col, val in patient_data.items()])
    
    # Updated Prompt with Doctor's Name
    prompt = f"""
    Act as {doc_name}, a senior Hepatologist (Liver Specialist).
    
    Patient Name: {p_name}
    Diagnosis Result: {prediction_text}
    Key Risk Factors: {risk_factors}
    Clinical Data: {data_str}
    
    Please provide a professional medical report in ENGLISH, signed by {doc_name}:
    1. **Executive Summary**: Explain the condition.
    2. **Analysis**: Highlight abnormal values.
    3. **Recommendations**: 3 lifestyle changes.
    4. **Next Steps**: Diagnostic tests.
    
    Tone: Professional, objective, and empathetic.
    """

    # Auto-Discovery Loop
    valid_model = None
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini' in m.name:
                    valid_model = m.name
                    break
    except:
        pass

    candidate_models = [valid_model] if valid_model else ['models/gemini-1.5-flash', 'models/gemini-pro']
    
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return f"✅ **Report by {doc_name} (Model: {model_name})**\n\n" + response.text
        except:
            continue
            
    return "❌ Unable to connect to AI Service."

# --- 6. Main Dashboard Layout ---
st.title("🏥 Liver Disease Prediction & Analysis System")

# Display Doctor and Patient names at the top
st.markdown(f"#### 👨‍⚕️ Attending Physician: **{doctor_name}** |  🤒 Patient: **{patient_name}**")

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy:.1%}")
col2.metric("Database Size", f"{len(df)} records")
col3.metric("Population Risk", f"{df['Dataset'].mean():.1%}")
st.divider()

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("🔍 Diagnostic Results")
    
    if st.button("RUN DIAGNOSIS"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        is_disease = prediction[0] == 1
        prob_percent = probability[0][1] * 100
        
        st.markdown("---")
        if is_disease:
            st.error(f"### ⚠️ HIGH RISK DETECTED")
            st.write(f"Probability: **{prob_percent:.1f}%**")
            pred_text = "High Risk of Liver Disease"
        else:
            st.success(f"### ✅ LOW RISK (HEALTHY)")
            st.write(f"Probability: **{prob_percent:.1f}%** (Healthy probability: {100-prob_percent:.1f}%)")
            pred_text = "Low Risk (Healthy)"

        st.markdown("### 🧬 Factor Analysis (SHAP)")
        with st.spinner('Calculating impact factors...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df, check_additivity=False)
            
            shap_val = None
            base_val = None
            if isinstance(shap_values, list):
                shap_val = shap_values[1]
                base_val = explainer.expected_value[1]
            else:
                shap_val = shap_values if len(shap_values.shape) == 2 else shap_values[:,:,1]
                base_val = explainer.expected_value
            
            if hasattr(shap_val, 'ndim') and shap_val.ndim == 2:
                shap_val = shap_val[0]
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[0]

            st_shap(shap.force_plot(base_val, shap_val, input_df.iloc[0]), height=150)
            
            feature_names = input_df.columns
            indices = np.argsort(np.abs(shap_val))[::-1]
            top_factors = [f"{feature_names[i]} ({shap_val[i]:.2f})" for i in indices[:3]]
            risk_factors_str = ", ".join(top_factors)

        st.markdown("---")
        st.subheader(f"🤖 Report by {doctor_name}")
        # Pass doctor_name to the AI function
        ai_response = consult_ai_doctor(pred_text, input_df.iloc[0].to_dict(), risk_factors_str, patient_name, doctor_name)
        st.info(ai_response)

with col_right:
    st.subheader("📊 Comparative Analytics")
    st.caption(f"Comparing {patient_name} against Population Benchmarks")
    
    features_to_plot = ['Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Albumin']
    
    patient_vals = input_df[features_to_plot].iloc[0].values
    disease_vals = df[df['Dataset']==1][features_to_plot].mean().values
    healthy_vals = df[df['Dataset']==0][features_to_plot].mean().values
    
    max_vals = np.maximum.reduce([patient_vals, disease_vals, healthy_vals])
    max_vals[max_vals == 0] = 1 
    
    p_norm = patient_vals / max_vals
    d_norm = disease_vals / max_vals
    h_norm = healthy_vals / max_vals
    
    categories = ['Bilirubin', 'ALP', 'ALT', 'AST', 'Albumin']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_norm, theta=categories, fill='toself', name='Avg Healthy', line=dict(color='#2ecc71', dash='dot'), opacity=0.4, hoverinfo='skip'))
    fig.add_trace(go.Scatterpolar(r=d_norm, theta=categories, fill='toself', name='Avg Disease', line=dict(color='#e74c3c', dash='dash'), opacity=0.4))
    fig.add_trace(go.Scatterpolar(r=p_norm, theta=categories, fill='toself', name=f'{patient_name}', line=dict(color='#3498db', width=3), marker=dict(size=8)))

    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1.1])), showlegend=True, margin=dict(l=40, r=40, t=20, b=20), height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### 🔢 Detailed Clinical Values")
    comparison_table = pd.DataFrame({'Test': categories, f'{patient_name}': [f"{v:.1f}" for v in patient_vals], 'Ref (Healthy)': [f"{v:.1f}" for v in healthy_vals], 'Ref (Disease)': [f"{v:.1f}" for v in disease_vals], 'Status': ['⚠️ High' if p > h * 1.5 else '✅ Normal' for p, h in zip(patient_vals, healthy_vals)]}).set_index('Test')
    
    def highlight_status(val):
        color = '#ffcccc' if 'High' in val else '#ccffcc'
        return f'background-color: {color}'

    st.dataframe(comparison_table.style.applymap(highlight_status, subset=['Status']), use_container_width=True)