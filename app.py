import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoML Guardian",
    page_icon="🛡️",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding='utf-8')

# --- Session State ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# --- UI: Sidebar ---
with st.sidebar:
    st.title("🛡️ AutoML Guardian")
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file:
        st.header("2. Configure Analysis")
        df = load_data(uploaded_file)
        target_column = st.selectbox("Select Your Target Column", df.columns)
        
        if st.button("🚀 Run AutoML Guardian", use_container_width=True, type="primary"):
            st.session_state.analysis_complete = False # Reset on new run

            # --- THE GUARDIAN'S PRE-FLIGHT CHECK ---
            with st.spinner("Running Pre-flight Checks..."):
                target_series = df[target_column].dropna()
                
                # Check 1: Determine Task Type (THIS IS THE CORE FIX)
                if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
                    task = "📈 Regression"
                else:
                    task = "🎯 Classification"
                st.session_state.task = task
            st.success(f"Pre-flight Check Complete! Task auto-detected: **{task}**")

            with st.spinner("Guardian at the helm... This may take a moment."):
                # 3. Preprocessing
                y = df[target_column]
                X = df.drop(columns=[target_column])
                
                cleaning_log = []
                
                if task == "🎯 Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state.results['label_encoder'] = le
                    cleaning_log.append("Encoded target column 'y' for Classification.")
                
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        if X[col].isnull().sum() > 0:
                            imputer = SimpleImputer(strategy='median')
                            X[col] = imputer.fit_transform(X[[col]]).flatten()
                            cleaning_log.append(f"Imputed missing values in '{col}' with median.")
                    else:
                        if X[col].isnull().sum() > 0:
                            imputer = SimpleImputer(strategy='most_frequent')
                            X[col] = imputer.fit_transform(X[[col]]).flatten()
                            cleaning_log.append(f"Imputed missing values in '{col}' with most frequent value.")
                        X[col] = LabelEncoder().fit_transform(X[col])
                        cleaning_log.append(f"Encoded categorical column '{col}'.")
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                # 4. Model Training
                model = RandomForestRegressor(random_state=42, n_jobs=-1) if task == "📈 Regression" else RandomForestClassifier(random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # 5. Explainability
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                # Store all results at once
                st.session_state.results.update({
                    "task": task,
                    "model": model,
                    "X_test_df": pd.DataFrame(X_test, columns=X.columns),
                    "y_test": y_test,
                    "explainer": explainer,
                    "shap_values": shap_values,
                    "cleaning_log": cleaning_log,
                    "scaler": scaler,
                    "original_features": X.columns.tolist()
                })
                st.session_state.analysis_complete = True
            st.success("Analysis complete!")
            st.rerun()

# --- Main Page Display ---
if not st.session_state.analysis_complete:
    st.info("Upload a dataset and launch the Guardian from the sidebar to begin.")
else:
    res = st.session_state.results
    
    st.header(f"Analysis Dashboard: {res['task']}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Performance", "🧠 Explainability", "💡 Simulator", "📦 Assets Hub"])

    with tab1:
        st.subheader("Model Evaluation")
        y_pred = res['model'].predict(res['X_test_df'])
        
        if res['task'] == '🎯 Classification':
            score = accuracy_score(res['y_test'], y_pred)
            st.metric("Model Accuracy", f"{score:.4f}")
            report = classification_report(res['y_test'], y_pred, target_names=res['label_encoder'].classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else: # Regression
            score = r2_score(res['y_test'], y_pred)
            st.metric("Model R-squared (R²)", f"{score:.4f}")

    with tab2:
        st.subheader("SHAP Model Explanations")
        st.write("Understand *why* the model makes its predictions.")
        
        st.write("**Global Feature Importance:**")
        summary_fig, ax = plt.subplots()
        shap.summary_plot(res['shap_values'], res['X_test_df'], plot_type="bar", show=False)
        st.pyplot(summary_fig)
        
        st.write("**Local Prediction Deconstruction:**")
        row_index = st.selectbox("Select a row from the test set to explain:", res['X_test_df'].index)
        st.dataframe(res['X_test_df'].iloc[[row_index]])
        
        decision_fig, ax = plt.subplots()
        shap.decision_plot(res['explainer'].expected_value, res['shap_values'][row_index], res['X_test_df'].iloc[row_index], show=False)
        st.pyplot(decision_fig)

    with tab3:
        st.subheader("What-If Scenario Simulator")
        input_data = {}
        for col in res['original_features']:
            input_data[col] = st.number_input(f"Input for '{col}'", value=0.0, key=f"sim_{col}")
        
        if st.button("Predict Scenario"):
            input_df = pd.DataFrame([input_data])
            input_scaled = res['scaler'].transform(input_df)
            prediction = res['model'].predict(input_scaled)
            if res['task'] == "🎯 Classification":
                prediction = res['label_encoder'].inverse_transform(prediction)
            st.success(f"**Simulated Prediction:** {prediction[0]}")
            
    with tab4:
        st.subheader("Downloadable Assets & Logs")
        
        st.write("**Data Cleaning Log:**")
        st.json(res['cleaning_log'])
        
        model_bytes = io.BytesIO()
        joblib.dump(res['model'], model_bytes)
        st.download_button("⬇️ Download Trained Model (.joblib)", data=model_bytes, file_name="guardian_model.joblib")
