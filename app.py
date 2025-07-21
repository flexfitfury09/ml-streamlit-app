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
st.set_page_config(page_title="AutoML Titan", page_icon="🛡️", layout="wide")

# --- Session State Initialization ---
st.session_state.setdefault('analysis_complete', False)
st.session_state.setdefault('results', {})

# --- Sidebar UI ---
with st.sidebar:
    st.title("🛡️ AutoML Titan")
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.header("2. Configure Analysis")
        target_column = st.selectbox("Select Target Column", df.columns, key="target_selector")

        if st.button("🚀 Launch Titan", use_container_width=True, type="primary"):
            st.session_state.analysis_complete = False # Reset on new run

            with st.spinner("Executing Pre-flight Checks..."):
                target_series = df[target_column].dropna()
                if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() >= 15:
                    task = "📈 Regression"
                else:
                    task = "🎯 Classification"
                st.session_state.task = task
            st.success(f"Check Complete! Task auto-detected: **{task}**")

            with st.spinner("Titan is running... This may take a moment."):
                y = df[target_column].copy()
                X = df.drop(columns=[target_column]).copy()
                
                if task == "🎯 Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                    st.session_state.results['label_encoder'] = le

                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        if X[col].isnull().sum() > 0:
                            imputer = SimpleImputer(strategy='median')
                            X[col] = imputer.fit_transform(X[[col]]).flatten()
                    else: # Categorical
                        imputer = SimpleImputer(strategy='most_frequent')
                        X[col] = imputer.fit_transform(X[[col]]).flatten()
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(random_state=42) if task == "📈 Regression" else RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                st.session_state.results.update({
                    "model": model, "scaler": scaler, "features": X.columns.tolist(),
                    "X_test_df": pd.DataFrame(X_test, columns=X.columns), "y_test": y_test,
                    "explainer": explainer, "shap_values": shap_values,
                })
                st.session_state.analysis_complete = True
            st.success("Analysis complete!")
            st.rerun()

# --- Main Page Display ---
if not st.session_state.analysis_complete:
    st.info("Upload a dataset and launch the Titan from the sidebar to begin.")
else:
    res = st.session_state.results
    st.header(f"Titan Analysis Dashboard: {st.session_state.task}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Performance", "🧠 Explainability", "💡 Simulator", "📦 Assets"])
    with tab1:
        st.subheader("Model Evaluation")
        y_pred = res['model'].predict(res['X_test_df'])
        if st.session_state.task == '🎯 Classification':
            score = accuracy_score(res['y_test'], y_pred)
            st.metric("Model Accuracy", f"{score:.4f}")
            report = classification_report(res['y_test'], y_pred, target_names=res['label_encoder'].classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            score = r2_score(res['y_test'], y_pred)
            st.metric("Model R-squared (R²)", f"{score:.4f}")

    with tab2:
        st.subheader("SHAP Model Explanations")
        st.write("**Global Feature Importance:** Shows the overall impact of features.")
        summary_fig, ax = plt.subplots()
        shap.summary_plot(res['shap_values'], res['X_test_df'], plot_type="bar", show=False)
        st.pyplot(summary_fig)
        
        st.write("**Local Prediction Explanation:** Deconstructs a single prediction.")
        row_index = st.selectbox("Select a row to explain:", res['X_test_df'].index)
        decision_fig = shap.decision_plot(res['explainer'].expected_value, res['shap_values'][row_index], res['X_test_df'].iloc[row_index], show=False, new_base_value=True)
        st.pyplot(decision_fig)

    with tab3:
        st.subheader("What-If Scenario Simulator")
        input_data = {}
        for col in res['features']:
            input_data[col] = st.number_input(f"Input for '{col}'", value=0.0, key=f"sim_{col}")
        if st.button("Predict Scenario"):
            input_df = pd.DataFrame([input_data])
            input_scaled = res['scaler'].transform(input_df)
            prediction = res['model'].predict(input_scaled)
            if st.session_state.task == "🎯 Classification":
                prediction = res['label_encoder'].inverse_transform(prediction)
            st.success(f"**Simulated Prediction:** {prediction[0]}")

    with tab4:
        st.subheader("Downloadable Assets")
        model_bytes = io.BytesIO()
        joblib.dump(res['model'], model_bytes)
        st.download_button("⬇️ Download Trained Model (.joblib)", data=model_bytes, file_name="titan_model.joblib")
