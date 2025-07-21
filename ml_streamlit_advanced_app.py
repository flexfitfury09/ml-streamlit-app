# Save this as ml_streamlit_advanced_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

def main():
    st.set_page_config(page_title="AutoML Streamlit App", layout="wide")
    st.title("🤖 AutoML Web App")
    st.markdown("Upload a dataset to automatically train ML models (Regression or Classification)")

    uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        st.markdown(f"**Shape:** {df.shape}")
        st.write("**Column Types:**", df.dtypes)

        st.subheader("📈 Data Exploration")
        if st.checkbox("Show Summary Statistics"):
            st.write(df.describe())
        if st.checkbox("Show Null Values"):
            st.write(df.isnull().sum())
        if st.checkbox("Correlation Heatmap (numeric only)"):
            numeric_df = df.select_dtypes(include=np.number)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        if st.checkbox("Histogram of Features"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_hist_col = st.selectbox("Select Column", numeric_cols)
            fig, ax = plt.subplots()
            df[selected_hist_col].hist(bins=30, edgecolor='black', ax=ax)
            ax.set_title(f"Histogram of {selected_hist_col}")
            st.pyplot(fig)

        st.subheader("🎯 Target & Task")
        target = st.selectbox("Select the target column", df.columns)
        problem_type = st.radio("Type of prediction problem", ["Regression", "Classification"])

        df = df.dropna(subset=[target])
        if st.checkbox("One-Hot Encode Categorical Features"):
            df = pd.get_dummies(df)
            st.success("Categorical features encoded.")

        if target not in df.columns:
            st.error("Target column missing after encoding.")
            return

        features = [col for col in df.columns if col != target]

        if st.checkbox("Scale Numeric Features"):
            numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()
            if numeric_features:
                scaler = StandardScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                st.success("Numeric features scaled.")
            else:
                st.warning("No numeric features found.")

        test_size = st.slider("Test Set Size (%)", 10, 50, 20)
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        st.subheader("🧠 Model Selection")
        if problem_type == "Regression":
            model_name = st.selectbox("Choose a regression model", ["LinearRegression", "RandomForestRegressor"])
        else:
            model_name = st.selectbox("Choose a classification model", ["LogisticRegression", "RandomForestClassifier"])

        if st.button("🚀 Train Model"):
            if model_name == "LinearRegression":
                model = LinearRegression()
            elif model_name == "RandomForestRegressor":
                model = RandomForestRegressor()
            elif model_name == "LogisticRegression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "RandomForestClassifier":
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📊 Evaluation Metrics")
            result_df = X_test.copy()
            result_df["True"] = y_test
            result_df["Predicted"] = y_pred

            if problem_type == "Regression":
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.write(f"**MAE**: {mae:.2f}")
                st.write(f"**RMSE**: {rmse:.2f}")
                st.write(f"**R² Score**: {r2:.2f}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                st.write(f"**Accuracy**: {acc:.2f}")
                st.write(f"**Precision**: {prec:.2f}")
                st.write(f"**Recall**: {rec:.2f}")
                st.write(f"**F1 Score**: {f1:.2f}")
                st.text("Classification Report")
                st.text(classification_report(y_test, y_pred, zero_division=0))
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

            st.subheader("📄 Prediction Results")
            st.dataframe(result_df.head(50))

            csv = result_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Full Prediction Results", csv, "predictions_results.csv", "text/csv")

if __name__ == "__main__":
    main()
