import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Smart ML Analyzer", layout="wide")
st.title("🤖 Smart ML App: Upload, Clean, Train & Predict Automatically")

# Load Dataset
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))

    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # Data Cleaning
    st.subheader("🧼 Data Cleaning & Preprocessing")
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')  # Drop empty columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical
    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    st.success("✅ Data cleaned and encoded successfully!")
    st.dataframe(df.head())

    # Feature Selection
    st.subheader("📌 Feature Selection")
    all_cols = df.columns.tolist()
    target = st.selectbox("🎯 Select Target Column (if available):", options=all_cols)
    features = [col for col in all_cols if col != target]

    X = df[features]
    y = df[target]

    # Task Type
    task_type = None
    if df[target].nunique() <= 10 and df[target].dtype in [int, np.int64]:
        task_type = 'classification'
    elif df[target].dtype in [float, int]:
        task_type = 'regression'

    st.write(f"🔍 Auto Detected Task Type: `{task_type}`")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select Model
    st.subheader("⚙️ Select Model")
    model_name = st.selectbox("Choose a model", [
        "Linear Regression", "Logistic Regression", "Random Forest", 
        "KNN", "SVM", "Naive Bayes"
    ])

    if st.button("🚀 Train & Evaluate"):
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Random Forest":
            model = RandomForestClassifier() if task_type == 'classification' else RandomForestRegressor()
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        elif model_name == "SVM":
            model = SVC()
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        else:
            st.error("Unsupported Model")
            st.stop()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            st.write("📈 Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))
        else:
            st.write("📉 MSE:", mean_squared_error(y_test, y_pred))

# Unsupervised Learning Section
    st.subheader("🔍 Unsupervised Learning (Optional)")
    if st.checkbox("Run K-Means Clustering"):
        try:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(X)

            st.write("📌 Clustered Data Sample:")
            df['Cluster'] = labels
            st.dataframe(df.head())

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X)

            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="Set1")
            ax.set_title("K-Means Clustering Visualization")
            st.pyplot(fig)
        except Exception as e:
            st.warning("Unable to cluster. Please select numerical columns only.")

# Deep Learning Placeholder
    st.subheader("🧠 Deep Learning (Coming Soon)")
    st.markdown("- CNN: Convolutional Neural Networks for Images")
    st.markdown("- RNN: Recurrent Neural Networks for Sequences")
    st.markdown("- Transformers: For Text and NLP tasks")

# Semi-Supervised & RL UI Explanation
    st.subheader("🔁 Semi-Supervised Learning")
    st.info("Semi-Supervised Learning is useful when you have a small set of labeled data and a large set of unlabeled data.")

    st.subheader("🎮 Reinforcement Learning")
    st.info("Reinforcement Learning trains agents via reward/penalty system. Algorithms: Q-Learning, DQN, PPO.")

