import streamlit as st

st.set_page_config(
    page_title="Welcome to the All-in-One ML Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Welcome to the Unified AI Platform")
st.sidebar.success("Select a module above to begin.")

st.markdown(
    """
    This is a powerful, all-in-one application that combines multiple machine learning
    and deep learning capabilities into a single, easy-to-use interface.
    
    ### What can you do here?
    
    👈 **Select a module from the sidebar** to get started:
    
    - **🤖 AutoML Platform**: 
      - Upload a CSV file with tabular data.
      - Automatically clean the data and run multiple models for **Supervised Learning** (Classification/Regression).
      - Automatically run algorithms for **Unsupervised Learning** (Clustering) to discover hidden patterns.
      
    - **🖼️ Deep Learning Classifier**:
      - Upload an image.
      - Use a state-of-the-art Convolutional Neural Network (CNN) to instantly identify the contents of your image.
      
    This platform demonstrates a full end-to-end workflow, from data ingestion to model analysis, 
    all within one website.
    """
)