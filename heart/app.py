import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import base64 

# Set page config with custom icon and wide layout
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Add custom CSS for background and styling
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        img_data = file.read()
    b64_img = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .reportview-container .main .block-container{{
            padding-top: 2rem;
        }}
        .css-1lcbmhc .css-1d391kg {{
            font-size: 18px;
            color: white;
            text-shadow: 1px 1px 3px black;
        }}
        </style>
        """, unsafe_allow_html=True
    )

add_bg_from_local('background.jpg')  # Make sure 'background.jpg' exists in your directory

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Heart Disease Prediction"])

if page == "Home":
    # Home page content
    st.title("Welcome to the Heart Disease Prediction System")
    st.write("""
        This application uses machine learning to predict the likelihood of heart disease based on user-provided data.
        
        ### How It Works
        - Enter patient information under the **Heart Disease Prediction** section.
        - The application uses a trained neural network model to analyze the data and predict the likelihood of heart disease.
        - This system is built to assist medical professionals and individuals in early risk detection, though it should not replace professional medical advice.
        
        ### Disclaimer
        This tool provides an estimation based on patterns from historical data and is not a substitute for clinical diagnosis. Please consult a healthcare provider for any health concerns.
    """)
    # Optional: Add a relevant image for the home page
    
elif page == "Heart Disease Prediction":
    # Title and description with icons
    st.title("❤️ Heart Disease Prediction System")
    st.write("This application predicts the presence of heart disease based on patient data.")

    # Columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🔢 Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("🚻 Sex", ["Male", "Female"])
        cp = st.selectbox("💔 Chest Pain Type", 
                        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("💉 Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        chol = st.number_input("🍔 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with col2:
        restecg = st.selectbox("🩺 Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("🏃‍♂️ Maximum Heart Rate", min_value=70, max_value=220, value=150)
        exang = st.selectbox("🚶 Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=6.2, value=0.0)
        slope = st.selectbox("🧭 Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("🔍 Major Vessels Colored", min_value=0, max_value=3, value=0)
        thal = st.selectbox("🧬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Function to preprocess input data
    def preprocess_input():
        # Encoding categorical features
        sex_n = 1 if sex == "Male" else 0
        cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
        cp_n = cp_map[cp]
        fbs_n = 1 if fbs == "Yes" else 0
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_n = restecg_map[restecg]
        exang_n = 1 if exang == "Yes" else 0
        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        slope_n = slope_map[slope]
        thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
        thal_n = thal_map[thal]
        
        # Input data array
        input_data = np.array([[age, sex_n, cp_n, trestbps, chol, fbs_n, restecg_n, thalach, exang_n, oldpeak, slope_n, ca, thal_n]])
        
        return input_data

    # Create prediction button
    if st.button("🔍 Predict"):
        try:
            # Load model
            model = load_model("heart_disease_neural_network_model.keras")
            
            # Preprocess and scale input data
            input_data = preprocess_input()
            scaler = StandardScaler()
            input_scaled = scaler.fit_transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled).flatten()[0]
            
            # Display result
            st.header("🩺 Prediction Result")
            if prediction < 0.5:
                st.success("✅ No Heart Disease Detected")
                st.write(f"Probability of Heart Disease: {prediction:.2%}")
            else:
                st.error("⚠️ Heart Disease Detected")
                st.write(f"Probability of Heart Disease: {prediction:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure all inputs are valid and try again.")

    # Feature Information section with toggle
    with st.expander("🔍 Feature Information"):
        st.write("""
        - **Age**: Age in years
        - **Sex**: Gender of the patient
        - **Chest Pain Type**: Type of chest pain experienced
        - **Resting Blood Pressure**: Blood pressure (mm Hg) while resting
        - **Cholesterol**: Serum cholesterol in mg/dl
        - **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl
        - **Resting ECG**: Results of resting electrocardiogram
        - **Maximum Heart Rate**: Maximum heart rate achieved
        - **Exercise Induced Angina**: Whether angina was induced by exercise
        - **ST Depression**: ST depression induced by exercise relative to rest
        - **Slope**: Slope of the peak exercise ST segment
        - **Number of Vessels**: Number of major vessels colored by fluoroscopy
        - **Thalassemia**: Type of thalassemia
        """)

# Add footer with separation
st.markdown("---")
st.markdown("💡 **Heart Disease Prediction System - Created with Streamlit**")
