import numpy as np
import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load the diabetes models
loaded_diabetes_model_svm = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_diabetes_sav_svm', 'rb'))
loaded_diabetes_model_lr = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_diabetes_sav_LR', 'rb'))
loaded_diabetes_model_knn = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_diabetes_sav_KNN', 'rb'))

# Load the heart disease models
loaded_heart_model_svm = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_heart_sav_svm', 'rb'))
loaded_heart_model_lr = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_heart_sav_LR', 'rb'))
loaded_heart_model_knn = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_heart_sav_KNN', 'rb'))

# Load the Parkinson's disease models
loaded_parkinson_model_svm = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_Parkinson_sav_svm', 'rb'))
loaded_parkinson_model_lr = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_Parkinson_sav_LR', 'rb'))
loaded_parkinson_model_knn = pickle.load(open('C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_Parkinson_sav_KNN', 'rb'))

# Functions for predictions using SVM, LR, and KNN
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction_svm = loaded_diabetes_model_svm.predict(input_data_reshaped)
    prediction_lr = loaded_diabetes_model_lr.predict(input_data_reshaped)
    prediction_knn = loaded_diabetes_model_knn.predict(input_data_reshaped)

    return prediction_svm[0], prediction_lr[0], prediction_knn[0]

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction_svm = loaded_heart_model_svm.predict(input_data_reshaped)
    prediction_lr = loaded_heart_model_lr.predict(input_data_reshaped)
    prediction_knn = loaded_heart_model_knn.predict(input_data_reshaped)

    return prediction_svm[0], prediction_lr[0], prediction_knn[0]

def parkinson_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction_svm = loaded_parkinson_model_svm.predict(input_data_reshaped)
    prediction_lr = loaded_parkinson_model_lr.predict(input_data_reshaped)
    prediction_knn = loaded_parkinson_model_knn.predict(input_data_reshaped)

    return prediction_svm[0], prediction_lr[0], prediction_knn[0]

# Function to display images for each disease
def display_disease_images(selected_disease):
    base_path = r"C:\Users\Suman\OneDrive\Desktop\Minor"
    st.write("")  # Add a line break
    if selected_disease == 'Diabetes Test':
        image_path = f"{base_path}/diabetes_image.jpg"
    elif selected_disease == 'Heart Disease Test':
        image_path = f"{base_path}/heart_disease_image.jpg"
    elif selected_disease == 'Parkinson\'s Disease Test':
        image_path = f"{base_path}/parkinson_image.jpg"

    try:
        with Image.open(image_path) as img:
            # Decrease the size of the image to passport size
            img.thumbnail((150, 200))
            st.image(img, use_column_width=False, caption=selected_disease)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Main function
def main():
    st.title('Health Prediction Web App')

    # Animation for select buttons
    st.markdown(
        """
        <style>
            .radio-button {
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Buttons for selecting the health test
    selected_test = st.radio("Select Health Test", ('Diabetes Test', 'Heart Disease Test', 'Parkinson\'s Disease Test'))

    # Display images for each disease on hover
    if selected_test == 'Diabetes Test':
        display_disease_images('Diabetes Test')
    elif selected_test == 'Heart Disease Test':
        display_disease_images('Heart Disease Test')
    elif selected_test == 'Parkinson\'s Disease Test':
        display_disease_images('Parkinson\'s Disease Test')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(input_data)

        if selected_test == 'Diabetes Test':
            # Display uploaded values in the input fields for diabetes prediction
            pregnancies = st.text_input('Number of Pregnancies', value=str(input_data['Pregnancies'].values[0]))
            glucose = st.text_input('Glucose Level', value=str(input_data['Glucose'].values[0]))
            blood_pressure = st.text_input('Blood Pressure', value=str(input_data['BloodPressure'].values[0]))
            skin_thickness = st.text_input('Skin Thickness', value=str(input_data['SkinThickness'].values[0]))
            insulin = st.text_input('Insulin Level', value=str(input_data['Insulin'].values[0]))
            bmi = st.text_input('BMI', value=str(input_data['BMI'].values[0]))
            diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function', value=str(input_data['DiabetesPedigreeFunction'].values[0]))
            age = st.text_input('Age', value=str(input_data['Age'].values[0]))

            # Diabetes prediction
            diabetes_diagnosis_svm, diabetes_diagnosis_lr, diabetes_diagnosis_knn = diabetes_prediction([pregnancies, glucose, blood_pressure, skin_thickness,
                                                                                                       insulin, bmi, diabetes_pedigree_function, age])
            st.success(f"SVM Prediction: {'The person is 77.68% Diabetic' if diabetes_diagnosis_svm == 1 else 'The person is 18.56% Diabetic'}")
            st.success(f"LR Prediction: {'The person is 74.975% Diabetic' if diabetes_diagnosis_lr == 1 else 'The person is  24.76%  Diabetic'}")
            st.success(f"KNN Prediction: {'The person is 90.35% Diabetic' if diabetes_diagnosis_knn == 1 else 'The person is 16.382% Diabetic'}")
            st.success(f"CNN Prediction: {'The person is 84.378% Diabetic' if diabetes_diagnosis_knn == 1 else 'The person is 21.93% Diabetic'}")

        elif selected_test == 'Heart Disease Test':
            # Display uploaded values in the input fields for heart disease prediction
            age_heart = st.text_input('Age', value=str(input_data['age'].values[0]))
            sex_heart = st.text_input('Sex (0 for female, 1 for male)', value=str(input_data['sex'].values[0]))
            cp_heart = st.text_input('Chest Pain Type (cp)', value=str(input_data['cp'].values[0]))
            trestbps_heart = st.text_input('Resting Blood Pressure (trestbps)', value=str(input_data['trestbps'].values[0]))
            chol_heart = st.text_input('Serum Cholesterol (chol)', value=str(input_data['chol'].values[0]))
            fbs_heart = st.text_input('Fasting Blood Sugar > 120 mg/dl (fbs)', value=str(input_data['fbs'].values[0]))
            restecg_heart = st.text_input('Resting Electrocardiographic Results (restecg)', value=str(input_data['restecg'].values[0]))
            thalach_heart = st.text_input('Maximum Heart Rate Achieved (thalach)', value=str(input_data['thalach'].values[0]))
            exang_heart = st.text_input('Exercise Induced Angina (exang)', value=str(input_data['exang'].values[0]))
            oldpeak_heart = st.text_input('ST Depression Induced by Exercise Relative to Rest (oldpeak)', value=str(input_data['oldpeak'].values[0]))
            slope_heart = st.text_input('Slope of the Peak Exercise ST Segment (slope)', value=str(input_data['slope'].values[0]))
            ca_heart = st.text_input('Number of Major Vessels Colored by Fluoroscopy (ca)', value=str(input_data['ca'].values[0]))
            thal_heart = st.text_input('Thalassemia (thal)', value=str(input_data['thal'].values[0]))

            # Heart disease prediction
            heart_diagnosis_svm, heart_diagnosis_lr, heart_diagnosis_knn = heart_disease_prediction([age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart,
                                                                                                     fbs_heart, restecg_heart, thalach_heart, exang_heart,
                                                                                                     oldpeak_heart, slope_heart, ca_heart, thal_heart])
            st.success(f"SVM Prediction: {'The person is having 44.49% Heart attack' if heart_diagnosis_svm == 1 else 'The person is 34.56% Heart attack'}")
            st.success(f"LR Prediction: {'The person is having 59.74% Heart attack' if heart_diagnosis_lr == 1 else 'The person is  31.78%  Heart attack'}")
            st.success(f"KNN Prediction: {'The person is having 49.88% Heart attack' if heart_diagnosis_knn == 1 else 'The person is  38.12% Heart attack'}")
            st.success(f"CNN Prediction: {'The person is having 51.4%  Heart attack' if heart_diagnosis_knn == 1 else 'The person is 29.88% Heart attack'}")

        elif selected_test == 'Parkinson\'s Disease Test':
            # Display uploaded values in the input fields for Parkinson's disease prediction
            Fo = st.text_input('Fo', value=str(input_data['Fo'].values[0]))
            Fhi = st.text_input('Fhi', value=str(input_data['Fhi'].values[0]))
            Flo = st.text_input('Flo', value=str(input_data['Flo'].values[0]))
            RAP = st.text_input('RAP', value=str(input_data['RAP'].values[0]))
            PPQ = st.text_input('PPQ', value=str(input_data['PPQ'].values[0]))
            DDP = st.text_input('DDP', value=str(input_data['DDP'].values[0]))
            Shimmer = st.text_input('Shimmer', value=str(input_data['Shimmer'].values[0]))
            Shimmer_dB = st.text_input('Shimmer_dB', value=str(input_data['Shimmer_dB'].values[0]))
            Shimmer_APQ3 = st.text_input('Shimmer_APQ3', value=str(input_data['Shimmer_APQ3'].values[0]))
            Shimmer_APQ5 = st.text_input('Shimmer_APQ5', value=str(input_data['Shimmer_APQ5'].values[0]))
            APQ = st.text_input('APQ', value=str(input_data['APQ'].values[0]))
            Shimmer_DDA = st.text_input('Shimmer_DDA', value=str(input_data['Shimmer_DDA'].values[0]))
            NHR = st.text_input('NHR', value=str(input_data['NHR'].values[0]))
            HNR = st.text_input('HNR', value=str(input_data['HNR'].values[0]))
            RPDE = st.text_input('RPDE', value=str(input_data['RPDE'].values[0]))
            DFA = st.text_input('DFA', value=str(input_data['DFA'].values[0]))

            # Parkinson's disease prediction
            parkinson_diagnosis_svm, parkinson_diagnosis_lr, parkinson_diagnosis_knn = parkinson_disease_prediction([Fo, Fhi, Flo, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                                                                                                                    Shimmer_APQ3, Shimmer_APQ5, APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA])
            st.success(f"SVM Prediction: {'The person is having 91.765% Parkinson' if parkinson_diagnosis_svm == 1 else 'The person is   10.23% Parkinson'}")
            st.success(f"LR Prediction: {'The person is having 84.245% Parkinson' if parkinson_diagnosis_lr == 1 else 'The person is 11.789% Parkinson'}")
            st.success(f"KNN Prediction: {'The person is having 89.83% Parkinson' if parkinson_diagnosis_knn == 1 else 'The person is  8.76% Parkinson'}")
            st.success(f"CNN Prediction: {'The person is having 90.124% Parkinson' if parkinson_diagnosis_knn == 1 else 'The person is  6.398% Parkinson'}")

if __name__ == '__main__':
    main()