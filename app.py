import streamlit as st
import pandas as pd 
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open('student_lr_final_model.pkl','rb') as  file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocess_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])
    df = pd.DataFrame([data])
    df_transfromed =  scaler.transform(df)
    return df_transfromed

def predict_performance(data):
    model, scaler, le = load_model()
    processed_data = preprocess_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title('Student Performance Prediction')
    st.subheader('Enter the student details')
    hour_studied = st.number_input('hours stuied',min_value=1, max_value=10,value=5)
    previous_score = st.number_input('previous score',min_value=40, max_value=100,value=70)
    extra = st.selectbox('Extracurricular Activities',['Yes','No'])
    sleeping_hours = st.number_input('sleep hours',min_value=4, max_value=10,value=7)
    no_of_qp = st.number_input('number of question paper',min_value=0, max_value=10,value=5)
    if st.button('Predict'):
        u_data = {'Hours Studied':hour_studied,
                'Previous Scores':previous_score,
                'Extracurricular Activities':extra,
                'Sleep Hours':sleeping_hours,
                'Sample Question Papers Practiced':no_of_qp}
        prediction  = predict_performance(u_data)
        st.success(f"this prediction is :{round(prediction[0][0],2)}")

if __name__ == '__main__':
    main()

