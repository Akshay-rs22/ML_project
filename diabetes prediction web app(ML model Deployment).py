# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 19:36:54 2022

@author: Akshay
"""

import numpy as np
import pickle
import streamlit as st


# loading a save model
loaded_model = pickle.load(open('C:/Users/Akshay/OneDrive/Desktop/ML/trained_model.sav','rb'))

#creating function for prediction
def Diabetes_prediction(input_data):
    

    #changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    #reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
    #standardize the input data
    #data=scaler.transform(input_data_reshape)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)
    if prediction[0]== 0:
      return 'The person is non-diabetes'
    else:
      return 'The person is diabetes'
  
    
def main():
    
    #create a title
    st.title("Diabetes Prediction web App")
    
    #getting input data from user
    
    
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("BloodPressure Value")
    SkinThickness = st.text_input("SkinThickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    Age = st.text_input(",Age of Person")
    
    
    #code for prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = Diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
    
    
if __name__=='__main__':
    main()
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
