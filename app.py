import numpy as np
import joblib
import streamlit as st
from PIL import Image

# loading the saved model
loaded_model  = joblib.load("kmeans_model.joblib")


def Cluster_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'Developing Country'
    elif (prediction[0] == 1):
      return 'Developed Country'
    else:
      return 'Under Developed Country'  


def main():
    
    # giving a title
    st.title('Cluster Prediction')
    
    # getting the input data from the user
    
    Birth_Rate = st.slider('Birth Rate', 0.007, 0.053, 0.01)
    CO2_Emissions = st.text_input('CO2_Emissions')
    Days_to_Start_Business = st.slider('Days to Start Business', 1, 694, 10)
    GDP = st.text_input('Total GDP')
    Health_Exp_Capita = st.slider('Health Exp/Capita', 2, 9908, 100)
    Health_Exp_GDP = st.slider('Health Exp % GDP', 0.02, 0.9, 0.45)
    Infant_Mortality_Rate = st.slider('Infant Mortality Rate', 0.002, 0.141, 0.05)
    Internet_Usage = st.slider('Internet Usage', 0.0, 1.0, 0.1)
    Life_Expectancy_Female = st.slider('Life Expectancy Female', 1, 99, 70)
    Life_Expectancy_Male = st.slider('Life Expectancy Male', 1, 99, 70)
    Mobile_Phone_Usage = st.slider('Mobile Phone Usage', 0.0, 2.9, 1.0)
    Population_0_14 = st.slider('Population 0-14 %', 0.1, 0.5, 0.2)
    Population_15_64 = st.slider('Population 15-64 %', 0.2, 1.0, 0.4)
    Population_65_and_above = st.slider('Population 65+', 0.001, 0.5, 0.05)
    Population_Total = st.text_input('Total population')
    Population_Urban = st.slider('Population Urban %', 0.082, 1.0, 0.5)
    Tourism_Inbound = st.text_input('$ earned in tourism')

    # code for Prediction
    Predict = ''
    
    # creating a button for Prediction
    
    if st.button('Submit'):
        Predict = Cluster_prediction([Birth_Rate,CO2_Emissions,Days_to_Start_Business, GDP,Health_Exp_Capita, Health_Exp_GDP, Infant_Mortality_Rate, Internet_Usage,Life_Expectancy_Female, Life_Expectancy_Male, Mobile_Phone_Usage, Population_0_14, Population_15_64,Population_65_and_above, Population_Total, Population_Urban,Tourism_Inbound])

    st.success(Predict)
    
if __name__ == '__main__':
    main()