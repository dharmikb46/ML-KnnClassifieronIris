# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:43:40 2022

@author: Rinku
"""

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
model = open("knn_classifier.pkl","rb")
knn_clf = joblib.load(model)

st.title("Iris Flower Species Classification App")
st.sidebar.title("Features")
setosa = Image.open("Irissetosa.jpg")
versicolor = Image.open("Iris_versicolor.jpg")
virginica = Image.open("Iris_virginica_2.jpg")
parameter_list=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
parameter_input_values=[]
parameter_default_values = ['5.2','3.2','4.2','1.2']
values=[]

for parameter, parameter_df in zip(parameter_list,parameter_default_values):
    values = st.sidebar.slider(label=parameter,
                               key=parameter,
                               value=float(parameter_df),
                               min_value=0.0,
                               max_value=8.0,step=0.1)
    parameter_input_values.append(values)
    
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)

st.write(input_variables)
prediction = 0
if st.button("Click Here To Classify"):
    prediction = knn_clf.predict(input_variables)

if prediction == 0:
    st.image(setosa)
elif prediction == 1:
    st.image(versicolor)
else:
    st.image(virginica)
    




