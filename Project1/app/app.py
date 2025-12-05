import streamlit as st
from predict import predict

st.title("Iris Flower Classifier")
st.write("Enter the flower measurements")

sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    species = predict(features)
    st.success(f"Predicted Iris species: **{species}**")