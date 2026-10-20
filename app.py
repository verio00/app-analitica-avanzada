import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

st.title("Aplicación de Analítica Avanzada - Clasificador de Iris")
st.write("Esta aplicación despliega un modelo de Machine Learning entrenado con el dataset de flores Iris.")

# Cargar dataset de Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir datos para entrenar y evaluar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluar precisión
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Precisión del modelo en prueba:** {acc * 100:.2f}%")

st.sidebar.header("Parámetros de Entrada")

def user_input_features():
    sepal_length = st.sidebar.slider("Longitud del Sépalo (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider("Ancho del Sépalo (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider("Longitud del Pétalo (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider("Ancho del Pétalo (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return data

df_input = user_input_features()

st.subheader("Valores seleccionados por el usuario:")
st.write(pd.DataFrame(df_input, columns=iris.feature_names))

# Predicción
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)

st.subheader("Predicción del Modelo:")
st.write(iris.target_names[prediction][0])

st.subheader("Probabilidades de Predicción:")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
