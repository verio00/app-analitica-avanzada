import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Analítica Avanzada - Clasificador Iris",
    page_icon="🌸",
    layout="wide"
)

# Estilo visual y título principal
st.title("🌸 Aplicación Interactiva de Analítica Avanzada")
st.markdown("""
Esta aplicación utiliza Machine Learning (**Random Forest**) entrenado con el clásico **Dataset de Iris** para clasificar el tipo de flor según sus características físicas. ¡Mueve los controles de la izquierda para probar el modelo en tiempo real!
""")

# Cargar dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
df_iris = pd.DataFrame(X, columns=iris.feature_names)
df_iris['target'] = y
df_iris['especie'] = df_iris['target'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Panel Lateral (Sliders)
st.sidebar.header("🎛️ Parámetros de Entrada")
st.sidebar.markdown("Modifica las medidas para realizar una predicción:")

def user_input_features():
    sepal_length = st.sidebar.slider("Longitud del Sépalo (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider("Ancho del Sépalo (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider("Longitud del Pétalo (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider("Ancho del Pétalo (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return data

df_input = user_input_features()

# Predicción
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)

# ---- CUERPO PRINCIPAL EN COLUMNAS ----
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Resultados de la Predicción")
    
    # Métrica visual destacada
    flor_predicha = iris.target_names[prediction][0].upper()
    st.metric(label="Especie Identificada", value=flor_predicha)
    
    st.write("Valores que ingresaste:")
    df_user_display = pd.DataFrame(df_input, columns=iris.feature_names)
    st.dataframe(df_user_display, use_container_width=True)
    
    st.write("Probabilidades por especie:")
    df_proba = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.dataframe(df_proba, use_container_width=True)

with col2:
    st.subheader("📈 Visualización del Dataset")
    # Gráfico interactivo con Plotly
    fig = px.scatter(
        df_iris, 
        x="sepal length (cm)", 
        y="sepal width (cm)", 
        color="especie",
        title="Distribución del Dataset Iris (Sépalo)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info(f"💡 **Precisión global del modelo evaluado en pruebas:** `{acc * 100:.2f}%`")
