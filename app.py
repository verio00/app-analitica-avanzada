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
    page_title="Clasificador Inteligente de Iris",
    page_icon="🌸",
    layout="wide"
)

# Estilo CSS personalizado para mejorar tarjetas y espaciados
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Título principal limpio y directo
st.title("🌸 Dashboard de Clasificación: Botánica Inteligente")
st.markdown("Modelo interactivo de Machine Learning impulsado por **Random Forest**.")

# Cargar dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
df_iris = pd.DataFrame(X, columns=iris.feature_names)
df_iris['Especie'] = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Panel Lateral de Control
st.sidebar.header("🎛️ Panel de Control")
st.sidebar.markdown("Ajusta las dimensiones de la flor:")

def user_input_features():
    sepal_length = st.sidebar.slider("Longitud Sépalo (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider("Ancho Sépalo (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider("Longitud Pétalo (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider("Ancho Pétalo (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

df_input = user_input_features()

# Predicciones
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)[0]
especie_ganadora = iris.target_names[prediction[0]].capitalize()
confianza = np.max(prediction_proba) * 100

# Cuerpo Principal en Columnas
col1, col2 = st.columns([1.2, 1.8], gap="large")

with col1:
    st.subheader("🎯 Resultado de la Predicción")
    
    # Tarjeta de resultado principal
    st.success(f"### Especie Detectada: **{especie_ganadora}**")
    st.metric(label="Confianza del Modelo", value=f"{confianza:.1f}%", delta="Alta precisión")
    
    st.markdown("---")
    st.markdown("#### 📋 Valores ingresados:")
    df_user_display = pd.DataFrame(df_input, columns=["Sépalo Largo", "Sépalo Ancho", "Pétalo Largo", "Pétalo Ancho"])
    st.dataframe(df_user_display, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Distribución Espacial del Dataset")
    
    # Gráfico interactivo moderno con Plotly
    fig = px.scatter(
        df_iris, 
        x="petal length (cm)", 
        y="petal width (cm)", 
        color="Especie",
        color_discrete_map={'Setosa': '#ff7b72', 'Versicolor': '#ffa657', 'Virginica': '#79c0ff'},
        title="Relación: Longitud vs Ancho de Pétalos",
        template="plotly_dark"
    )
    
    # Añadir el punto actual del usuario en el gráfico
    fig.add_add_annotation = True
    fig.add_scatter(
        x=[df_input[0][2]], 
        y=[df_input[0][3]], 
        mode='markers',
        marker=dict(size=18, color='yellow', symbol='star'),
        name='Tu Selección'
    )
    
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Pie de página ejecutivo
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.info(f"🛡️ **Precisión General del Algoritmo:** `{acc * 100:.1f}%`")
with col_f2:
    st.caption("✨ Desarrollado para Analítica Avanzada | Desplegado en Streamlit Cloud")
