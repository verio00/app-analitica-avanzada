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
    page_title="Analítica de Vinos | Machine Learning",
    page_icon="🍷",
    layout="wide"
)

# Estilo visual moderno y limpio
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado principal estilo ejecutivo
st.title("🍷 Panel Ejecutivo: Clasificación y Calidad de Vinos")
st.markdown("Sistema de analítica avanzada impulsado por **Random Forest** para evaluar la procedencia y perfil químico de los vinos.")

# Cargar dataset de vinos
wine = datasets.load_wine()
X = wine.data
y = wine.target
df_wine = pd.DataFrame(X, columns=wine.feature_names)
df_wine['Clase'] = pd.Series(y).map({0: 'Origen A', 1: 'Origen B', 2: 'Origen C'})

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Panel Lateral de Control con parámetros clave del vino
st.sidebar.header("🎛️ Perfil Químico del Vino")
st.sidebar.markdown("Ajusta los componentes clave:")

def user_input_features():
    alcohol = st.sidebar.slider("Alcohol", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    malic_acid = st.sidebar.slider("Ácido Málico", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    flavanoids = st.sidebar.slider("Flavonoides", float(X[:,6].min()), float(X[:,6].max()), float(X[:,6].mean()))
    color_intensity = st.sidebar.slider("Intensidad de Color", float(X[:,9].min()), float(X[:,9].max()), float(X[:,9].mean()))
    proline = st.sidebar.slider("Prolina", float(X[:,12].min()), float(X[:,12].max()), float(X[:,12].mean()))
    
    # Rellenar el resto de variables con las medias para mantener la compatibilidad del modelo
    full_data = X.mean(axis=0).reshape(1, -1)
    full_data[0, 0] = alcohol
    full_data[0, 1] = malic_acid
    full_data[0, 6] = flavanoids
    full_data[0, 9] = color_intensity
    full_data[0, 12] = proline
    return full_data

df_input = user_input_features()

# Predicción
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)[0]
clase_detectada = wine.target_names[prediction[0]].capitalize()
confianza = np.max(prediction_proba) * 100

# Estructura principal en dos columnas limpias
col1, col2 = st.columns([1.2, 1.8], gap="large")

with col1:
    st.subheader("🎯 Diagnóstico del Modelo")
    
    # Tarjeta de resultado visual
    st.success(f"### Categoría Detectada: **{clase_detectada}**")
    st.metric(label="Nivel de Confianza", value=f"{confianza:.1f}%", delta="Modelo Optimizado")
    
    st.markdown("---")
    st.markdown("#### 🧪 Principales Parámetros Analizados:")
    resumen_input = pd.DataFrame({
        'Componente': ['Alcohol', 'Ácido Málico', 'Flavonoides', 'Intensidad Color', 'Prolina'],
        'Valor': [df_input[0][0], df_input[0][1], df_input[0][6], df_input[0][9], df_input[0][12]]
    })
    st.dataframe(resumen_input, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Análisis Multivariable del Dataset")
    
    # Gráfico interactivo moderno de Plotly con colores elegantes
    fig = px.scatter(
        df_wine, 
        x="alcohol", 
        y="color_intensity", 
        color="Clase",
        color_discrete_map={'Origen A': '#e53935', 'Origen B': '#fb8c00', 'Origen C': '#8e24aa'},
        title="Relación: Alcohol vs Intensidad de Color por Origen",
        template="plotly_dark"
    )
    
    # Añadir marcador de la selección actual del usuario en forma de diamante dorado
    fig.add_scatter(
        x=[df_input[0][0]], 
        y=[df_input[0][9]], 
        mode='markers',
        marker=dict(size=22, color='gold', symbol='diamond'),
        name='Muestra Analizada'
    )
    
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Pie de página profesional
st.markdown("---")
st.info(f"🛡️ **Precisión Global del Clasificador de Vinos:** `{acc * 100:.1f}%` | Desarrollado para Analítica Avanzada")
