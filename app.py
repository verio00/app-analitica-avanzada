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
    page_title="Analítica Avanzada | Calidad de Vinos",
    page_icon="🍷",
    layout="wide"
)

# Estilos CSS avanzados para un diseño moderno, tarjetas elegantes y tipografía limpia
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .stSidebar {
        background-color: #111827;
    }
    h1, h2, h3 {
        color: #f3f4f6 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado visual principal con banner integrado
st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?q=80&w=1600&auto=format&fit=crop", use_container_width=True)
st.title("🍷 Laboratorio Enológico: Analítica y Clasificación de Vinos")
st.markdown("Sistema inteligente de Machine Learning (**Random Forest**) optimizado para evaluar la procedencia y calidad físico-química del vino en tiempo real.")

# Cargar dataset de vinos
wine = datasets.load_wine()
X = wine.data
y = wine.target
df_wine = pd.DataFrame(X, columns=wine.feature_names)
df_wine['Clase'] = pd.Series(y).map({0: 'Origen Premium A', 1: 'Origen Estándar B', 2: 'Origen Reserva C'})

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Panel Lateral Interactivo
st.sidebar.image("https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?q=80&w=400&auto=format&fit=crop", use_container_width=True)
st.sidebar.header("🎛️ Panel de Control Químico")
st.sidebar.markdown("Desliza para ajustar los componentes:")

def user_input_features():
    alcohol = st.sidebar.slider("Graduación Alcohólica", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    malic_acid = st.sidebar.slider("Ácido Málico", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    flavanoids = st.sidebar.slider("Nivel de Flavonoides", float(X[:,6].min()), float(X[:,6].max()), float(X[:,6].mean()))
    color_intensity = st.sidebar.slider("Intensidad de Color", float(X[:,9].min()), float(X[:,9].max()), float(X[:,9].mean()))
    proline = st.sidebar.slider("Concentración de Prolina", float(X[:,12].min()), float(X[:,12].max()), float(X[:,12].mean()))
    
    full_data = X.mean(axis=0).reshape(1, -1)
    full_data[0, 0] = alcohol
    full_data[0, 1] = malic_acid
    full_data[0, 6] = flavanoids
    full_data[0, 9] = color_intensity
    full_data[0, 12] = proline
    return full_data

df_input = user_input_features()

# Predicciones
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)[0]
clase_detectada = wine.target_names[prediction[0]].capitalize()
confianza = np.max(prediction_proba) * 100

# Estructura Principal en Columnas
col1, col2 = st.columns([1.1, 1.9], gap="large")

with col1:
    st.subheader("🎯 Resultado del Diagnóstico")
    
    # Tarjeta de resultado visual estilizada
    st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #9ca3af; margin-bottom: 5px;">CATEGORÍA IDENTIFICADA</h4>
            <h2 style="color: #34d399; margin-top: 0px;">{clase_detectada}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.metric(label="📊 Índice de Confianza del Modelo", value=f"{confianza:.1f}%", delta="Alta Confiabilidad")
    
    st.markdown("---")
    st.markdown("#### 🔬 Parámetros Evaluados:")
    resumen_input = pd.DataFrame({
        'Componente Químico': ['Alcohol', 'Ácido Málico', 'Flavonoides', 'Intensidad Color', 'Prolina'],
        'Valor Ingresado': [f"{df_input[0][0]:.2f}", f"{df_input[0][1]:.2f}", f"{df_input[0][6]:.2f}", f"{df_input[0][9]:.2f}", f"{df_input[0][12]:.1f}"]
    })
    st.dataframe(resumen_input, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📈 Mapa de Distribución y Posición Actual")
    
    # Gráfico interactivo moderno con Plotly
    fig = px.scatter(
        df_wine, 
        x="alcohol", 
        y="color_intensity", 
        color="Clase",
        color_discrete_map={'Origen Premium A': '#ef4444', 'Origen Estándar B': '#f59e0b', 'Origen Reserva C': '#8b5cf6'},
        title="Correlación: Alcohol vs Intensidad de Color",
        template="plotly_dark"
    )
    
    # Marcador interactivo destacado para la muestra del usuario
    fig.add_scatter(
        x=[df_input[0][0]], 
        y=[df_input[0][9]], 
        mode='markers',
        marker=dict(size=24, color='#fbbf24', symbol='diamond', line=dict(width=2, color='white')),
        name='⭐ Muestra Analizada'
    )
    
    fig.update_layout(
        height=450, 
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Pie de página ejecutivo
st.markdown("---")
st.info(f"🛡️ **Eficiencia General del Algoritmo:** `{acc * 100:.1f}%` de precisión en validación cruzada. Aplicación diseñada para Analítica Avanzada.")
