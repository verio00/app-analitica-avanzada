import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import plotly.express as px

# Configuración de página minimalista
st.set_page_config(
    page_title="Wine Analytics Lab",
    page_icon="🍷",
    layout="wide"
)

# Estilos CSS avanzados para un diseño estético y moderno
st.markdown("""
    <style>
    /* Ocultar elementos predeterminados de Streamlit para limpieza visual */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fondo general */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tarjetas contenedoras estéticas */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Estilo para los títulos */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado limpio y estético
st.markdown("""
    <div style="padding: 10px 0px 20px 0px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 5px;">🍷 Wine Quality Analytics</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">Plataforma inteligente de clasificación y perfil enológico basada en Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

# Cargar dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target
df_wine = pd.DataFrame(X, columns=wine.feature_names)
df_wine['Clase'] = pd.Series(y).map({0: 'Gran Reserva A', 1: 'Estándar B', 2: 'Selección C'})

# Modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Panel Lateral Estilizado
st.sidebar.markdown("## 🎛️ Panel Enológico")
st.sidebar.markdown("---")

def user_input_features():
    alcohol = st.sidebar.slider("Graduación Alcohólica", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    malic_acid = st.sidebar.slider("Ácido Málico", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    flavanoids = st.sidebar.slider("Flavonoides", float(X[:,6].min()), float(X[:,6].max()), float(X[:,6].mean()))
    color_intensity = st.sidebar.slider("Intensidad de Color", float(X[:,9].min()), float(X[:,9].max()), float(X[:,9].mean()))
    proline = st.sidebar.slider("Prolina", float(X[:,12].min()), float(X[:,12].max()), float(X[:,12].mean()))
    
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

# Distribución en columnas limpias
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #38bdf8 !important;">🎯 Resultado del Diagnóstico</h3>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h2 style='color: #4ade80; margin: 10px 0;'>{clase_detectada}</h2>", unsafe_allow_html=True)
    st.metric(label="Nivel de Confianza Predictiva", value=f"{confianza:.1f}%")
    
    st.markdown("---")
    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Valores evaluados en la muestra:</p>", unsafe_allow_html=True)
    
    resumen_input = pd.DataFrame({
        'Parámetro': ['Alcohol', 'Ácido Málico', 'Flavonoides', 'Color', 'Prolina'],
        'Valor': [f"{df_input[0][0]:.2f}", f"{df_input[0][1]:.2f}", f"{df_input[0][6]:.2f}", f"{df_input[0][9]:.2f}", f"{df_input[0][12]:.1f}"]
    })
    st.dataframe(resumen_input, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #38bdf8 !important;">📊 Espacio Multivariable</h3>
    """, unsafe_allow_html=True)
    
    fig = px.scatter(
        df_wine, 
        x="alcohol", 
        y="color_intensity", 
        color="Clase",
        color_discrete_map={'Gran Reserva A': '#f43f5e', 'Estándar B': '#fbbf24', 'Selección C': '#a855f7'},
        template="plotly_dark"
    )
    
    fig.add_scatter(
        x=[df_input[0][0]], 
        y=[df_input[0][9]], 
        mode='markers',
        marker=dict(size=24, color='#38bdf8', symbol='star', line=dict(width=2, color='white')),
        name='Muestra Actual'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"""
    <div style="text-align: center; color: #64748b; padding: 10px;">
        Precisión global del modelo: <b>{acc * 100:.1f}%</b> • Desarrollado con Streamlit & Scikit-Learn
    </div>
""", unsafe_allow_html=True)
