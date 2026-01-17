import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. Configuración de página con estilo "Wide"
st.set_page_config(page_title="CarPrice Pro Dashboard", page_icon="🏎️", layout="wide")

# CSS personalizado para mejorar la estética
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div.stButton > button:first-child {
        background-color: #007bff; color: white; border-radius: 8px; height: 3em; width: 100%; font-weight: bold;
    }
    .prediction-card {
        padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; text-align: center; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Carga y Procesamiento de Datos (Optimizado)
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('car_price_prediction.csv')
    # Limpieza básica de strings a números
    df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(float)
    df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '').astype(float)
    # Filtramos valores extremos para que los gráficos se vean mejor
    df_clean = df[(df['Price'] > 500) & (df['Price'] < 100000)].copy()
    return df_clean

@st.cache_resource
def train_model(df):
    # Seleccionamos las variables clave
    features = ['Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Fuel type', 'Gear box type']
    X = df[features].copy()
    y = df['Price']
    
    # Encoders para texto
    le_fuel = LabelEncoder()
    X['Fuel type'] = le_fuel.fit_transform(X['Fuel type'])
    le_gear = LabelEncoder()
    X['Gear box type'] = le_gear.fit_transform(X['Gear box type'])
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_fuel, le_gear

# Inicializar datos y modelo
df = load_and_clean_data()
model, le_fuel, le_gear = train_model(df)

# --- SIDEBAR (Entradas de Usuario) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/744/744465.png", width=100)
st.sidebar.title("Configuración")
st.sidebar.markdown("Ajusta los detalles del vehículo:")

with st.sidebar:
    in_year = st.slider("Año de Fabricación", int(df['Prod. year'].min()), 2024, 2018)
    in_engine = st.number_input("Cilindrada (Engine Vol)", 0.5, 8.0, 2.0, step=0.1)
    in_mileage = st.number_input("Kilometraje (km)", 0, 500000, 50000)
    in_cyl = st.selectbox("Número de Cilindros", sorted(df['Cylinders'].unique().tolist()), index=3)
    in_fuel = st.selectbox("Tipo de Combustible", le_fuel.classes_)
    in_gear = st.selectbox("Transmisión", le_gear.classes_)
    
    predict_clicked = st.button("CALCULAR PRECIO")

# --- ÁREA PRINCIPAL ---
st.title("🏎️ CarPrice Pro: Inteligencia Predictiva")
st.markdown("Dashboard interactivo para la estimación de valores de mercado automotriz.")

# Fila 1: Métricas y Predicción
col1, col2, col3 = st.columns([1, 1, 1.5])

with col1:
    st.metric("Modelos Analizados", f"{len(df):,}")
with col2:
    st.metric("Precio Promedio", f"${df['Price'].mean():,.0f}")

with col3:
    if predict_clicked:
        # Preparar entrada para el modelo
        input_df = pd.DataFrame([[
            in_year, in_engine, in_mileage, in_cyl,
            le_fuel.transform([in_fuel])[0],
            le_gear.transform([in_gear])[0]
        ]], columns=['Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Fuel type', 'Gear box type'])
        
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"""
            <div class="prediction-card">
                <p style="margin:0; font-size:1.2em;">PRECIO ESTIMADO</p>
                <h1 style="margin:0; color: #00ffcc;">${prediction:,.2f}</h1>
                <p style="margin:0; font-size:0.8em; opacity:0.8;">Basado en Inteligencia Artificial</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Presiona el botón en la barra lateral para ver la predicción.")

st.markdown("---")

# Fila 2: Visualizaciones Dinámicas
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Tendencia: Año vs Precio")
    fig_scatter = px.scatter(df.sample(2000), x="Prod. year", y="Price", color="Fuel type",
                             hover_data=['Manufacturer'], template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📊 Distribución por Cilindros")
    fig_box = px.box(df, x="Cylinders", y="Price", color="Cylinders",
                     template="plotly_white", title="Rango de Precios según Potencia")
    st.plotly_chart(fig_box, use_container_width=True)

# Fila 3: Explorador de Datos
with st.expander("📂 Explorar Base de Datos Completa"):
    st.dataframe(df.head(100), use_container_width=True)
    st.caption("Mostrando los primeros 100 registros del dataset original procesado.")
