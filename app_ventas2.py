import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
import requests
from io import BytesIO
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
import locale
import os

# ==================================================
# CONFIGURACIÓN INICIAL
# ==================================================

# Configuración de locale para evitar errores
def configure_locale():
    locale_options = ['en_US.UTF-8', 'C.UTF-8', 'en_US.utf8', 'en_US', 'C', 'POSIX']
    for loc in locale_options:
        try:
            locale.setlocale(locale.LC_ALL, loc)
            os.environ['LC_ALL'] = loc
            os.environ['LANG'] = loc
            os.environ['LANGUAGE'] = loc
            break
        except (locale.Error, Exception):
            continue

configure_locale()

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Dashboard Comercial Integral",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .metric-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 15px;
    }
    @media (max-width: 768px) {
        .stMetric { padding: 5px !important; }
        .stDataFrame { font-size: 12px !important; }
        .stPlotlyChart { height: 300px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# FUNCIÓN DE CARGA DE DATOS MEJORADA
# ==================================================

@st.cache_data(ttl=3600)
def load_data():
    """Carga y procesa los datos con manejo robusto de errores"""
    try:
        # Configuración adicional de locale
        configure_locale()
        
        # Descarga del archivo con timeout
        file_id = "1i53R94PaYc9GmEhM1zAdP0Wx0OlVJSFZ"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Carga del Excel con motor explícito
        with BytesIO(response.content) as file:
            df = pd.read_excel(
                file,
                sheet_name="Hoja2",
                engine='openpyxl',
                dtype={
                    'Order Lines/Untaxed Invoiced Amount': str,
                    'Order Lines/Created on': str
                }
            )
        
        # Limpieza de columnas
        df.columns = [col.strip().replace('\n', ' ') for col in df.columns]
        
        # Conversión de montos segura
        amount_col = 'Order Lines/Untaxed Invoiced Amount'
        df[amount_col] = (
            df[amount_col]
            .astype(str)
            .str.extract(r'([\d\.]+)', expand=False)
            .fillna('0')
            .astype(float)
        )
        
        # Manejo de fechas robusto
        date_col = 'Order Lines/Created on'
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors='coerce',
            utc=True,
            format='mixed'
        )
        df = df.dropna(subset=[date_col])
        
        # Mapeo de días de la semana sin locale
        weekday_map = {
            0: "Lunes", 1: "Martes", 2: "Miércoles",
            3: "Jueves", 4: "Viernes", 5: "Sábado",
            6: "Domingo"
        }
        
        # Campos derivados
        df['Año'] = df[date_col].dt.year
        df['Mes'] = df[date_col].dt.month
        df['Mes-Año'] = df[date_col].dt.strftime('%Y-%m')
        df['DíaSemana'] = df[date_col].dt.weekday.map(weekday_map)
        df['Hora'] = df[date_col].dt.hour
        df['Trimestre'] = df[date_col].dt.quarter
        
        return df
    
    except Exception as e:
        st.error(f"Error crítico al cargar datos: {str(e)}")
        return pd.DataFrame()

# ==================================================
# FUNCIÓN CALCULAR RFM CORREGIDA
# ==================================================

def calculate_rfm(df_input):
    """Calcula métricas RFM con manejo robusto de errores y nombres de columnas consistentes"""
    try:
        # Verificar DataFrame vacío
        if df_input.empty:
            st.warning("DataFrame vacío recibido para cálculo RFM")
            return None

        # Verificar columnas requeridas
        required_cols = [
            'Order Lines/Customer/Company Name',
            'Order Lines/Created on',
            'Order Lines/Invoice Lines/Number',
            'Order Lines/Untaxed Invoiced Amount'
        ]
        
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        if missing_cols:
            st.error(f"Columnas faltantes para RFM: {', '.join(missing_cols)}")
            return None

        # Asegurar formato de fecha
        if not pd.api.types.is_datetime64_any_dtype(df_input['Order Lines/Created on']):
            df_input['Order Lines/Created on'] = pd.to_datetime(
                df_input['Order Lines/Created on'],
                errors='coerce',
                utc=True
            )
            df_input = df_input.dropna(subset=['Order Lines/Created on'])
            if df_input.empty:
                st.warning("No hay fechas válidas después de la limpieza")
                return None

        # Calcular RFM con nombres de columnas explícitos
        now = pd.Timestamp.now(tz='UTC')
        
        rfm = df_input.groupby('Order Lines/Customer/Company Name').agg(
            Recencia=('Order Lines/Created on', lambda x: (now - x.max()).days),
            Frecuencia=('Order Lines/Invoice Lines/Number', 'nunique'),
            ValorMonetario=('Order Lines/Untaxed Invoiced Amount', 'sum'),
            TicketPromedio=('Order Lines/Untaxed Invoiced Amount', 'mean')
        ).reset_index()
        
        # Renombrar columna de cliente para consistencia
        rfm = rfm.rename(columns={'Order Lines/Customer/Company Name': 'Cliente'})
        
        # Manejar valores nulos
        rfm['Recencia'] = rfm['Recencia'].fillna(365)  # 1 año como valor por defecto
        rfm['ValorMonetario'] = rfm['ValorMonetario'].fillna(0)
        rfm['TicketPromedio'] = rfm['TicketPromedio'].fillna(0)
        
        return rfm
    
    except Exception as e:
        st.error(f"Error inesperado en cálculo RFM: {str(e)}")
        return None

# ==================================================
# CARGA DE DATOS Y FILTROS
# ==================================================

# Cargar datos
df = load_data()

if df.empty:
    st.warning("""
    No se pudieron cargar los datos principales. Opciones:
    1. Verificar conexión a Internet
    2. Comprobar acceso al archivo
    3. Subir archivo manualmente
    """)
    
    uploaded_file = st.file_uploader("Subir archivo Excel", type=['xlsx'])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="Hoja2", engine='openpyxl')
            st.success("Archivo cargado correctamente!")
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")
            st.stop()
    else:
        st.stop()

# Filtros en sidebar
with st.sidebar:
    st.header("⚙️ Filtros")
    
    # Filtro de fechas
    min_date = df['Order Lines/Created on'].min().date()
    max_date = df['Order Lines/Created on'].max().date()
    date_range = st.date_input("Rango de fechas", [min_date, max_date])
    
    # Filtro de asesores
    asesores = sorted(df['Order Lines/Customer/Asesor (Gestor)'].dropna().unique())
    asesores_seleccionados = st.multiselect("Asesores", asesores, default=asesores)
    
    # Filtro de montos
    min_venta, max_venta = st.slider(
        "Rango de valor de compra ($)",
        float(df['Order Lines/Untaxed Invoiced Amount'].min()),
        float(df['Order Lines/Untaxed Invoiced Amount'].max()),
        (float(df['Order Lines/Untaxed Invoiced Amount'].min()), 
        float(df['Order Lines/Untaxed Invoiced Amount'].max()))
    )

# Aplicar filtros
df_filtrado = df.copy()
if len(date_range) == 2:
    df_filtrado = df_filtrado[
        (df_filtrado['Order Lines/Created on'].dt.date >= date_range[0]) & 
        (df_filtrado['Order Lines/Created on'].dt.date <= date_range[1])
    ]
if asesores_seleccionados:
    df_filtrado = df_filtrado[df_filtrado['Order Lines/Customer/Asesor (Gestor)'].isin(asesores_seleccionados)]
df_filtrado = df_filtrado[
    (df_filtrado['Order Lines/Untaxed Invoiced Amount'] >= min_venta) & 
    (df_filtrado['Order Lines/Untaxed Invoiced Amount'] <= max_venta)
]

# ==================================================
# PESTAÑAS PRINCIPALES
# ==================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen General", 
    "👥 Comportamiento Clientes", 
    "📦 Análisis Productos", 
    "🔄 Evolución Temporal",
    "🌍 Georeferenciación"
])

# ==================================================
# USO CORREGIDO EN PESTAÑA 1
# ==================================================

with tab1:
    st.title("📊 Resumen Comercial")
    
    # Calcular RFM de forma segura
    rfm_data = calculate_rfm(df_filtrado)
    
    # Manejar caso cuando el cálculo falla
    if rfm_data is None or rfm_data.empty:
        st.warning("No se pudieron calcular las métricas RFM. Mostrando datos de ejemplo.")
        rfm_data = pd.DataFrame({
            'Cliente': ['Cliente Ejemplo 1', 'Cliente Ejemplo 2'],
            'Recencia': [30, 180],
            'Frecuencia': [5, 2],
            'ValorMonetario': [5000, 2000],
            'TicketPromedio': [1000, 1000]
        })
    
    # Calcular KPIs con nombres de columnas consistentes
    clientes_unicos = len(rfm_data)
    tasa_repeticion = len(rfm_data[rfm_data['Frecuencia'] > 1]) / clientes_unicos if clientes_unicos > 0 else 0
    valor_vida_cliente = rfm_data['ValorMonetario'].mean()
    ticket_promedio = rfm_data['TicketPromedio'].mean()
    total_ventas = rfm_data['ValorMonetario'].sum()
    
    # Mostrar KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ventas Totales", f"${total_ventas:,.2f}")
    with col2:
        st.metric("Clientes Únicos", f"{clientes_unicos:,}")
    with col3:
        st.metric("Tasa Repetición", f"{tasa_repeticion:.1%}")
    with col4:
        st.metric("CLV Promedio", f"${valor_vida_cliente:,.2f}")
    
    st.markdown("---")
    
    # Gráfico de ventas mensuales
    try:
        ventas_mensuales = df_filtrado.groupby('Mes-Año')['Order Lines/Untaxed Invoiced Amount'].sum().reset_index()
        fig_ventas = px.line(
            ventas_mensuales,
            x='Mes-Año',
            y='Order Lines/Untaxed Invoiced Amount',
            title="Evolución de Ventas Mensuales",
            markers=True
        )
        st.plotly_chart(fig_ventas, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar gráfico de ventas: {str(e)}")

    # Top productos
    try:
        top_productos = df_filtrado.groupby('Order Lines/Product').agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'count'
        }).reset_index()
        top_productos.columns = ['Producto', 'Ventas', 'Unidades']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Productos por Ventas")
            fig_prod_ventas = px.bar(
                top_productos.nlargest(5, 'Ventas'),
                x='Producto',
                y='Ventas',
                color='Ventas'
            )
            st.plotly_chart(fig_prod_ventas, use_container_width=True)
        
        with col2:
            st.subheader("Top 5 Productos por Unidades")
            fig_prod_uni = px.bar(
                top_productos.nlargest(5, 'Unidades'),
                x='Producto',
                y='Unidades',
                color='Unidades'
            )
            st.plotly_chart(fig_prod_uni, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar gráficos de productos: {str(e)}")

# [Resto de pestañas (tab2, tab3, tab4, tab5) permanecen igual...]

# Pie de página
st.markdown("---")
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Dashboard desarrollado con Streamlit")

# Mostrar datos filtrados (opcional)
if st.checkbox("📋 Mostrar datos filtrados", key="show_data"):
    st.dataframe(df_filtrado, use_container_width=True)
