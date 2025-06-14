import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from io import BytesIO
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
import locale
import os

# --------------------------------------------------
# CONFIGURACIÓN INICIAL DE LOCALE (SOLUCIÓN AL ERROR)
# --------------------------------------------------
def configure_locale():
    """Configura el locale para evitar errores en Streamlit Cloud"""
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            # Configuración de respaldo
            os.environ['LC_ALL'] = 'C.UTF-8'
            os.environ['LANG'] = 'C.UTF-8'
            os.environ['LANGUAGE'] = 'C.UTF-8'
            try:
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
            except:
                st.warning("No se pudo configurar el locale correctamente. Algunas funciones pueden no trabajar óptimamente.")

# Ejecutar la configuración al inicio
configure_locale()

# --------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Dashboard Comercial Integral",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- CSS para mejor visualización ---
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

# --------------------------------------------------
# FUNCIÓN DE CARGA DE DATOS MEJORADA
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """Carga y procesa los datos con manejo robusto de locale"""
    try:
        # Configuración adicional de locale para esta función
        configure_locale()
        
        # Descarga del archivo
        file_id = "1i53R94PaYc9GmEhM1zAdP0Wx0OlVJSFZ"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        response.raise_for_status()
        
        # Carga del Excel
        df = pd.read_excel(BytesIO(response.content), sheet_name="Hoja2")
        
        # Limpieza y transformación
        df.columns = df.columns.str.strip()
        
        # Conversión de montos - método robusto
        df['Order Lines/Untaxed Invoiced Amount'] = (
            df['Order Lines/Untaxed Invoiced Amount']
            .astype(str)
            .str.replace('[^\d.]', '', regex=True)
            .astype(float)
        )
        
        # Manejo de fechas independiente de locale
        df['Order Lines/Created on'] = pd.to_datetime(
            df['Order Lines/Created on'], 
            errors='coerce',
            utc=True  # Usar UTC para mayor compatibilidad
        )
        df = df.dropna(subset=['Order Lines/Created on'])
        
        # Mapeo de días de la semana sin dependencia de locale
        dias_semana = {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo"
        }
        
        # Campos adicionales
        df['Año'] = df['Order Lines/Created on'].dt.year
        df['Mes'] = df['Order Lines/Created on'].dt.month
        df['Mes-Año'] = df['Order Lines/Created on'].dt.strftime('%Y-%m')
        df['DíaSemana'] = df['Order Lines/Created on'].dt.weekday.map(dias_semana)
        df['Hora'] = df['Order Lines/Created on'].dt.hour
        df['Trimestre'] = df['Order Lines/Created on'].dt.quarter
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------
# CARGA Y VERIFICACIÓN DE DATOS
# --------------------------------------------------
df = load_data()

if df.empty:
    st.warning("""
        No se pudieron cargar los datos. Verifica:
        1. Que el archivo esté disponible en Google Drive
        2. Que tengas permisos para acceder al archivo
        3. Que la estructura del archivo sea correcta
    """)
    st.stop()

# --- Sidebar con Filtros ---
with st.sidebar:
    st.header("⚙️ Filtros")
    
    # Filtro temporal
    min_date = df['Order Lines/Created on'].min().date()
    max_date = df['Order Lines/Created on'].max().date()
    date_range = st.date_input("Rango de fechas", [min_date, max_date])
    
    # Filtro por asesores
    asesores = sorted(df['Order Lines/Customer/Asesor (Gestor)'].dropna().unique())
    asesores_seleccionados = st.multiselect("Asesores", asesores, default=asesores)
    
    # Filtro por valor de compra
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

# --- Pestañas Principales ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen General", 
    "👥 Comportamiento Clientes", 
    "📦 Análisis Productos", 
    "🔄 Evolución Temporal",
    "🌍 Georeferenciación"
])

# -----------------------------------------
# PESTAÑA 1: RESUMEN GENERAL
# -----------------------------------------
with tab1:
    st.title("📊 Resumen Comercial")
    
    # Calculate RFM safely
    rfm = calculate_rfm(df_filtrado)
    
    if rfm is None:
        st.warning("No se pudieron calcular las métricas RFM. Mostrando datos de ejemplo.")
        # Create sample RFM data
        rfm = pd.DataFrame({
            'Company Name': ['Cliente Ejemplo'],
            'Recencia': [30],
            'Frecuencia': [1],
            'ValorMonetario': [1000],
            'TicketPromedio': [1000]
        })
    
    # Calcular métricas
    clientes_unicos = df_filtrado['Order Lines/Customer/Company Name'].nunique()
    tasa_repeticion = len(rfm[rfm['Frecuencia'] > 1]) / clientes_unicos if clientes_unicos > 0 else 0
    valor_vida_cliente = rfm['ValorMonetario'].mean()
    ticket_promedio = df_filtrado['Order Lines/Untaxed Invoiced Amount'].mean()
    total_ventas = df_filtrado['Order Lines/Untaxed Invoiced Amount'].sum()
    
    # KPIs
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
    
    # Gráfico de ventas por mes
    ventas_mensuales = df_filtrado.groupby('Mes-Año')['Order Lines/Untaxed Invoiced Amount'].sum().reset_index()
    fig_ventas = px.line(
        ventas_mensuales,
        x='Mes-Año',
        y='Order Lines/Untaxed Invoiced Amount',
        title="Evolución de Ventas Mensuales",
        markers=True
    )
    st.plotly_chart(fig_ventas, use_container_width=True)
    
    # Top productos
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

# -----------------------------------------
# PESTAÑA 2: COMPORTAMIENTO CLIENTES
# -----------------------------------------
with tab2:
    st.title("👥 Análisis de Comportamiento de Clientes")
    
    # Segmentación RFM
    st.header("🔍 Segmentación RFM")
    rfm['R_Score'] = pd.qcut(rfm['Recencia'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frecuencia'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['ValorMonetario'], 5, labels=[1,2,3,4,5])
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    fig_rfm = px.scatter(
        rfm,
        x='Frecuencia',
        y='Recencia',
        size='ValorMonetario',
        color='RFM_Score',
        hover_name='Cliente',
        log_x=True,
        title='Segmentación RFM de Clientes'
    )
    st.plotly_chart(fig_rfm, use_container_width=True)
    
with tab2:
    st.header("📅 Retención por Cohortes")
    
    # Convertir a string en lugar de Period
    df_filtrado['Cohorte'] = df_filtrado.groupby('Order Lines/Customer/Company Name')['Order Lines/Created on'].transform('min').dt.strftime('%Y-%m')
    cohortes = df_filtrado.groupby(['Cohorte', 'Mes-Año']).agg({
        'Order Lines/Customer/Company Name': 'nunique',
        'Order Lines/Untaxed Invoiced Amount': 'sum'
    }).reset_index()
    
    # Calcular meses desde cohorte como entero
    cohortes['MesesDesdeCohorte'] = (pd.to_datetime(cohortes['Mes-Año']) - pd.to_datetime(cohortes['Cohorte'])).dt.days // 30
    
    # Crear matriz de retención
    retention_pivot = cohortes.pivot_table(
        index='Cohorte',
        columns='MesesDesdeCohorte',
        values='Order Lines/Customer/Company Name',
        aggfunc='sum'
    ).fillna(0)
    
    # Convertir índices a string
    retention_pivot.index = retention_pivot.index.astype(str)
    retention_pivot.columns = retention_pivot.columns.astype(str)
    
    fig_cohort = px.imshow(
        retention_pivot,
        labels=dict(x="Meses desde Cohorte", y="Cohorte", color="Clientes"),
        title='Retención de Clientes por Cohorte',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_cohort, use_container_width=True)
    
    # Top Clientes
    st.header("🏆 Clientes más Valiosos")
    fig_clientes = px.bar(
        rfm.sort_values('ValorMonetario', ascending=False).head(10),
        x='Cliente',
        y='ValorMonetario',
        color='Frecuencia',
        title='Top 10 Clientes por Valor Total'
    )
    st.plotly_chart(fig_clientes, use_container_width=True)

# -----------------------------------------
# PESTAÑA 3: ANÁLISIS DE PRODUCTOS
# -----------------------------------------
with tab3:
    st.title("📦 Análisis de Productos")
    
    # Asociación de Productos
    st.header("🛒 Productos Comprados Juntos")
    try:
        transacciones = df_filtrado.groupby(['Order Lines/Invoice Lines/Number', 'Order Lines/Product'])['Order Lines/Product'].count().unstack().fillna(0)
        reglas = apriori(transacciones>0, min_support=0.05, use_colnames=True)
        reglas = association_rules(reglas, metric="lift", min_threshold=1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Reglas de Asociación")
            st.dataframe(reglas.sort_values('lift', ascending=False).head(5))
        
        with col2:
            st.subheader("Red de Relaciones")
            fig_network = plt.figure(figsize=(8,6))
            G = nx.Graph()
            for _, row in reglas.sort_values('lift', ascending=False).head(8).iterrows():
                G.add_edge(', '.join(list(row['antecedents'])), ', '.join(list(row['consequents'])), weight=row['lift'])
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=9)
            st.pyplot(fig_network)
    except Exception as e:
        st.warning(f"No se pudo generar el análisis de asociación: {str(e)}")
    
    # Distribución de Ventas por Producto
    st.header("📊 Distribución de Ventas")
    fig_prod = px.treemap(
        df_filtrado.groupby('Order Lines/Product')['Order Lines/Untaxed Invoiced Amount'].sum().reset_index(),
        path=['Order Lines/Product'],
        values='Order Lines/Untaxed Invoiced Amount',
        title='Participación de cada Producto en Ventas Totales'
    )
    st.plotly_chart(fig_prod, use_container_width=True)

# -----------------------------------------
# PESTAÑA 4: EVOLUCIÓN TEMPORAL
# -----------------------------------------
with tab4:
    st.title("🔄 Evolución Temporal")
    
    # Selector de frecuencia
    freq = st.radio(
        "Frecuencia de análisis:",
        ["Diario", "Semanal", "Mensual"],
        horizontal=True
    )
    
    if freq == "Diario":
        datos_temporales = df_filtrado.groupby(df_filtrado['Order Lines/Created on'].dt.date).agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'nunique'
        }).reset_index()
        x_col = 'Order Lines/Created on'
    elif freq == "Semanal":
        df_filtrado['Semana'] = df_filtrado['Order Lines/Created on'].dt.strftime('%Y-%U')
        datos_temporales = df_filtrado.groupby('Semana').agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'nunique'
        }).reset_index()
        x_col = 'Semana'
    else:  # Mensual
        datos_temporales = df_filtrado.groupby('Mes-Año').agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'nunique'
        }).reset_index()
        x_col = 'Mes-Año'
    
    # Gráfico temporal
    fig_temp = px.line(
        datos_temporales,
        x=x_col,
        y='Order Lines/Untaxed Invoiced Amount',
        title=f"Evolución de Ventas ({freq})",
        markers=True
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Patrones horarios
    st.header("⏰ Patrones de Compra por Hora")
    patrones_hora = df_filtrado.groupby(['DíaSemana', 'Hora']).agg({
        'Order Lines/Untaxed Invoiced Amount': 'sum'
    }).reset_index()
    
    fig_hora = px.density_heatmap(
        patrones_hora,
        x='Hora',
        y='DíaSemana',
        z='Order Lines/Untaxed Invoiced Amount',
        title='Intensidad de Compras por Día y Hora',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_hora, use_container_width=True)

# -----------------------------------------
# PESTAÑA 5: GEOREFERENCIACIÓN
# -----------------------------------------
with tab5:  # Pestaña de Georeferenciación
    st.title("🌍 Análisis Geográfico")
    
    if all(col in df_filtrado.columns for col in ['Order Lines/Customer/Geo Latitude', 'Order Lines/Customer/Geo Longitude']):
        # Mapa de puntos interactivo con zoom
        st.header("Mapa Interactivo de Clientes")
        
        geo_data = df_filtrado[
            ['Order Lines/Customer/Company Name', 
             'Order Lines/Customer/Geo Latitude', 
             'Order Lines/Customer/Geo Longitude',
             'Order Lines/Untaxed Invoiced Amount']
        ].dropna()
        
        if not geo_data.empty:
            geo_data = geo_data.rename(columns={
                'Order Lines/Customer/Geo Latitude': 'lat',
                'Order Lines/Customer/Geo Longitude': 'lon',
                'Order Lines/Customer/Company Name': 'Cliente',
                'Order Lines/Untaxed Invoiced Amount': 'Ventas'
            })
            
            # Configuración del mapa con zoom
            fig = px.scatter_mapbox(
                geo_data,
                lat="lat",
                lon="lon",
                size="Ventas",
                color="Ventas",
                color_continuous_scale="reds",  # Escala de rojos
                hover_name="Cliente",
                hover_data=["Ventas"],
                zoom=11,  # Nivel de zoom inicial
                height=600,
                title="Distribución Geográfica de Clientes"
            )
            
            # Personalización adicional del mapa
            fig.update_layout(
                mapbox_style="open-street-map",  # Estilo de mapa que permite zoom
                mapbox=dict(
                    center=dict(lat=geo_data['lat'].mean(), lon=geo_data['lon'].mean()),
                    zoom=11,
                    style='open-street-map'  # Estilo alternativo: "stamen-terrain", "carto-positron"
                ),
                margin={"r":0,"t":40,"l":0,"b":0}
            )
            
            # Mostrar el mapa con controles de zoom
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Opcional: Añadir controles manuales de zoom
            st.markdown("""
                **Controles del Mapa:**
                - Zoom: Desplázate con la rueda del mouse o gesto de pellizco en móviles
                - Movimiento: Arrastra el mapa para navegar
                - Click: Ver detalles del cliente
            """)
            
        else:
            st.warning("No hay datos geográficos disponibles para los filtros seleccionados.")
    else:
        st.warning("No se encontraron coordenadas geográficas en los datos.")

# --- Pie de página ---
st.markdown("---")
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Dashboard desarrollado con Streamlit")

# Mostrar datos filtrados
if st.checkbox("📋 Mostrar datos filtrados", key="show_data"):
    st.dataframe(df_filtrado, use_container_width=True)
