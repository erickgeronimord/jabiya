import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
def configurar_locale():
    try:
        locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except locale.Error:
            os.environ['LC_ALL'] = 'C.UTF-8'
            os.environ['LANG'] = 'C.UTF-8'

configurar_locale()

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Validacion y Cruce con clientes jabiya SDQ",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .header { font-size: 24px !important; font-weight: bold !important; }
    .metric-box { 
        padding: 15px; 
        border-radius: 10px; 
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .stAlert { border-radius: 10px; }
    .stPlotlyChart { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# FUNCIÓN DE CARGA DE DATOS MEJORADA
# ==================================================

@st.cache_data(ttl=3600)
def cargar_datos():
    """Carga y procesa los datos con manejo robusto de errores"""
    try:
        # Intento de carga desde Google Drive
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
        st.error(f"Error al cargar datos principales: {str(e)}")
        
        # Datos de ejemplo como fallback
        fecha_inicio = datetime(2023, 1, 1)
        rango_fechas = [fecha_inicio + timedelta(days=i) for i in range(90)]
        df = pd.DataFrame({
            'Order Lines/Created on': rango_fechas,
            'Order Lines/Customer/Company Name': np.random.choice(['Cliente A', 'Cliente B', 'Cliente C'], 90),
            'Order Lines/Untaxed Invoiced Amount': np.random.uniform(100, 5000, 90),
            'Order Lines/Product': np.random.choice(['Producto 1', 'Producto 2', 'Producto 3'], 90),
            'Order Lines/Invoice Lines/Number': np.random.randint(1000, 9999, 90),
            'Order Lines/Customer/Asesor (Gestor)': np.random.choice(['Asesor 1', 'Asesor 2', 'Asesor 3'], 90)
        })
        
        st.warning("Se están utilizando datos de ejemplo. La funcionalidad será limitada.")
        return df

# ==================================================
# FUNCIÓN PARA CÁLCULO RFM MEJORADA
# ==================================================

def calcular_rfm(df_input):
    """Calcula métricas RFM con validación robusta"""
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

        # Calcular RFM con nombres explícitos
        now = pd.Timestamp.now(tz='UTC')
        
        rfm = df_input.groupby('Order Lines/Customer/Company Name').agg(
            Recencia=('Order Lines/Created on', lambda x: (now - x.max()).days),
            Frecuencia=('Order Lines/Invoice Lines/Number', 'nunique'),
            ValorMonetario=('Order Lines/Untaxed Invoiced Amount', 'sum'),
            TicketPromedio=('Order Lines/Untaxed Invoiced Amount', 'mean')
        ).reset_index().rename(columns={'Order Lines/Customer/Company Name': 'Cliente'})
        
        # Manejar valores nulos
        rfm = rfm.fillna({
            'Recencia': 365,
            'ValorMonetario': 0,
            'TicketPromedio': 0
        })
        
        # Segmentación RFM
        rfm['R_Score'] = pd.qcut(rfm['Recencia'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frecuencia'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['ValorMonetario'], 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm
    
    except Exception as e:
        st.error(f"Error inesperado en cálculo RFM: {str(e)}")
        return None

# ==================================================
# CARGA DE DATOS Y FILTROS
# ==================================================

# Cargar datos
df = cargar_datos()

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
# PESTAÑAS PRINCIPALES (TODAS INCLUIDAS)
# ==================================================

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
    
    # Subtítulo y descripción
    st.subheader("Análisis de métricas clave y tendencias")
    st.markdown("""
    Este tablero muestra un panorama completo del desempeño comercial, incluyendo:
    - Total de ventas y clientes únicos
    - Evolución mensual de ingresos
    - Productos más vendidos por volumen y valor
    *Los datos nos ayudarán a identificar patrones generales y puntos destacados del período seleccionado.*
    """)
    
    # Calcular RFM de forma segura
    rfm = calcular_rfm(df_filtrado)
    
    if rfm is None or rfm.empty:
        st.warning("No se pudieron calcular métricas RFM. Mostrando datos de ejemplo.")
        rfm = pd.DataFrame({
            'Cliente': ['Cliente Ejemplo 1', 'Cliente Ejemplo 2'],
            'Recencia': [30, 180],
            'Frecuencia': [5, 2],
            'ValorMonetario': [5000, 2000],
            'TicketPromedio': [1000, 1000],
            'R_Score': [5, 3],
            'F_Score': [5, 2],
            'M_Score': [5, 3],
            'RFM_Score': ['555', '323']
        })
    
    # Calcular KPIs
    clientes_unicos = len(rfm)
    tasa_repeticion = len(rfm[rfm['Frecuencia'] > 1]) / clientes_unicos if clientes_unicos > 0 else 0
    valor_vida_cliente = rfm['ValorMonetario'].mean()
    ticket_promedio = rfm['TicketPromedio'].mean()
    total_ventas = rfm['ValorMonetario'].sum()
    
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

# -----------------------------------------
# PESTAÑA 2: COMPORTAMIENTO CLIENTES
# -----------------------------------------
with tab2:
    st.title("👥 Análisis de Comportamiento de Clientes")
    
    # Subtítulo y descripción
    st.subheader("Segmentación RFM y análisis de retención")
    st.markdown("""
    Aquí analizamos a los clientes mediante:
    - Modelo RFM: Clasificación por Recencia (última compra), Frecuencia (visitas) y Valor Monetario (gasto total).
    - Análisis de Cohortes: Mide cómo se mantienen los clientes en el tiempo desde su primera compra.
    - Clientes Top: Identifica quiénes generan mayor valor para tu negocio
       """)
    
    # Mostrar segmentación RFM
    if rfm is not None and not rfm.empty:
        st.header("🔍 Segmentación RFM")
        try:
            fig_rfm = px.scatter(
                rfm,
                x='Frecuencia',
                y='Recencia',
                size='ValorMonetario',
                color='RFM_Score',
                hover_name='Cliente',
                log_x=True,
                title='Segmentación RFM de Clientes',
                height=600
            )
            st.plotly_chart(fig_rfm, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar gráfico RFM: {str(e)}")
    
    # Análisis de Cohortes
    st.header("📅 Retención por Cohortes")
    try:
        df_filtrado['Cohorte'] = df_filtrado.groupby('Order Lines/Customer/Company Name')['Order Lines/Created on'].transform('min').dt.strftime('%Y-%m')
        cohortes = df_filtrado.groupby(['Cohorte', 'Mes-Año']).agg({
            'Order Lines/Customer/Company Name': 'nunique',
            'Order Lines/Untaxed Invoiced Amount': 'sum'
        }).reset_index()
        
        cohortes['MesesDesdeCohorte'] = (pd.to_datetime(cohortes['Mes-Año']) - pd.to_datetime(cohortes['Cohorte'])).dt.days // 30
        
        retention_pivot = cohortes.pivot_table(
            index='Cohorte',
            columns='MesesDesdeCohorte',
            values='Order Lines/Customer/Company Name',
            aggfunc='sum',
            fill_value=0
        )
        
        fig_cohort = px.imshow(
            retention_pivot,
            labels=dict(x="Meses desde Cohorte", y="Cohorte", color="Clientes"),
            title='Retención de Clientes por Cohorte',
            color_continuous_scale='Blues',
            aspect='auto'
        )
        st.plotly_chart(fig_cohort, use_container_width=True)
    except Exception as e:
        st.error(f"Error en análisis de cohortes: {str(e)}")
    
    # Top Clientes
    st.header("🏆 Clientes más Valiosos")
    try:
        if rfm is not None and not rfm.empty:
            fig_clientes = px.bar(
                rfm.sort_values('ValorMonetario', ascending=False).head(10),
                x='Cliente',
                y='ValorMonetario',
                color='Frecuencia',
                title='Top 10 Clientes por Valor Total',
                hover_data=['Recencia']
            )
            st.plotly_chart(fig_clientes, use_container_width=True)
        else:
            st.warning("No hay datos de clientes para mostrar")
    except Exception as e:
        st.error(f"Error al generar gráfico de clientes: {str(e)}")

# -----------------------------------------
# PESTAÑA 3: ANÁLISIS DE PRODUCTOS
# -----------------------------------------
with tab3:
    st.title("📦 Análisis de Productos")
        
    # Subtítulo y descripción
    st.subheader("Desempeño y relaciones entre productos")
    st.markdown("""
    En esta sección encontraremos:
    - Productos estrella: Los más vendidos en unidades y valor económico.
    - Asociaciones: Qué productos se compran juntos frecuentemente (para paquetes y promociones).
    - Distribución: Participación porcentual de cada producto en las ventas totales.
       """)
    
    # Asociación de Productos
    st.header("🛒 Productos Comprados Juntos")
    try:
        transacciones = df_filtrado.groupby(['Order Lines/Invoice Lines/Number', 'Order Lines/Product'])['Order Lines/Product'].count().unstack().fillna(0)
        transacciones = transacciones.applymap(lambda x: 1 if x > 0 else 0)
        
        reglas = apriori(transacciones, min_support=0.05, use_colnames=True, max_len=2)
        reglas = association_rules(reglas, metric="lift", min_threshold=1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Reglas de Asociación")
            st.dataframe(reglas.sort_values('lift', ascending=False).head(5))
        
        with col2:
            st.subheader("Red de Relaciones")
            fig_network = plt.figure(figsize=(10,8))
            G = nx.Graph()
            
            for _, row in reglas.sort_values('lift', ascending=False).head(8).iterrows():
                antecedentes = ', '.join(list(row['antecedents']))
                consecuentes = ', '.join(list(row['consequents']))
                G.add_edge(antecedentes, consecuentes, weight=row['lift'])
            
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", 
                   font_size=10, width=[d['weight']*0.5 for (u,v,d) in G.edges(data=True)])
            st.pyplot(fig_network)
    except Exception as e:
        st.warning(f"No se pudo generar el análisis de asociación: {str(e)}")
    
    # Distribución de Ventas por Producto
    st.header("📊 Distribución de Ventas")
    try:
        fig_prod = px.treemap(
            df_filtrado.groupby('Order Lines/Product')['Order Lines/Untaxed Invoiced Amount'].sum().reset_index(),
            path=['Order Lines/Product'],
            values='Order Lines/Untaxed Invoiced Amount',
            title='Participación de cada Producto en Ventas Totales'
        )
        st.plotly_chart(fig_prod, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar gráfico de productos: {str(e)}")

# -----------------------------------------
# PESTAÑA 4: EVOLUCIÓN TEMPORAL
# -----------------------------------------
with tab4:
    st.title("🔄 Evolución Temporal")
        
    # Subtítulo y descripción
    st.subheader("Análisis de métricas clave y tendencias")
    st.markdown("""
    Este tablero muestra un panorama completo del desempeño comercial, incluyendo:
    - Total de ventas y clientes únicos
    - Evolución mensual de ingresos
    - Productos más vendidos por volumen y valor
    """)
    
    # Selector de frecuencia
    freq = st.radio(
        "Frecuencia de análisis:",
        ["Diario", "Semanal", "Mensual"],
        horizontal=True
    )
    
    try:
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
    except Exception as e:
        st.error(f"Error al generar análisis temporal: {str(e)}")
    
    # Patrones horarios
    st.header("⏰ Patrones de Compra por Hora")
    try:
        patrones_hora = df_filtrado.groupby(['DíaSemana', 'Hora']).agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum'
        }).reset_index()
        
        fig_hora = px.density_heatmap(
            patrones_hora,
            x='Hora',
            y='DíaSemana',
            z='Order Lines/Untaxed Invoiced Amount',
            title='Intensidad de Compras por Día y Hora',
            color_continuous_scale='Viridis',
            category_orders={"DíaSemana": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]}
        )
        st.plotly_chart(fig_hora, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar análisis horario: {str(e)}")

# -----------------------------------------
# PESTAÑA 5: GEOREFERENCIACIÓN (VERSIÓN FUNCIONAL)
# -----------------------------------------
with tab5:
    st.title("🌍 Análisis Geográfico")
            
    # Subtítulo y descripción
    st.subheader("SDistribución geográfica de clientes y ventas")
    st.markdown("""
   Visualizamo en mapas interactivos::
    - Concentración de clientes por zonas geográficas.
    - "Ventas por ubicación" (tamaño de puntos según monto).
    - Rutas de distribución potenciales basadas en clusters.
    """)
    
    # Verificar si existen columnas de geolocalización
    geo_cols = ['Order Lines/Customer/Geo Latitude', 'Order Lines/Customer/Geo Longitude']
    
    if all(col in df_filtrado.columns for col in geo_cols):
        # Mapa de calor geográfico
        st.header("🗺️ Mapa de Calor de Clientes")
        
        try:
            # Preparar datos geográficos
            geo_data = df_filtrado[
                ['Order Lines/Customer/Company Name', 
                 geo_cols[0], 
                 geo_cols[1],
                 'Order Lines/Untaxed Invoiced Amount']
            ].dropna()
            
            if not geo_data.empty:
                geo_data = geo_data.rename(columns={
                    geo_cols[0]: 'lat',
                    geo_cols[1]: 'lon',
                    'Order Lines/Customer/Company Name': 'Cliente',
                    'Order Lines/Untaxed Invoiced Amount': 'Ventas'
                })
                
                # Crear figura con un estilo de mapa que no requiere token
                fig = px.scatter_mapbox(
                    geo_data,
                    lat='lat',
                    lon='lon',
                    size='Ventas',
                    color='Ventas',
                    color_continuous_scale="reds",
                    hover_name='Cliente',
                    hover_data=['Ventas'],
                    zoom=11,
                    height=600,
                    title='Distribución Geográfica de Clientes'
                )
                
                # Usar un estilo de mapa de acceso abierto
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Opción alternativa con densidad
                st.header("🔍 Mapa de Densidad")
                fig_density = px.density_mapbox(
                    geo_data,
                    lat='lat',
                    lon='lon',
                    z='Ventas',
                    radius=20,
                    center=dict(lat=geo_data['lat'].mean(), lon=geo_data['lon'].mean()),
                    zoom=11,
                    mapbox_style="open-street-map",
                    height=600,
                    title='Concentración Geográfica de Ventas'
                )
                st.plotly_chart(fig_density, use_container_width=True)
                
            else:
                st.warning("No hay datos geográficos válidos para los filtros seleccionados")
                
        except Exception as e:
            st.error(f"Error al generar mapa geográfico: {str(e)}")
            st.info("""
            Consejos para solucionar problemas:
            1. Verifica que las columnas de latitud y longitud contengan valores válidos
            2. Asegúrate que los valores de latitud estén entre -90 y 90
            3. Asegúrate que los valores de longitud estén entre -180 y 180
            """)
            
    else:
        st.warning("""
        No se encontraron datos de coordenadas geográficas en el dataset. 
        Se requieren columnas llamadas:
        - 'Order Lines/Customer/Geo Latitude' 
        - 'Order Lines/Customer/Geo Longitude'
        """)
        
        # Mostrar columnas disponibles para diagnóstico
        st.write("Columnas disponibles en los datos:", df_filtrado.columns.tolist())

# ==================================================
# PIE DE PÁGINA Y OPCIONES ADICIONALES
# ==================================================

st.markdown("---")
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Dashboard para cruzar los clientes de jabiya con el centro de SDQ")

# Opción para mostrar datos filtrados
if st.checkbox("📋 Mostrar datos filtrados", key="show_data"):
    st.dataframe(df_filtrado, use_container_width=True)

# Opción para descargar datos
if st.button("💾 Descargar datos filtrados"):
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar como CSV",
        data=csv,
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )
