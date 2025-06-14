import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from io import BytesIO

# Configuración de la página (responsive para móviles)
st.set_page_config(
    layout="wide",
    page_title="Dashboard Ventas",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- CSS para móviles ---
st.markdown("""
<style>
    @media (max-width: 768px) {
        .stMetric { padding: 5px !important; }
        .stDataFrame { font-size: 12px !important; }
        .stPlotlyChart { height: 300px !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- Carga de datos desde Google Drive (Excel público) ---
@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_data():
    try:
        # ENLACE PÚBLICO (Reemplaza con tu ID de archivo)
        file_id = "1i53R94PaYc9GmEhM1zAdP0Wx0OlVJSFZ"  # Ejemplo
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Descargar archivo
        response = requests.get(download_url)
        response.raise_for_status()
        
        # Leer Excel
        df = pd.read_excel(BytesIO(response.content), sheet_name="Hoja2")
        
        # Limpieza de columnas
        df.columns = df.columns.str.strip()
        
        # Columnas requeridas
        required_cols = [
            'Order Lines/Invoice Lines/Number',
            'Order Lines/Untaxed Invoiced Amount',
            'Order Lines/Customer/Company Name',
            'Order Lines/Customer/Asesor (Gestor)',
            'Order Lines/Created on',
            'Order Lines/Product'
        ]
        
        # Verificar columnas
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Columnas faltantes: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Limpieza de datos
        df['Order Lines/Untaxed Invoiced Amount'] = (
            df['Order Lines/Untaxed Invoiced Amount']
            .astype(str)
            .str.replace('[^\d.]', '', regex=True)
            .astype(float)
        )
        
        # Procesamiento de fechas
        df['Order Lines/Created on'] = pd.to_datetime(df['Order Lines/Created on'], errors='coerce')
        df = df.dropna(subset=['Order Lines/Created on'])
        
        # Campos temporales
        df['Año'] = df['Order Lines/Created on'].dt.year
        df['Mes'] = df['Order Lines/Created on'].dt.month
        df['Mes-Año'] = df['Order Lines/Created on'].dt.strftime('%Y-%m')
        df['Día'] = df['Order Lines/Created on'].dt.day_name(locale='es')
        
        return df
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

# --- Carga de datos ---
df = load_data()

# Verificar datos cargados
if df.empty:
    st.warning("No se pudieron cargar los datos. Verifica el archivo fuente.")
    st.stop()

# Sidebar con filtros (sin filtro de productos)
st.sidebar.header("⚙️ Filtros")

# Filtro por meses
all_months = sorted(df['Mes-Año'].unique())
selected_months = st.sidebar.multiselect(
    "Seleccionar meses para comparar", 
    all_months, 
    default=all_months[-2:] if len(all_months) >= 2 else all_months
)

# Filtro por asesores
all_asesores = sorted(df['Order Lines/Customer/Asesor (Gestor)'].unique())
selected_asesores = st.sidebar.multiselect(
    "Filtrar por asesor", 
    all_asesores, 
    default=all_asesores
)

# Aplicar filtros (sin filtro de productos)
filtered_df = df.copy()
if selected_months:
    filtered_df = filtered_df[filtered_df['Mes-Año'].isin(selected_months)]
if selected_asesores:
    filtered_df = filtered_df[filtered_df['Order Lines/Customer/Asesor (Gestor)'].isin(selected_asesores)]

# Validar selección
if len(selected_months) == 0:
    st.warning("⚠️ Selecciona al menos un mes para visualizar datos")
    st.stop()

# Título principal
st.title("📊 Dashboard Comparativo de Ventas")
st.markdown("---")

# Sección de KPIs
st.header("📈 Métricas Clave")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    total_pedidos = filtered_df['Order Lines/Invoice Lines/Number'].nunique()
    st.metric("Total Pedidos", f"{total_pedidos:,}")

with kpi2:
    total_ventas = filtered_df['Order Lines/Untaxed Invoiced Amount'].sum()
    st.metric("Total Ventas", f"${total_ventas:,.2f}")

with kpi3:
    clientes_unicos = filtered_df['Order Lines/Customer/Company Name'].nunique()
    st.metric("Clientes Únicos", f"{clientes_unicos:,}")

with kpi4:
    avg_venta = total_ventas / total_pedidos if total_pedidos > 0 else 0
    st.metric("Ticket Promedio", f"${avg_venta:,.2f}")

st.markdown("---")

# --------------------------------------------------
# COMPARATIVA GENERAL ENTRE MESES
# --------------------------------------------------
if len(selected_months) >= 2:
    st.header("🔄 Comparativa General entre Meses")
    
    comparativa_meses = filtered_df.groupby('Mes-Año').agg({
        'Order Lines/Untaxed Invoiced Amount': ['sum', 'mean'],
        'Order Lines/Invoice Lines/Number': 'nunique',
        'Order Lines/Customer/Company Name': 'nunique'
    }).reset_index()
    
    comparativa_meses.columns = [
        'Mes-Año', 
        'Ventas Totales', 
        'Venta Promedio', 
        'Total Pedidos', 
        'Clientes Únicos'
    ]
    
    meses_ordenados = comparativa_meses.sort_values('Mes-Año')
    meses_ordenados['Diferencia Ventas'] = meses_ordenados['Ventas Totales'].diff()
    meses_ordenados['Variación % Ventas'] = (meses_ordenados['Ventas Totales'].pct_change() * 100).round(2)
    
    st.subheader("📊 Tabla Comparativa")
    st.dataframe(
        meses_ordenados.style.format({
            'Ventas Totales': '${:,.2f}',
            'Venta Promedio': '${:,.2f}',
            'Diferencia Ventas': '${:,.2f}',
            'Variación % Ventas': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ventas = px.bar(
            meses_ordenados,
            x='Mes-Año',
            y='Ventas Totales',
            title="Ventas Totales por Mes",
            text='Ventas Totales',
            color='Mes-Año'
        )
        fig_ventas.update_traces(
            texttemplate='$%{text:,.2f}',
            textposition='outside'
        )
        fig_ventas.update_layout(
            yaxis=dict(title='Ventas ($)'),
            showlegend=False
        )
        st.plotly_chart(fig_ventas, use_container_width=True)
    
    with col2:
        fig_variacion = px.line(
            meses_ordenados,
            x='Mes-Año',
            y='Variación % Ventas',
            title="Variación Porcentual de Ventas",
            markers=True,
            text='Variación % Ventas'
        )
        fig_variacion.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='top center',
            line=dict(color='#FFA15A', width=3)
        )
        fig_variacion.update_layout(
            yaxis=dict(title='Variación %'),
            showlegend=False
        )
        st.plotly_chart(fig_variacion, use_container_width=True)

# --------------------------------------------------
# PESTAÑAS PARA ANÁLISIS DETALLADO
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Evolución Temporal", 
    "📊 Análisis de Productos", 
    "👥 Análisis de Clientes", 
    "🌍 Mapa Geográfico"  # Tab 4 restaurado
])

with tab1:
    st.header("Tendencia Temporal de Ventas")
    
    freq = st.radio(
        "Frecuencia de análisis:",
        ["Diario", "Semanal", "Mensual"],
        horizontal=True
    )
    
    if freq == "Diario":
        ventas_temporales = filtered_df.groupby(filtered_df['Order Lines/Created on'].dt.date).agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'nunique'
        }).reset_index()
        x_col = 'Order Lines/Created on'
    elif freq == "Semanal":
        filtered_df['Semana'] = filtered_df['Order Lines/Created on'].dt.strftime('%Y-%U')
        ventas_temporales = filtered_df.groupby('Semana').agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'nunique'
        }).reset_index()
        x_col = 'Semana'
    else:  # Mensual
        ventas_temporales = filtered_df.groupby('Mes-Año').agg({
            'Order Lines/Untaxed Invoiced Amount': 'sum',
            'Order Lines/Invoice Lines/Number': 'nunique'
        }).reset_index()
        x_col = 'Mes-Año'
    
    fig_temporal = px.line(
        ventas_temporales,
        x=x_col,
        y='Order Lines/Untaxed Invoiced Amount',
        title=f"Evolución de Ventas ({freq})",
        markers=True
    )
    fig_temporal.update_layout(
        yaxis_title="Ventas ($)",
        xaxis_title=freq
    )
    st.plotly_chart(fig_temporal, use_container_width=True)

with tab2:
    st.header("Análisis de Productos")
    
    top_productos = filtered_df.groupby('Order Lines/Product').agg({
        'Order Lines/Untaxed Invoiced Amount': ['sum', 'count'],
        'Order Lines/Invoice Lines/Number': 'nunique'
    }).reset_index()
    top_productos.columns = ['Producto', 'Ventas Totales', 'Unidades Vendidas', 'Pedidos']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 por Ventas")
        top_ventas = top_productos.nlargest(10, 'Ventas Totales')
        fig_top_ventas = px.bar(
            top_ventas,
            x='Producto',
            y='Ventas Totales',
            color='Ventas Totales',
            text='Ventas Totales'
        )
        fig_top_ventas.update_traces(
            texttemplate='$%{text:,.2f}',
            textposition='outside'
        )
        st.plotly_chart(fig_top_ventas, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 por Unidades")
        top_unidades = top_productos.nlargest(10, 'Unidades Vendidas')
        fig_top_unidades = px.bar(
            top_unidades,
            x='Producto',
            y='Unidades Vendidas',
            color='Unidades Vendidas',
            text='Unidades Vendidas'
        )
        fig_top_unidades.update_traces(textposition='outside')
        st.plotly_chart(fig_top_unidades, use_container_width=True)

with tab3:
    st.header("Análisis de Clientes")
    
    top_clientes = filtered_df.groupby('Order Lines/Customer/Company Name').agg({
        'Order Lines/Untaxed Invoiced Amount': 'sum',
        'Order Lines/Invoice Lines/Number': 'nunique'
    }).reset_index()
    top_clientes.columns = ['Cliente', 'Ventas Totales', 'Pedidos']
    
    fig_clientes = px.treemap(
        top_clientes,
        path=['Cliente'],
        values='Ventas Totales',
        color='Ventas Totales',
        title="Distribución de Ventas por Cliente"
    )
    st.plotly_chart(fig_clientes, use_container_width=True)

with tab4:  # Tab 4 restaurado completamente
    st.header("Distribución Geográfica")
    
    if all(col in filtered_df.columns for col in ['Order Lines/Customer/Geo Latitude', 'Order Lines/Customer/Geo Longitude']):
        geo_data = filtered_df[
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
            
            # Mapa con tamaño según ventas
            st.map(geo_data, size='Ventas', zoom=10)
            
            # Tabla de datos geográficos
            st.dataframe(
                geo_data.sort_values('Ventas', ascending=False),
                column_config={
                    "lat": "Latitud",
                    "lon": "Longitud",
                    "Cliente": "Cliente",
                    "Ventas": st.column_config.NumberColumn(
                        "Ventas ($)",
                        format="$%.2f"
                    )
                },
                hide_index=True
            )
        else:
            st.warning("No hay datos geográficos disponibles para los filtros seleccionados.")
    else:
        st.warning("No se encontraron coordenadas geográficas en los datos.")

# Mostrar datos brutos
st.markdown("---")
if st.checkbox("📋 Mostrar datos brutos (muestra limitada)"):
    st.dataframe(
        filtered_df.head(1000),
        use_container_width=True,
        hide_index=True
    )

# Notas finales
st.caption("Dashboard comparativo para ver ventas cruzadas con rutas de SDQ" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))