import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(page_title="EDA Dashboard", layout="wide")

# ============================================================
# TÍTULO Y DESCRIPCIÓN
# ============================================================
st.title('📊 Dashboard de Análisis Exploratorio de Datos (EDA)')
st.markdown('Esta aplicación permite realizar un análisis exploratorio de datos cargando un archivo CSV.')

# ============================================================
# CARGA DE DATOS
# ============================================================
st.sidebar.header('📁 Cargar Datos')
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=['csv'])

if uploaded_file is not None:
    # Cargar datos
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.success(f'✅ Archivo cargado: {uploaded_file.name}')
    st.sidebar.metric("Número de filas", df.shape[0])
    st.sidebar.metric("Número de columnas", df.shape[1])
    
    # Mostrar vista previa de los datos
    st.subheader('📋 Vista Previa de los Datos')
    st.dataframe(df.head(10), use_container_width=True)
    
    # ============================================================
    # SEPARAR VARIABLES POR TIPO
    # ============================================================
    # Variables numéricas (cuantitativas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Variables categóricas (cualitativas)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # ============================================================
    # TABS PARA LOS 3 BLOQUES PRINCIPALES
    # ============================================================
    tab1, tab2, tab3 = st.tabs(['📝 Variables Cualitativas', '📈 Variables Cuantitativas', '📊 Gráficos Avanzados'])
    
    # ============================================================
    # BLOQUE 1: VARIABLES CUALITATIVAS
    # ============================================================
    with tab1:
        st.header('Análisis de Variables Cualitativas')
        
        if len(categorical_cols) > 0:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_cat_var = st.selectbox('Selecciona una variable:', categorical_cols)
            
            if selected_cat_var:
                st.subheader(f'Análisis de: {selected_cat_var}')
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write('**Tabla de Frecuencias**')
                    freq_table = df[selected_cat_var].value_counts().reset_index()
                    freq_table.columns = [selected_cat_var, 'Frecuencia']
                    freq_table['Porcentaje'] = (freq_table['Frecuencia'] / freq_table['Frecuencia'].sum() * 100).round(2)
                    st.dataframe(freq_table, use_container_width=True)
                
                with col_b:
                    st.write('**Estadísticas**')
                    st.metric('Valores únicos', df[selected_cat_var].nunique())
                    st.metric('Valor más frecuente', df[selected_cat_var].mode()[0] if len(df[selected_cat_var].mode()) > 0 else 'N/A')
                    st.metric('Valores nulos', df[selected_cat_var].isnull().sum())
                
                # Gráfico de barras
                st.write('**Gráfico de Barras**')
                fig = px.bar(freq_table.head(20), x=selected_cat_var, y='Frecuencia', 
                            title=f'Distribución de {selected_cat_var}',
                            color='Frecuencia', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No se encontraron variables cualitativas en el dataset.')
    
    # ============================================================
    # BLOQUE 2: VARIABLES CUANTITATIVAS
    # ============================================================
    with tab2:
        st.header('Análisis de Variables Cuantitativas')
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_num_var = st.selectbox('Selecciona una variable:', numeric_cols)
            
            if selected_num_var:
                st.subheader(f'Análisis de: {selected_num_var}')
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write('**Estadísticas Descriptivas**')
                    stats = df[selected_num_var].describe()
                    st.dataframe(stats, use_container_width=True)
                
                with col_b:
                    st.write('**Información Adicional**')
                    st.metric('Media', f"{df[selected_num_var].mean():.2f}")
                    st.metric('Mediana', f"{df[selected_num_var].median():.2f}")
                    st.metric('Desviación estándar', f"{df[selected_num_var].std():.2f}")
                    st.metric('Valores nulos', df[selected_num_var].isnull().sum())
                
                # Histograma y Boxplot
                col_c, col_d = st.columns(2)
                
                with col_c:
                    st.write('**Histograma**')
                    fig_hist = px.histogram(df, x=selected_num_var, 
                                          title=f'Distribución de {selected_num_var}',
                                          nbins=30, color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_d:
                    st.write('**Boxplot**')
                    fig_box = px.box(df, y=selected_num_var, 
                                    title=f'Boxplot de {selected_num_var}',
                                    color_discrete_sequence=['#EF553B'])
                    st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info('No se encontraron variables cuantitativas en el dataset.')
    
    # ============================================================
    # BLOQUE 3: GRÁFICOS AVANZADOS
    # ============================================================
    with tab3:
        st.header('Gráficos Avanzados')
        
        # Selector de tipo de gráfico
        chart_type = st.selectbox('Selecciona el tipo de gráfico:', 
                                  ['Scatter Plot', 'Matriz de Correlación', 'Gráfico de Pares'])
        
        if chart_type == 'Scatter Plot':
            st.subheader('Gráfico de Dispersión')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_var = st.selectbox('Variable X:', numeric_cols if len(numeric_cols) > 0 else df.columns)
            
            with col2:
                y_var = st.selectbox('Variable Y:', numeric_cols if len(numeric_cols) > 1 else df.columns)
            
            with col3:
                color_var = st.selectbox('Color (opcional):', ['None'] + categorical_cols)
            
            if x_var and y_var:
                color_param = None if color_var == 'None' else color_var
                fig_scatter = px.scatter(df, x=x_var, y=y_var, color=color_param,
                                        title=f'{y_var} vs {x_var}',
                                        hover_data=df.columns)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        elif chart_type == 'Matriz de Correlación':
            st.subheader('Matriz de Correlación')
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto='.2f',
                                    color_continuous_scale='RdBu_r',
                                    aspect='auto',
                                    title='Matriz de Correlación')
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning('Se necesitan al menos 2 variables numéricas para calcular la correlación.')
        
        elif chart_type == 'Gráfico de Pares':
            st.subheader('Gráfico de Pares (Pairplot)')
            
            if len(numeric_cols) > 1:
                # Limitar a máximo 5 variables para mejor visualización
                selected_cols = st.multiselect('Selecciona variables (máx. 5):', 
                                              numeric_cols, 
                                              default=numeric_cols[:min(3, len(numeric_cols))])
                
                if len(selected_cols) > 1 and len(selected_cols) <= 5:
                    fig_pair = px.scatter_matrix(df[selected_cols],
                                                title='Matriz de Gráficos de Pares')
                    fig_pair.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig_pair, use_container_width=True)
                elif len(selected_cols) > 5:
                    st.warning('Por favor selecciona máximo 5 variables.')
                else:
                    st.info('Selecciona al menos 2 variables.')
            else:
                st.warning('Se necesitan al menos 2 variables numéricas.')

else:
    st.info('👆 Por favor, carga un archivo CSV desde la barra lateral para comenzar el análisis.')
    
    # Mostrar ejemplo de datos esperados
    st.subheader('📝 Formato de Datos Esperado')
    st.markdown("""
    El archivo CSV debe contener:
    - Una fila de encabezados con los nombres de las columnas
    - Datos en formato tabular
    - Variables numéricas y/o categóricas
    
    **Ejemplo:**
    ```
    nombre,edad,ciudad,salario
    Juan,25,Madrid,30000
    María,30,Barcelona,35000
    ```
    """)