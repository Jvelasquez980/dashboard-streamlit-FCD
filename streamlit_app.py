import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
from datetime import datetime

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
    
    # Control de cantidad de muestras
    st.sidebar.subheader('⚙️ Configuración de Vista')
    num_samples = st.sidebar.slider('Número de filas a mostrar:', 
                                     min_value=5, 
                                     max_value=min(100, df.shape[0]), 
                                     value=10, 
                                     step=5)
    
    # Selector de columnas
    st.sidebar.subheader('🔧 Selección de Columnas')
    show_all_cols = st.sidebar.checkbox('Mostrar todas las columnas', value=True)
    
    if not show_all_cols:
        selected_columns = st.sidebar.multiselect(
            'Selecciona columnas para visualizar:',
            options=df.columns.tolist(),
            default=df.columns.tolist()[:min(5, len(df.columns))]
        )
    else:
        selected_columns = df.columns.tolist()
    
    # Filtros de datos
    st.sidebar.subheader('🔍 Filtros de Datos')
    use_filters = st.sidebar.checkbox('Aplicar filtros')
    
    df_filtered = df.copy()
    
    if use_filters and len(selected_columns) > 0:
        with st.sidebar.expander('Configurar Filtros'):
            # Filtros para variables categóricas
            categorical_cols_available = [col for col in df.select_dtypes(include=['object', 'category']).columns if col in selected_columns]
            if categorical_cols_available:
                for col in categorical_cols_available:
                    unique_vals = df[col].dropna().unique().tolist()
                    if len(unique_vals) <= 20:  # Solo mostrar si no hay demasiados valores únicos
                        selected_vals = st.multiselect(
                            f'Filtrar {col}:',
                            options=unique_vals,
                            default=unique_vals,
                            key=f'filter_{col}'
                        )
                        if selected_vals:
                            df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
            
            # Filtros para variables numéricas
            numeric_cols_available = [col for col in df.select_dtypes(include=[np.number]).columns if col in selected_columns]
            if numeric_cols_available:
                for col in numeric_cols_available:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    if min_val != max_val:
                        range_vals = st.slider(
                            f'Rango de {col}:',
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f'range_{col}'
                        )
                        df_filtered = df_filtered[
                            (df_filtered[col] >= range_vals[0]) & 
                            (df_filtered[col] <= range_vals[1])
                        ]
    
    # Mostrar información del dataset filtrado
    if use_filters:
        st.sidebar.info(f'Filas después de filtrar: {df_filtered.shape[0]}')
    
    # Botón de descarga
    st.sidebar.subheader('💾 Descargar Datos')
    csv = df_filtered[selected_columns].to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Descargar CSV filtrado",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv',
    )
    
    # API Key de Groq para informes IA
    st.sidebar.subheader('🤖 Informes con IA')
    groq_api_key = st.sidebar.text_input(
        'API Key de Groq (opcional):',
        type='password',
        help='Ingresa tu API key de Groq para generar informes con IA. Obtén una gratis en https://console.groq.com'
    )
    
    # Mostrar vista previa de los datos
    st.subheader('📋 Vista Previa de los Datos')
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric('Filas mostradas', min(num_samples, df_filtered.shape[0]))
    with col_info2:
        st.metric('Columnas mostradas', len(selected_columns))
    with col_info3:
        st.metric('Total de filas', df_filtered.shape[0])
    
    st.dataframe(df_filtered[selected_columns].head(num_samples), use_container_width=True)
    
    # ============================================================
    # SEPARAR VARIABLES POR TIPO
    # ============================================================
    # Variables numéricas (cuantitativas) - usar datos filtrados
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in selected_columns]
    
    # Variables categóricas (cualitativas) - usar datos filtrados
    categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col in selected_columns]
    
    # ============================================================
    # TABS PARA LOS 3 BLOQUES PRINCIPALES
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs(['📝 Variables Cualitativas', '📈 Variables Cuantitativas', '📊 Gráficos Avanzados', '🤖 Informes IA'])
    
    # ============================================================
    # BLOQUE 1: VARIABLES CUALITATIVAS
    # ============================================================
    with tab1:
        st.header('Análisis de Variables Cualitativas')
        
        if len(categorical_cols) > 0:
            # Modo de análisis
            analysis_mode = st.radio('Modo de análisis:', ['Variable única', 'Comparar variables'], horizontal=True)
            
            if analysis_mode == 'Variable única':
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    selected_cat_var = st.selectbox('Selecciona una variable:', categorical_cols)
                
                with col2:
                    chart_type_cat = st.selectbox('Tipo de gráfico:', ['Barras', 'Pastel', 'Barras horizontales'])
                
                with col3:
                    top_n = st.number_input('Mostrar top N valores:', min_value=5, max_value=50, value=20, step=5)
                
                if selected_cat_var:
                    st.subheader(f'Análisis de: {selected_cat_var}')
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write('**Tabla de Frecuencias**')
                        freq_table = df_filtered[selected_cat_var].value_counts().reset_index()
                        freq_table.columns = [selected_cat_var, 'Frecuencia']
                        freq_table['Porcentaje'] = (freq_table['Frecuencia'] / freq_table['Frecuencia'].sum() * 100).round(2)
                        st.dataframe(freq_table, use_container_width=True)
                    
                    with col_b:
                        st.write('**Estadísticas**')
                        st.metric('Valores únicos', df_filtered[selected_cat_var].nunique())
                        st.metric('Valor más frecuente', df_filtered[selected_cat_var].mode()[0] if len(df_filtered[selected_cat_var].mode()) > 0 else 'N/A')
                        st.metric('Valores nulos', df_filtered[selected_cat_var].isnull().sum())
                    
                    # Gráficos personalizados
                    st.write(f'**Gráfico de {chart_type_cat}**')
                    freq_table_top = freq_table.head(top_n)
                    
                    if chart_type_cat == 'Barras':
                        fig = px.bar(freq_table_top, x=selected_cat_var, y='Frecuencia', 
                                    title=f'Distribución de {selected_cat_var}',
                                    color='Frecuencia', color_continuous_scale='Blues')
                    elif chart_type_cat == 'Pastel':
                        fig = px.pie(freq_table_top, names=selected_cat_var, values='Frecuencia',
                                    title=f'Distribución de {selected_cat_var}')
                    else:  # Barras horizontales
                        fig = px.bar(freq_table_top, y=selected_cat_var, x='Frecuencia', 
                                    title=f'Distribución de {selected_cat_var}',
                                    color='Frecuencia', color_continuous_scale='Blues',
                                    orientation='h')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Comparar variables
                st.subheader('Comparación de Variables Cualitativas')
                compare_vars = st.multiselect('Selecciona variables para comparar (máx. 3):', 
                                             categorical_cols,
                                             default=categorical_cols[:min(2, len(categorical_cols))])
                
                if len(compare_vars) >= 2 and len(compare_vars) <= 3:
                    # Tabla de contingencia
                    if len(compare_vars) == 2:
                        st.write('**Tabla de Contingencia**')
                        contingency = pd.crosstab(df_filtered[compare_vars[0]], df_filtered[compare_vars[1]])
                        st.dataframe(contingency, use_container_width=True)
                        
                        # Heatmap de frecuencias
                        st.write('**Heatmap de Frecuencias**')
                        fig_heat = px.imshow(contingency, text_auto=True, 
                                           color_continuous_scale='Blues',
                                           title=f'{compare_vars[0]} vs {compare_vars[1]}')
                        st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Gráfico de barras agrupadas
                    st.write('**Distribución Comparativa**')
                    for var in compare_vars:
                        freq = df_filtered[var].value_counts().reset_index()
                        freq.columns = [var, 'Frecuencia']
                        freq['Variable'] = var
                        if var == compare_vars[0]:
                            combined_freq = freq.head(10).copy()
                        else:
                            combined_freq = pd.concat([combined_freq, freq.head(10)])
                    
                    fig_compare = px.bar(combined_freq, x=compare_vars[0] if len(compare_vars) == 1 else combined_freq.columns[0], 
                                       y='Frecuencia', color='Variable',
                                       title='Comparación de Frecuencias', barmode='group')
                    st.plotly_chart(fig_compare, use_container_width=True)
                elif len(compare_vars) > 3:
                    st.warning('Por favor selecciona máximo 3 variables para comparar.')
                else:
                    st.info('Selecciona al menos 2 variables para comparar.')
        else:
            st.info('No se encontraron variables cualitativas en el dataset.')
    
    # ============================================================
    # BLOQUE 2: VARIABLES CUANTITATIVAS
    # ============================================================
    with tab2:
        st.header('Análisis de Variables Cuantitativas')
        
        if len(numeric_cols) > 0:
            # Modo de análisis
            analysis_mode_num = st.radio('Modo de análisis:', ['Variable única', 'Comparar variables', 'Resumen general'], horizontal=True)
            
            if analysis_mode_num == 'Variable única':
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    selected_num_var = st.selectbox('Selecciona una variable:', numeric_cols)
                
                with col2:
                    num_bins = st.slider('Número de bins (histograma):', min_value=10, max_value=100, value=30)
                
                with col3:
                    group_by_var = st.selectbox('Agrupar por (opcional):', ['Ninguno'] + categorical_cols)
                
                if selected_num_var:
                    st.subheader(f'Análisis de: {selected_num_var}')
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write('**Estadísticas Descriptivas**')
                        stats = df_filtered[selected_num_var].describe()
                        st.dataframe(stats, use_container_width=True)
                    
                    with col_b:
                        st.write('**Información Adicional**')
                        st.metric('Media', f"{df_filtered[selected_num_var].mean():.2f}")
                        st.metric('Mediana', f"{df_filtered[selected_num_var].median():.2f}")
                        st.metric('Desviación estándar', f"{df_filtered[selected_num_var].std():.2f}")
                        st.metric('Valores nulos', df_filtered[selected_num_var].isnull().sum())
                    
                    # Histograma y Boxplot
                    col_c, col_d = st.columns(2)
                    
                    with col_c:
                        st.write('**Histograma**')
                        if group_by_var != 'Ninguno':
                            fig_hist = px.histogram(df_filtered, x=selected_num_var, color=group_by_var,
                                                  title=f'Distribución de {selected_num_var} por {group_by_var}',
                                                  nbins=num_bins, barmode='overlay')
                        else:
                            fig_hist = px.histogram(df_filtered, x=selected_num_var, 
                                                  title=f'Distribución de {selected_num_var}',
                                                  nbins=num_bins, color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_d:
                        st.write('**Boxplot**')
                        if group_by_var != 'Ninguno':
                            fig_box = px.box(df_filtered, y=selected_num_var, x=group_by_var,
                                           title=f'Boxplot de {selected_num_var} por {group_by_var}')
                        else:
                            fig_box = px.box(df_filtered, y=selected_num_var, 
                                           title=f'Boxplot de {selected_num_var}',
                                           color_discrete_sequence=['#EF553B'])
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Gráfico de violín
                    st.write('**Gráfico de Violín**')
                    if group_by_var != 'Ninguno':
                        fig_violin = px.violin(df_filtered, y=selected_num_var, x=group_by_var,
                                             title=f'Violin Plot de {selected_num_var} por {group_by_var}',
                                             box=True)
                    else:
                        fig_violin = px.violin(df_filtered, y=selected_num_var,
                                             title=f'Violin Plot de {selected_num_var}',
                                             box=True)
                    st.plotly_chart(fig_violin, use_container_width=True)
            
            elif analysis_mode_num == 'Comparar variables':
                st.subheader('Comparación de Variables Cuantitativas')
                compare_num_vars = st.multiselect('Selecciona variables para comparar:', 
                                                 numeric_cols,
                                                 default=numeric_cols[:min(3, len(numeric_cols))])
                
                if len(compare_num_vars) >= 2:
                    # Estadísticas comparativas
                    st.write('**Estadísticas Comparativas**')
                    stats_compare = df_filtered[compare_num_vars].describe().T
                    st.dataframe(stats_compare, use_container_width=True)
                    
                    # Boxplots comparativos
                    st.write('**Boxplots Comparativos**')
                    df_melted = df_filtered[compare_num_vars].melt(var_name='Variable', value_name='Valor')
                    fig_compare_box = px.box(df_melted, x='Variable', y='Valor',
                                           title='Comparación de Distribuciones')
                    st.plotly_chart(fig_compare_box, use_container_width=True)
                    
                    # Histogramas superpuestos
                    st.write('**Histogramas Superpuestos**')
                    fig_compare_hist = go.Figure()
                    for var in compare_num_vars:
                        fig_compare_hist.add_trace(go.Histogram(x=df_filtered[var], name=var, opacity=0.7))
                    fig_compare_hist.update_layout(barmode='overlay', title='Distribuciones Comparadas')
                    st.plotly_chart(fig_compare_hist, use_container_width=True)
                else:
                    st.info('Selecciona al menos 2 variables para comparar.')
            
            else:  # Resumen general
                st.subheader('Resumen General de Variables Cuantitativas')
                
                # Tabla de estadísticas resumidas
                st.write('**Todas las Estadísticas**')
                all_stats = df_filtered[numeric_cols].describe().T
                all_stats['CV (%)'] = (all_stats['std'] / all_stats['mean'] * 100).round(2)
                st.dataframe(all_stats, use_container_width=True)
                
                # Matriz de correlación resumida
                if len(numeric_cols) > 1:
                    st.write('**Correlaciones Más Fuertes**')
                    corr_matrix = df_filtered[numeric_cols].corr()
                    
                    # Obtener las correlaciones más fuertes
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlación': corr_matrix.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlación', key=abs, ascending=False)
                    st.dataframe(corr_df.head(10), use_container_width=True)
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
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                x_var = st.selectbox('Variable X:', numeric_cols if len(numeric_cols) > 0 else df_filtered.columns)
            
            with col2:
                y_var = st.selectbox('Variable Y:', numeric_cols if len(numeric_cols) > 1 else df_filtered.columns)
            
            with col3:
                color_var = st.selectbox('Color:', ['Ninguno'] + categorical_cols + numeric_cols)
            
            with col4:
                size_var = st.selectbox('Tamaño:', ['Ninguno'] + numeric_cols)
            
            # Opciones adicionales
            show_trendline = st.checkbox('Mostrar línea de tendencia', value=False)
            
            if x_var and y_var:
                color_param = None if color_var == 'Ninguno' else color_var
                size_param = None if size_var == 'Ninguno' else size_var
                trendline_param = 'ols' if show_trendline else None
                
                fig_scatter = px.scatter(df_filtered, x=x_var, y=y_var, 
                                        color=color_param, size=size_param,
                                        title=f'{y_var} vs {x_var}',
                                        hover_data=df_filtered.columns,
                                        trendline=trendline_param)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        elif chart_type == 'Matriz de Correlación':
            st.subheader('Matriz de Correlación')
            
            if len(numeric_cols) > 1:
                # Selector de variables para correlación
                vars_for_corr = st.multiselect('Selecciona variables para correlación:', 
                                              numeric_cols,
                                              default=numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    corr_method = st.selectbox('Método de correlación:', ['pearson', 'spearman', 'kendall'])
                with col2:
                    color_scale = st.selectbox('Escala de colores:', ['RdBu_r', 'Viridis', 'Blues', 'RdYlGn_r'])
                
                if len(vars_for_corr) > 1:
                    corr_matrix = df_filtered[vars_for_corr].corr(method=corr_method)
                    
                    fig_corr = px.imshow(corr_matrix, 
                                        text_auto='.2f',
                                        color_continuous_scale=color_scale,
                                        aspect='auto',
                                        title=f'Matriz de Correlación ({corr_method.capitalize()})')
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info('Selecciona al menos 2 variables.')
            else:
                st.warning('Se necesitan al menos 2 variables numéricas para calcular la correlación.')
        
        elif chart_type == 'Gráfico de Pares':
            st.subheader('Gráfico de Pares (Pairplot)')
            
            if len(numeric_cols) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Limitar a máximo 5 variables para mejor visualización
                    selected_cols = st.multiselect('Selecciona variables (máx. 5):', 
                                                  numeric_cols, 
                                                  default=numeric_cols[:min(3, len(numeric_cols))])
                
                with col2:
                    color_by = st.selectbox('Colorear por:', ['Ninguno'] + categorical_cols)
                
                if len(selected_cols) > 1 and len(selected_cols) <= 5:
                    color_param = None if color_by == 'Ninguno' else color_by
                    
                    fig_pair = px.scatter_matrix(df_filtered, dimensions=selected_cols,
                                                color=color_param,
                                                title='Matriz de Gráficos de Pares')
                    fig_pair.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig_pair, use_container_width=True)
                elif len(selected_cols) > 5:
                    st.warning('Por favor selecciona máximo 5 variables.')
                else:
                    st.info('Selecciona al menos 2 variables.')
            else:
                st.warning('Se necesitan al menos 2 variables numéricas.')
    
    # ============================================================
    # BLOQUE 4: INFORMES CON IA
    # ============================================================
    with tab4:
        st.header('Informes Detallados con IA')
        
        if groq_api_key:
            try:
                # Inicializar cliente de Groq
                client = Groq(api_key=groq_api_key)
                
                st.success('✅ API Key configurada correctamente')
                
                # Tipo de informe
                report_type = st.selectbox(
                    'Selecciona el tipo de informe:',
                    ['Análisis Exploratorio Completo', 
                     'Análisis de Variable Específica',
                     'Insights y Recomendaciones',
                     'Análisis de Correlaciones',
                     'Detección de Anomalías']
                )
                
                # Configuración adicional según el tipo de informe
                if report_type == 'Análisis de Variable Específica':
                    all_cols = selected_columns
                    var_to_analyze = st.selectbox('Selecciona la variable:', all_cols)
                
                # Nivel de detalle
                detail_level = st.select_slider(
                    'Nivel de detalle del informe:',
                    options=['Resumido', 'Normal', 'Detallado', 'Muy Detallado'],
                    value='Normal'
                )
                
                # Botón para generar informe
                if st.button('🚀 Generar Informe', type='primary', use_container_width=True):
                    with st.spinner('Generando informe con IA... Esto puede tomar unos momentos.'):
                        
                        # Preparar estadísticas del dataset
                        stats_summary = {
                            'filas': int(df_filtered.shape[0]),
                            'columnas': int(df_filtered.shape[1]),
                            'columnas_numericas': len(numeric_cols),
                            'columnas_categoricas': len(categorical_cols)
                        }
                        
                        # Estadísticas básicas
                        if len(numeric_cols) > 0:
                            stats_summary['estadisticas_numericas'] = df_filtered[numeric_cols].describe().to_dict()
                        
                        if len(categorical_cols) > 0:
                            stats_summary['estadisticas_categoricas'] = {}
                            for col in categorical_cols[:5]:  # Limitar a 5 para no saturar
                                stats_summary['estadisticas_categoricas'][col] = {
                                    'valores_unicos': int(df_filtered[col].nunique()),
                                    'top_valores': df_filtered[col].value_counts().head(5).to_dict()
                                }
                        
                        # Correlaciones si hay variables numéricas
                        if len(numeric_cols) > 1:
                            corr_matrix = df_filtered[numeric_cols].corr()
                            # Top 5 correlaciones más fuertes
                            corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    corr_pairs.append({
                                        'var1': corr_matrix.columns[i],
                                        'var2': corr_matrix.columns[j],
                                        'corr': float(corr_matrix.iloc[i, j])
                                    })
                            corr_pairs.sort(key=lambda x: abs(x['corr']), reverse=True)
                            stats_summary['top_correlaciones'] = corr_pairs[:5]
                        
                        # Construir el prompt según el tipo de informe
                        if report_type == 'Análisis Exploratorio Completo':
                            prompt = f"""Eres un analista de datos experto. Genera un informe de análisis exploratorio completo basado en los siguientes datos:

Estadísticas del Dataset:
{json.dumps(stats_summary, indent=2, ensure_ascii=False)}

Columnas disponibles: {', '.join(selected_columns)}

Nivel de detalle: {detail_level}

Por favor, proporciona:
1. Resumen ejecutivo del dataset
2. Análisis de variables numéricas (distribuciones, outliers, tendencias)
3. Análisis de variables categóricas (frecuencias, patrones)
4. Análisis de correlaciones y relaciones entre variables
5. Hallazgos clave e insights
6. Recomendaciones para análisis posteriores

Formatea el informe en Markdown con secciones claras."""
                        
                        elif report_type == 'Análisis de Variable Específica':
                            var_stats = {}
                            if var_to_analyze in numeric_cols:
                                var_stats = df_filtered[var_to_analyze].describe().to_dict()
                                var_stats['tipo'] = 'numérica'
                            else:
                                var_stats = {
                                    'tipo': 'categórica',
                                    'valores_unicos': int(df_filtered[var_to_analyze].nunique()),
                                    'distribucion': df_filtered[var_to_analyze].value_counts().head(10).to_dict()
                                }
                            
                            prompt = f"""Eres un analista de datos experto. Genera un informe detallado sobre la variable '{var_to_analyze}'.

Estadísticas de la variable:
{json.dumps(var_stats, indent=2, ensure_ascii=False)}

Nivel de detalle: {detail_level}

Por favor, proporciona:
1. Descripción general de la variable
2. Análisis de distribución
3. Identificación de valores atípicos o anomalías
4. Patrones y tendencias observadas
5. Relación con otras variables (si es relevante)
6. Recomendaciones de uso o tratamiento

Formatea el informe en Markdown."""
                        
                        elif report_type == 'Insights y Recomendaciones':
                            prompt = f"""Eres un analista de datos experto. Basándote en las siguientes estadísticas, genera insights accionables y recomendaciones:

Estadísticas del Dataset:
{json.dumps(stats_summary, indent=2, ensure_ascii=False)}

Columnas: {', '.join(selected_columns)}

Nivel de detalle: {detail_level}

Por favor, proporciona:
1. Top 5 insights más importantes del dataset
2. Oportunidades de análisis identificadas
3. Recomendaciones de mejora de calidad de datos
4. Sugerencias de visualizaciones adicionales
5. Próximos pasos recomendados

Formatea el informe en Markdown."""
                        
                        elif report_type == 'Análisis de Correlaciones':
                            prompt = f"""Eres un analista de datos experto. Analiza las correlaciones entre variables:

Estadísticas del Dataset:
{json.dumps(stats_summary, indent=2, ensure_ascii=False)}

Nivel de detalle: {detail_level}

Por favor, proporciona:
1. Análisis de las correlaciones más fuertes
2. Interpretación de las relaciones encontradas
3. Correlaciones inesperadas o interesantes
4. Advertencias sobre posibles correlaciones espurias
5. Recomendaciones para análisis de causalidad

Formatea el informe en Markdown."""
                        
                        else:  # Detección de Anomalías
                            prompt = f"""Eres un analista de datos experto. Basándote en las estadísticas, identifica posibles anomalías y problemas en los datos:

Estadísticas del Dataset:
{json.dumps(stats_summary, indent=2, ensure_ascii=False)}

Nivel de detalle: {detail_level}

Por favor, proporciona:
1. Valores atípicos detectados en variables numéricas
2. Patrones inusuales en variables categóricas
3. Problemas de calidad de datos identificados
4. Distribuciones anómalas
5. Recomendaciones de limpieza y preprocesamiento

Formatea el informe en Markdown."""
                        
                        try:
                            # Llamar a la API de Groq
                            chat_completion = client.chat.completions.create(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "Eres un analista de datos experto que genera informes detallados, claros y accionables en español."
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                model="llama-3.3-70b-versatile",
                                temperature=0.7,
                                max_tokens=4000
                            )
                            
                            # Obtener el informe generado
                            report_content = chat_completion.choices[0].message.content
                            
                            # Mostrar el informe
                            st.subheader('📄 Informe Generado')
                            st.markdown(report_content)
                            
                            # Preparar el informe para descarga
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            report_filename = f"informe_{report_type.replace(' ', '_').lower()}_{timestamp}.md"
                            
                            # Agregar metadatos al informe
                            full_report = f"""# {report_type}
**Generado:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** {uploaded_file.name}
**Filas analizadas:** {df_filtered.shape[0]}
**Columnas analizadas:** {len(selected_columns)}
**Nivel de detalle:** {detail_level}

---

{report_content}

---

*Informe generado automáticamente con IA (Groq)*
"""
                            
                            # Botón de descarga
                            st.download_button(
                                label="📥 Descargar Informe (Markdown)",
                                data=full_report.encode('utf-8'),
                                file_name=report_filename,
                                mime='text/markdown',
                                use_container_width=True
                            )
                            
                            # También ofrecer descarga en texto plano
                            st.download_button(
                                label="📥 Descargar Informe (TXT)",
                                data=full_report.encode('utf-8'),
                                file_name=report_filename.replace('.md', '.txt'),
                                mime='text/plain',
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f'❌ Error al generar el informe: {str(e)}')
                            st.info('Verifica que tu API key sea correcta y tenga créditos disponibles.')
            
            except Exception as e:
                st.error(f'❌ Error al conectar con Groq: {str(e)}')
                st.info('Verifica que tu API key sea válida.')
        
        else:
            st.info('🔑 Para generar informes con IA, ingresa tu API Key de Groq en la barra lateral.')
            st.markdown("""
            ### ¿Cómo obtener una API Key de Groq?
            
            1. Visita [https://console.groq.com](https://console.groq.com)
            2. Crea una cuenta gratuita
            3. Genera una nueva API Key
            4. Copia y pega la clave en el campo de la barra lateral
            
            ### ¿Qué tipos de informes puedes generar?
            
            - **Análisis Exploratorio Completo:** Resumen integral del dataset
            - **Análisis de Variable Específica:** Análisis profundo de una columna
            - **Insights y Recomendaciones:** Hallazgos clave y próximos pasos
            - **Análisis de Correlaciones:** Relaciones entre variables
            - **Detección de Anomalías:** Identificación de outliers y problemas
            
            Todos los informes son descargables en formato Markdown y TXT.
            """)

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