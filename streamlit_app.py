import streamlit as st

# Título de la aplicación
st.title('Mi Aplicación de Streamlit')

# Descripción de la aplicación
st.write('Esta es una aplicación de ejemplo construida con Streamlit.')

# Sección de entrada de usuario
user_input = st.text_input('Introduce algo:')

# Mostrar la entrada del usuario
if user_input:
    st.write(f'Has introducido: {user_input}')