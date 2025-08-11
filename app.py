import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib

# Carregar modelo e scalers
model = keras.models.load_model('modelo_trajetoria.h5')
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

st.title('Simulador de Trajetória de Partículas com IA')

# Inputs do usuário
v0 = st.slider('Velocidade Inicial (m/s)', 0.0, 100.0, 50.0)
angulo = st.slider('Ângulo de Lançamento (graus)', 0.0, 90.0, 45.0)
t_max = st.slider('Tempo Máximo (s)', 0.0, 20.0, 10.0)

if st.button('Prever Trajetória'):
    # Converter ângulo para radianos
    theta = np.radians(angulo)
    
    # Gerar pontos de tempo
    tempos = np.linspace(0, t_max, 100)
    
    # Calcular x e y teóricos (para comparação, opcional)
    g = 9.8
    x_teorico = v0 * np.cos(theta) * tempos
    y_teorico = v0 * np.sin(theta) * tempos - 0.5 * g * tempos**2
    
    # Preparar inputs para o modelo: [[tempo, x] for each point]
    inputs = np.column_stack((tempos, x_teorico))
    inputs_scaled = scaler_x.transform(inputs)
    predictions_scaled = model.predict(inputs_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_teorico, y_teorico, 'b-', label='Teórico')  # Linha teórica (opcional)
    ax.plot(x_teorico, predictions, 'r--', label='Previsto pela IA')  # Previsão
    ax.set_xlabel('Posição X (m)')
    ax.set_ylabel('Posição Y (m)')
    ax.set_title('Trajetória da Partícula')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)