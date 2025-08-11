import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib


# Carregar os dados do CSV gerado
# skiprows=1 para pular o cabeçalho
dados = np.loadtxt('dados_trajetoria.csv', delimiter=',', skiprows=1)
t = dados[:, 0]  # Coluna de tempo
t_x = dados[:, 0:2]  # Entradas: tempo e x (para prever y)
y = dados[:, 2]  # Saída: y

# Normalizar os dados de entrada e saída para melhorar o treinamento
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
t_x_normalized = scaler_x.fit_transform(t_x)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1))


# Salvar os scalers
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Criar um modelo de rede neural sequencial
# Camada de entrada: 2 neurônios (t e x)
# Camadas ocultas: duas com 64 neurônios cada, ativação ReLU
# Camada de saída: 1 neurônio (previsão de y)
modelo = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compilar o modelo
# Otimizador: Adam (comum para treinamento)
# Função de perda: Erro Quadrático Médio (para regressão)
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Treinar com dados normalizados e mais épocas
modelo.fit(t_x_normalized, y_normalized, epochs=2000, validation_split=0.2)

# Salvar o modelo treinado para uso posterior
modelo.save('modelo_trajetoria.h5')
print('Modelo treinado e salvo!')