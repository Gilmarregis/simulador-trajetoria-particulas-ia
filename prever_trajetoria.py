import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Carregar o modelo treinado
modelo = keras.models.load_model('modelo_trajetoria.h5')

# Gerar novos dados de teste
t_teste = np.linspace(0, 10, 100)
x_teste = 50 * np.cos(np.pi / 4) * t_teste
t_x_teste = np.column_stack((t_teste, x_teste))

# Normalizar os dados de teste (usando os mesmos scalers do treinamento)
# Nota: Você precisa salvar os scalers no treinar_modelo.py ou recriá-los aqui com os mesmos dados.
# Para simplicidade, vamos assumir recriação (mas idealmente, salve-os com joblib).
from sklearn.preprocessing import MinMaxScaler
import joblib
# Carregue se salvos, ou recrie (melhor salvar no treinamento)
# Para este exemplo, recriamos com os dados originais
dados = np.loadtxt('dados_trajetoria.csv', delimiter=',', skiprows=1)
t_x_original = dados[:, 0:2]
y_original = dados[:, 2].reshape(-1, 1)
scaler_x = MinMaxScaler().fit(t_x_original)
scaler_y = MinMaxScaler().fit(y_original)
t_x_teste_normalized = scaler_x.transform(t_x_teste)

# Fazer previsões
y_pred_normalized = modelo.predict(t_x_teste_normalized)
y_pred = scaler_y.inverse_transform(y_pred_normalized)

# Calcular y real para comparação
y_real = 50 * np.sin(np.pi / 4) * t_teste - 0.5 * 9.8 * t_teste**2

# Plotar resultados
plt.plot(x_teste, y_real, label='Real')
plt.plot(x_teste, y_pred, label='Previsto pela IA', linestyle='--')
plt.xlabel('Distância (m)')
plt.ylabel('Altura (m)')
plt.title('Previsão de Trajetória')
plt.legend()
plt.show()