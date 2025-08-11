import numpy as np
import matplotlib.pyplot as plt

# Definir parâmetros da simulação
g = 9.8  # aceleração gravitacional em m/s²
t = np.linspace(0, 10, 100)  # array de tempos de 0 a 10 segundos, com 100 pontos
v0 = 50  # velocidade inicial em m/s
angulo = np.pi / 4  # ângulo de lançamento em radianos (45 graus)

# Calcular posições x e y usando equações de movimento balístico
# x = v0 * cos(angulo) * t
# y = v0 * sin(angulo) * t - (1/2) * g * t²
x = v0 * np.cos(angulo) * t
y = v0 * np.sin(angulo) * t - 0.5 * g * t**2

# Plotar a trajetória para visualização
plt.plot(x, y)
plt.xlabel('Distância (m)')  # Rótulo do eixo x
plt.ylabel('Altura (m)')  # Rótulo do eixo y
plt.title('Trajetória de uma Partícula')  # Título do gráfico
plt.show()  # Exibir o gráfico

# Salvar os dados em um arquivo CSV para uso no treinamento do modelo
# As colunas são: tempo (t), posição x, posição y
np.savetxt('dados_trajetoria.csv', np.column_stack((t, x, y)), delimiter=',', header='tempo,x,y', comments='')