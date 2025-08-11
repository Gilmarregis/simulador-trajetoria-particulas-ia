# Simulador de Movimento de Partículas com IA

## Descrição
Este projeto simula o movimento balístico de uma partícula sob gravidade e usa uma rede neural para prever a trajetória. É inspirado em conceitos de física teórica e machine learning, ideal para quem tem background em física e matemática.

## Requisitos
- Python 3.x
- Bibliotecas: `numpy`, `matplotlib`, `tensorflow`

Instale com:
```bash
pip install numpy matplotlib tensorflow
```

## Estrutura dos Arquivos
- `gerar_dados.py`: Gera dados de trajetória e salva em `dados_trajetoria.csv`.
- `treinar_modelo.py`: Treina uma rede neural com os dados e salva o modelo em `modelo_trajetoria.h5`.
- `prever_trajetoria.py`: Carrega o modelo e faz previsões, comparando com valores reais.
- `dados_trajetoria.csv`: Arquivo de dados gerado (não versionado).
- `modelo_trajetoria.h5`: Modelo treinado (não versionado).

## Como Executar
1. Rode `python gerar_dados.py` para gerar os dados.
2. Rode `python treinar_modelo.py` para treinar o modelo (pode levar alguns minutos).
3. Rode `python prever_trajetoria.py` para ver as previsões.

## Documentação Técnica
- **Modelo**: Rede neural feedforward com duas camadas ocultas (64 neurônios cada).
- **Treinamento**: 500 épocas, perda MSE, otimizador Adam.
- **Entrada**: Tempo (t) e posição x.
- **Saída**: Posição y prevista.
- **Física Base**: Equações de movimento balístico sem resistência do ar.

## Limitações
- O modelo é uma aproximação; para cenários reais, considere mais variáveis (ex: ar).
- Expanda adicionando mais dados ou complexidade ao modelo.

## Autor
Gerado com assistência de IA para fins educacionais.