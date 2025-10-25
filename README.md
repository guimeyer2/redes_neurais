# Trabalho - Redes Neurais com Backpropagation

Este repositório contém duas redes neurais feedforward implementadas do zero usando apenas NumPy
(sem Keras / PyTorch / Scikit-learn):

## 1. Problema XOR

- Entrada: 2 neurônios
- Camada oculta: 2 neurônios
- Saída: 1 neurônio
- Ativação: sigmóide em todas as camadas
- Loss: MSE
- Otimização: gradiente descendente clássico
- A rede aprende a função XOR, que não é linearmente separável.

## 2. Reconhecimento de dígitos em display de 7 segmentos

- Entrada: 7 neurônios (segmentos a,b,c,d,e,f,g)
- Camada oculta: 5 neurônios
- Saída: 4 neurônios (representando o dígito [0–9] em binário de 4 bits)
- Também é avaliada a robustez com ruído (simulando falha de um LED/segmento).

## Como rodar

Pré-requisito:

```bash
pip install numpy
```
