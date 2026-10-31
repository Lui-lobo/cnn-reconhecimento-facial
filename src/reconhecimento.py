# Funções usadas pelo servidor para reconhecer quem está na webcam, e o Grad-CAM ("onde a rede olhou").
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import IMG_SIZE, CAMINHO_MODELO, CAMINHO_CLASSES
from src.preprocessamento import preprocessar, caixa_do_rosto


def carregar_modelo():
    # Lê do disco a CNN treinada e a lista com os nomes dos integrantes (na ordem dos rótulos)
    model = keras.models.load_model(CAMINHO_MODELO)
    with open(CAMINHO_CLASSES, encoding="utf-8") as f:
        classes = json.load(f)
    return model, classes


def regioes_da_tela(largura, n_pessoas):
    # Divide a tela em n faixas verticais, uma por pessoa: devolve (x inicial, largura) de cada faixa.
    # Sem detector de rosto, a rede não sabe onde estão os rostos: cada pessoa se posiciona em uma faixa.
    # Com 1 pessoa, a faixa é a tela inteira: o mesmo enquadramento das fotos de treino.
    faixa = largura // n_pessoas
    return [(i * faixa, faixa) for i in range(n_pessoas)]


def reconhecer_quadro(model, quadro, n_pessoas):
    # Aplica o MESMO pré-processamento do treino em cada faixa do quadro da webcam e passa tudo pela CNN.
    # Devolve uma lista com a posição do quadrado do rosto de cada faixa e a probabilidade de cada classe.
    altura, largura = quadro.shape[:2]
    entradas, caixas = [], []
    for x0, faixa in regioes_da_tela(largura, n_pessoas):
        entradas.append(preprocessar(quadro[:, x0:x0 + faixa]))      # recorta o quadrado do rosto da faixa
        topo, esquerda, lado = caixa_do_rosto(altura, faixa)         # onde está esse quadrado, para desenhar
        caixas.append((x0 + esquerda, topo, lado))
    probs = model(np.array(entradas), training=False).numpy()       # todas as faixas de uma vez
    return [{"x": int(x), "y": int(y), "lado": int(lado), "probabilidades": probs[i].tolist()}
            for i, (x, y, lado) in enumerate(caixas)]


def criar_modelo_gradcam(model):
    # Modelo auxiliar que devolve, ao mesmo tempo, os mapas da conv4 e as pontuações (antes do softmax)
    return keras.Model(model.input, [model.get_layer("conv4").output, model.get_layer("pontuacoes").output])


def ampliar_bilinear(img, tamanho):
    # Amplia um mapa quadrado pequeno (ex.: 12x12) para tamanho x tamanho por interpolação bilinear:
    # cada pixel novo é a média dos 4 vizinhos da grade pequena, pesada pela distância. Assim o mapa de calor
    # fica suave e acompanha o contorno do rosto, em vez de virar quadrados de 8x8 pixels (vizinho mais próximo).
    n = img.shape[0]
    pos = np.clip((np.arange(tamanho) + 0.5) * n / tamanho - 0.5, 0, n - 1)   # posição de cada pixel novo na grade pequena
    i0 = np.floor(pos).astype(int)                                             # vizinho de cima / da esquerda
    i1 = np.minimum(i0 + 1, n - 1)                                             # vizinho de baixo / da direita
    t = pos - i0                                                               # distância até o primeiro vizinho (0-1)
    linhas = img[i0] * (1 - t)[:, None] + img[i1] * t[:, None]                 # interpola na vertical
    return linhas[:, i0] * (1 - t) + linhas[:, i1] * t                         # e depois na horizontal


def grad_cam(modelo_gradcam, img):
    # Gera um mapa de calor (96x96, valores 0-1) das regiões da foto que mais pesaram na decisão.
    # Usamos a pontuação de antes do softmax: com quase 100% de certeza, a probabilidade quase não muda
    # e o gradiente sairia zerado.
    with tf.GradientTape() as fita:                              # "grava" as contas para calcular gradientes
        mapas, pontuacoes = modelo_gradcam(img[np.newaxis])
        classe = int(tf.argmax(pontuacoes[0]))                   # integrante previsto
        nota = pontuacoes[0, classe]                             # pontuação dele
    gradientes = fita.gradient(nota, mapas)                      # quanto a nota muda com cada valor dos mapas
    importancia = tf.reduce_mean(gradientes, axis=(0, 1, 2))     # importância média de cada mapa
    calor = tf.nn.relu(tf.reduce_sum(mapas[0] * importancia, axis=-1)).numpy()   # soma ponderada (só positivos)
    calor = calor / (calor.max() + 1e-9)                         # escala para 0-1
    return ampliar_bilinear(calor, IMG_SIZE), classe             # amplia de 12x12 para 96x96, suavizando
