# Pré-processamento das fotos, feito só com NumPy (sem OpenCV e sem funções prontas de imagem).
# Toda foto passa pelas mesmas 6 etapas: recorte do rosto -> cinza -> 96x96 -> equalização -> máscara oval -> escala 0-1.
import numpy as np

from src.config import IMG_SIZE, FRACAO_ROSTO, ALTURA_ROSTO


def caixa_do_rosto(altura, largura, fracao=FRACAO_ROSTO):
    # Onde fica o quadrado do rosto em uma foto (altura x largura): devolve (topo, esquerda, lado).
    # Sem detector de rosto, usamos uma posição fixa: centralizado na horizontal e um pouco acima do meio.
    # "fracao" muda o tamanho do quadrado: maior = a pessoa parece mais longe (usado no aumento de dados).
    lado = max(1, int(min(altura, largura) * fracao))                       # 60% do menor lado
    topo = int(np.clip(altura * ALTURA_ROSTO - lado / 2, 0, altura - lado))  # centro a 45% da altura
    esquerda = (largura - lado) // 2                                        # centralizado na horizontal
    return topo, esquerda, lado


def recorte_rosto(img, fracao=FRACAO_ROSTO):
    # Etapa 1: recorta o quadrado onde o rosto deve estar.
    # A CNN exige entrada quadrada; cortar não deforma o rosto (redimensionar direto o achataria),
    # e um quadrado mais fechado deixa de fora a maior parte do fundo (que a rede poderia "decorar").
    topo, esquerda, lado = caixa_do_rosto(*img.shape[:2], fracao)
    return img[topo:topo + lado, esquerda:esquerda + lado]


def para_cinza(img):
    # Etapa 2: média ponderada dos canais R, G e B (pesos da sensibilidade do olho humano).
    # 1 canal em vez de 3: a identidade está na forma do rosto, não na cor.
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def redimensionar(img, tamanho):
    # Etapa 3: reduz uma imagem quadrada 2D para tamanho x tamanho.
    # A média de blocos evita o serrilhado de pegar só 1 pixel a cada N.
    fator = img.shape[0] // tamanho              # quantos pixels originais cabem em 1 pixel novo
    if fator > 1:
        # (a) média de blocos: cada bloco fator x fator vira 1 pixel (a média dele)
        n = img.shape[0] // fator
        img = img[:n * fator, :n * fator]        # descarta as sobras para a divisão ficar exata
        img = img.reshape(n, fator, n, fator).mean(axis=(1, 3))
    # (b) vizinho mais próximo: para cada pixel novo, pega o pixel original correspondente
    indices = (np.arange(tamanho) * img.shape[0] / tamanho).astype(int)
    return img[indices][:, indices]


def equalizar_histograma(img):
    # Etapa 4: redistribui os tons de cinza usando o histograma acumulado (CDF).
    # Assim, fotos claras e escuras ficam parecidas.
    img = img.astype(np.uint8)                              # tons inteiros de 0 a 255
    histograma = np.bincount(img.ravel(), minlength=256)    # quantos pixels existem de cada tom
    acumulado = histograma.cumsum()                         # quantos pixels têm tom <= t
    acumulado = (acumulado - acumulado.min()) / max(acumulado.max() - acumulado.min(), 1) * 255
    return acumulado[img]                                   # troca cada tom pelo seu valor acumulado


def mascara_oval(tamanho, escala=1.0):
    # Peso de cada pixel de uma imagem tamanho x tamanho: 1 dentro de um oval do formato de um rosto, 0 fora,
    # com uma transição suave na borda. "escala" encolhe o oval (usado no aumento de dados e no teste de troca de fundo).
    centro = (tamanho - 1) / 2
    y, x = np.mgrid[0:tamanho, 0:tamanho] - centro                         # distância de cada pixel ao centro
    r = np.sqrt((x / (0.40 * tamanho * escala)) ** 2 + (y / (0.48 * tamanho * escala)) ** 2)   # r = 1 na borda do oval
    return np.clip((1 - r) / 0.12, 0, 1)                                    # 1 até r = 0,88; cai até 0 em r = 1


def aplicar_mascara_oval(img, escala=1.0, cinza=127.5):
    # Etapa 5: mantém o oval do rosto e troca o resto (o fundo) por um cinza neutro.
    # Assim a rede quase não vê o ambiente e precisa aprender pelo rosto.
    m = mascara_oval(img.shape[0], escala)
    return img * m + cinza * (1 - m)                                        # mistura: rosto onde m = 1, cinza onde m = 0


def preprocessar(img, mostrar_etapas=False, fracao=FRACAO_ROSTO):
    # Aplica as 6 etapas em sequência. Usada no treino E no reconhecimento ao vivo (mesmo tratamento).
    # Com mostrar_etapas=True, devolve também as imagens intermediárias:
    # [original, recorte, cinza, 96x96, equalizada, com máscara].
    # "fracao" só muda no aumento de dados (para simular a pessoa mais perto ou mais longe da câmera).
    e1 = recorte_rosto(img, fracao)
    e2 = para_cinza(e1)
    e3 = redimensionar(e2, IMG_SIZE)
    e4 = equalizar_histograma(e3)
    e5 = aplicar_mascara_oval(e4)
    e6 = e5 / 255.0                                          # Etapa 6: valores entre 0 e 1
    final = e6.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)   # a CNN espera (altura, largura, canais)
    if mostrar_etapas:
        return final, [img, e1, e2, e3, e4, e5]
    return final
