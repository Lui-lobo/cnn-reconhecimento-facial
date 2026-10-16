# Pré-processamento das fotos, feito só com NumPy (sem OpenCV e sem funções prontas de imagem).
# Toda foto passa pelas mesmas 5 etapas: recorte central -> cinza -> 96x96 -> equalização -> escala 0-1.
import numpy as np

from src.config import IMG_SIZE


def recorte_central(img):
    # Etapa 1: recorta o maior quadrado possível do centro da imagem.
    # A CNN exige entrada quadrada de tamanho fixo; cortar não deforma o rosto (redimensionar direto o achataria).
    altura, largura = img.shape[:2]
    lado = min(altura, largura)                  # o lado do quadrado é a menor dimensão
    topo = (altura - lado) // 2                  # quanto sobra em cima
    esquerda = (largura - lado) // 2             # quanto sobra à esquerda
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


def preprocessar(img, mostrar_etapas=False):
    # Aplica as 5 etapas em sequência. Usada no treino E no reconhecimento ao vivo (mesmo tratamento).
    # Com mostrar_etapas=True, devolve também as imagens intermediárias: [original, recorte, cinza, 96x96, equalizada].
    e1 = recorte_central(img)
    e2 = para_cinza(e1)
    e3 = redimensionar(e2, IMG_SIZE)
    e4 = equalizar_histograma(e3)
    e5 = e4 / 255.0                                          # Etapa 5: valores entre 0 e 1
    final = e5.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)   # a CNN espera (altura, largura, canais)
    if mostrar_etapas:
        return final, [img, e1, e2, e3, e4]
    return final
