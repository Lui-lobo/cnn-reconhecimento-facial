# Aumento de dados feito com NumPy (sem ImageDataGenerator e sem camadas prontas do Keras).
# Cria versões levemente modificadas das fotos de TREINO para a rede não decorar as imagens (overfitting).
import numpy as np

from src.config import IMG_SIZE, COPIAS_AUMENTADAS


def aumentar(img):
    # Recebe uma imagem (96, 96, 1) com valores 0-1 e devolve uma versão modificada aleatoriamente

    # 1. Espelhamento horizontal (50% de chance): simula o rosto virado para o outro lado
    if np.random.rand() < 0.5:
        img = img[:, ::-1]

    # 2. Deslocamento: aumenta 8 px de borda (repetindo os pixels da beirada) e recorta numa posição sorteada
    dy, dx = np.random.randint(-8, 9, size=2)
    com_borda = np.pad(img, ((8, 8), (8, 8), (0, 0)), mode="edge")
    img = com_borda[8 + dy : 8 + dy + IMG_SIZE, 8 + dx : 8 + dx + IMG_SIZE]

    # 3. Contraste (multiplica a distância até o cinza médio) e brilho (soma um valor): iluminação diferente
    img = (img - 0.5) * np.random.uniform(0.8, 1.2) + 0.5 + np.random.uniform(-0.1, 0.1)

    # 4. Ruído: soma um valor aleatório pequeno a cada pixel (câmera de qualidade pior)
    img = img + np.random.normal(0, 0.02, img.shape)

    return np.clip(img, 0, 1).astype(np.float32)    # garante que os valores fiquem entre 0 e 1


def montar_treino_aumentado(X_treino, y_treino):
    # Monta o treino final: as fotos originais + COPIAS_AUMENTADAS versões modificadas de cada uma
    X_final = [X_treino]
    y_final = [y_treino]
    for _ in range(COPIAS_AUMENTADAS):
        X_final.append(np.array([aumentar(img) for img in X_treino]))
        y_final.append(y_treino)          # o rótulo não muda: continua sendo a mesma pessoa
    return np.concatenate(X_final), np.concatenate(y_final)
