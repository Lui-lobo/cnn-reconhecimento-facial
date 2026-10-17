# Aumento de dados feito com NumPy (sem ImageDataGenerator e sem camadas prontas do Keras).
# Cria versões levemente modificadas das fotos de TREINO para a rede não decorar as imagens (overfitting).
import numpy as np

from src.config import IMG_SIZE, COPIAS_AUMENTADAS
from src.preprocessamento import aplicar_mascara_oval
from src.robustez import trocar_fundo


def aumentar(img):
    # Recebe uma imagem (96, 96, 1) com valores 0-1 e devolve uma versão modificada aleatoriamente

    # 1. Espelhamento horizontal (50% de chance): simula o rosto virado para o outro lado
    if np.random.rand() < 0.5:
        img = img[:, ::-1]

    # 2. Deslocamento: aumenta 8 px de borda (repetindo os pixels da beirada) e recorta numa posição sorteada
    dy, dx = np.random.randint(-8, 9, size=2)
    com_borda = np.pad(img, ((8, 8), (8, 8), (0, 0)), mode="edge")
    img = com_borda[8 + dy : 8 + dy + IMG_SIZE, 8 + dx : 8 + dx + IMG_SIZE]

    # 3. Esconde mais fundo: aplica de novo a máscara oval, encolhida até 25% (o cinza avança sobre a borda do oval).
    #    Cada cópia mostra uma quantidade diferente de fundo, então o fundo deixa de ser uma pista confiável.
    img = aplicar_mascara_oval(img[:, :, 0], escala=np.random.uniform(0.75, 1.0), cinza=0.5)[:, :, np.newaxis]

    # 4. Contraste (multiplica a distância até o cinza médio) e brilho (soma um valor): iluminação diferente
    img = (img - 0.5) * np.random.uniform(0.8, 1.2) + 0.5 + np.random.uniform(-0.1, 0.1)

    # 5. Ruído: soma um valor aleatório pequeno a cada pixel (câmera de qualidade pior)
    img = img + np.random.normal(0, 0.02, img.shape)

    return np.clip(img, 0, 1).astype(np.float32)    # garante que os valores fiquem entre 0 e 1


def variar_cena(i, X, X_distancia, y, indice_ninguem):
    # Ponto de partida de uma cópia da foto i, ANTES do aumentar:
    # 1. Distância (50% de chance): usa a mesma foto recortada com outro tamanho de quadrado, como se a pessoa
    #    estivesse mais perto ou mais longe da câmera (o rosto fica maior ou menor dentro do oval).
    # 2. Troca de entorno (50% de chance, só para integrantes): o anel em volta do rosto vem da foto de OUTRA
    #    classe, e o rótulo continua o do dono do rosto. Assim o cenário deixa de indicar quem é a pessoa.
    img = X_distancia[i] if np.random.rand() < 0.5 else X[i]
    if y[i] != indice_ninguem and np.random.rand() < 0.5:
        outra = np.random.choice(np.where(y != y[i])[0])        # uma foto de qualquer outra classe
        img = trocar_fundo(img, X[outra])
    return img


def montar_treino_aumentado(X_treino, X_distancia, y_treino, indice_ninguem):
    # Monta o treino final: as fotos originais + COPIAS_AUMENTADAS versões modificadas de cada uma
    X_final = [X_treino]
    y_final = [y_treino]
    for _ in range(COPIAS_AUMENTADAS):
        X_final.append(np.array([aumentar(variar_cena(i, X_treino, X_distancia, y_treino, indice_ninguem))
                                 for i in range(len(X_treino))]))
        y_final.append(y_treino)          # o rótulo não muda: continua sendo a mesma pessoa
    return np.concatenate(X_final), np.concatenate(y_final)
