# Teste de troca de fundo: a rede está olhando para o rosto ou para o resto da imagem?
# Montamos imagens com o ROSTO de um integrante e o ENTORNO (fundo, roupa, cabelo) da foto de outro integrante.
# Se a rede responde o dono do rosto, aprendeu o rosto; se responde o dono do entorno, aprendeu o cenário.
import numpy as np

from src.config import IMG_SIZE
from src.preprocessamento import mascara_oval


def trocar_fundo(X_rosto, X_entorno):
    # Junta o miolo do oval (do tamanho do rosto) de X_rosto com o resto da imagem de X_entorno
    m = mascara_oval(IMG_SIZE, escala=0.7)[:, :, np.newaxis]    # 1 no rosto, 0 em volta
    return (X_rosto * m + X_entorno * (1 - m)).astype(np.float32)


def teste_troca_de_fundo(model, X, y, indice_ninguem, gerador):
    # Para cada foto de teste de um integrante, sorteia uma foto de OUTRO integrante para servir de entorno
    pessoas = np.where(y != indice_ninguem)[0]
    entornos = np.array([gerador.choice(pessoas[y[pessoas] != y[i]]) for i in pessoas])
    previsto = model.predict(trocar_fundo(X[pessoas], X[entornos]), verbose=0).argmax(axis=1)

    # E as fotos de teste sem ninguém: a rede deveria responder "ninguém"
    vazias = y == indice_ninguem
    previsto_vazias = model.predict(X[vazias], verbose=0).argmax(axis=1)
    return {
        "segue_o_rosto": float(np.mean(previsto == y[pessoas])),               # quanto maior, melhor
        "segue_o_entorno": float(np.mean(previsto == y[entornos])),            # quanto MENOR, melhor
        "vazio_e_ninguem": float(np.mean(previsto_vazias == indice_ninguem)),  # quanto maior, melhor
    }
