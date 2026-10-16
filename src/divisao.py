# Divisão das fotos em treino (~70%), validação (~15%) e teste (~15%).
# Fotos tiradas em sequência são quase idênticas; por isso sorteamos BLOCOS de fotos seguidas,
# e não fotos soltas. Assim, uma foto do teste nunca tem uma "gêmea" no treino.
import numpy as np

from src.config import SEED


def dividir_dados(rotulos, classes):
    # Devolve os índices das fotos de treino, validação e teste.
    # A divisão é feita por integrante, para que todos apareçam nos três conjuntos.
    gerador = np.random.default_rng(SEED)
    treino, validacao, teste = [], [], []

    for classe in range(len(classes)):
        indices = np.where(rotulos == classe)[0]                          # fotos desta pessoa, em ordem de captura
        assert len(indices) >= 3, f"{classes[classe]} tem só {len(indices)} foto(s): são necessárias pelo menos 3"

        # Blocos de até 10 fotos seguidas. Com poucas fotos, os blocos diminuem para formar pelo menos ~7 blocos
        # (senão poderia não sobrar nenhum bloco para o treino)
        tamanho_bloco = min(10, max(1, len(indices) // 7))
        blocos = [indices[i:i + tamanho_bloco] for i in range(0, len(indices), tamanho_bloco)]
        blocos = [blocos[i] for i in gerador.permutation(len(blocos))]    # embaralha a ordem dos blocos

        n_teste = max(1, round(0.15 * len(blocos)))                       # ~15% dos blocos para teste
        n_val = max(1, round(0.15 * len(blocos)))                         # ~15% dos blocos para validação
        teste     += blocos[:n_teste]
        validacao += blocos[n_teste:n_teste + n_val]
        treino    += blocos[n_teste + n_val:]                             # o restante (~70%) para treino

    return np.concatenate(treino), np.concatenate(validacao), np.concatenate(teste)
