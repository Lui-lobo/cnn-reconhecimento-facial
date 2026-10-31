# Emoção do rosto (alegre, triste, bravo, com medo...) com um modelo PRONTO: o FER+ da Microsoft, em formato ONNX.
# É a única parte do projeto com modelo pré-treinado (o trabalho permite para a emoção). Quem é a pessoa, quem é
# "desconhecido" e quando não há "ninguém" continua sendo decidido só pela nossa CNN.
# O FER+ não acha o rosto sozinho: ele recebe um quadrado no MESMO centro do oval do nosso pré-processamento,
# só que mais fechado no rosto (FRACAO_EMOCAO). A entrada da nossa CNN não muda.
# Uso: python -m src.emocao   (só baixa o modelo; o servidor também baixa sozinho no primeiro uso)
import os
import urllib.request
import numpy as np

from src.config import CAMINHO_EMOCAO, URL_EMOCAO, TAMANHO_EMOCAO, FRACAO_EMOCAO
from src.preprocessamento import recorte_rosto, para_cinza, redimensionar

# As 8 saídas do FER+, na ordem dele (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)
EMOCOES = ["neutro(a)", "alegre", "surpreso(a)", "triste", "bravo(a)", "com nojo", "com medo", "com desprezo"]


def baixar_modelo_emocao():
    # Baixa o arquivo do FER+ (~35 MB) para modelo/ se ele ainda não estiver lá
    if not os.path.exists(CAMINHO_EMOCAO):
        os.makedirs(os.path.dirname(CAMINHO_EMOCAO), exist_ok=True)
        print("Baixando o modelo de emoções (FER+, ~35 MB)...")
        urllib.request.urlretrieve(URL_EMOCAO, CAMINHO_EMOCAO + ".parcial")   # só renomeia se terminar
        os.replace(CAMINHO_EMOCAO + ".parcial", CAMINHO_EMOCAO)
    return CAMINHO_EMOCAO


def carregar_modelo_emocao():
    # Abre o FER+ com o onnxruntime (biblioteca que só executa o modelo pronto, sem treinar nada)
    import onnxruntime
    return onnxruntime.InferenceSession(baixar_modelo_emocao(), providers=["CPUExecutionProvider"])


def entrada_emocao(img):
    # O formato que o FER+ espera: o rosto em cinza, 64 x 64, com tons de 0 a 255 (sem equalização e sem máscara,
    # porque o FER+ foi treinado com rostos inteiros "normais"). Reaproveita as nossas etapas 1 a 3, mas com um
    # quadrado mais fechado (da testa ao queixo, como as fotos do FER+); o centro é o mesmo do oval.
    rosto = redimensionar(para_cinza(recorte_rosto(img, FRACAO_EMOCAO)), TAMANHO_EMOCAO)
    return rosto.reshape(1, TAMANHO_EMOCAO, TAMANHO_EMOCAO).astype(np.float32)   # (canais, altura, largura)


def softmax(pontuacoes):
    # Transforma as pontuações do FER+ em probabilidades que somam 1 (o modelo não tem softmax no fim)
    e = np.exp(pontuacoes - pontuacoes.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def classificar_emocoes(sessao, rostos):
    # Recebe uma lista de imagens (uma por faixa da tela) e devolve as probabilidades das 8 emoções de cada uma
    # O FER+ só aceita 1 rosto por vez (lote de tamanho 1), então rodamos uma faixa de cada vez
    nome_entrada = sessao.get_inputs()[0].name
    pontuacoes = [sessao.run(None, {nome_entrada: entrada_emocao(r)[np.newaxis]})[0][0] for r in rostos]   # 8 cada
    return softmax(np.array(pontuacoes))


if __name__ == "__main__":
    print("Modelo de emoções em", baixar_modelo_emocao())
