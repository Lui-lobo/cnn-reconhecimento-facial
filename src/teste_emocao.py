# Teste do modelo pronto de emoções (FER+) em fotos públicas com a emoção já rotulada: emocoes_de_teste/.
# Serve para documentar que o FER+ reconhece cada uma das 8 emoções, inclusive as que o grupo não consegue encenar
# na webcam. São 20 fotos por emoção, sorteadas do conjunto de TESTE do FER2013/FER+ (fotos que o FER+ não viu no
# treino); detalhes em emocoes_de_teste/LEIAME.md. Nada aqui treina ou altera o FER+, e a nossa CNN não participa.
# Uso: python -m src.teste_emocao   (imprime a tabela e grava as figuras em web/img/emocao/, usadas no site e no README)
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # desenha direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt
from PIL import Image

from src.config import EMOCOES_TESTE_DIR, IMAGENS_EMOCAO_DIR, TAMANHO_EMOCAO
from src.emocao import EMOCOES, carregar_modelo_emocao, softmax
from src.metricas import matriz_confusao, calcular_metricas
from src.reconhecimento import ampliar_bilinear

# Uma pasta por emoção, na mesma ordem das saídas do FER+ (PASTAS[k] guarda as fotos da emoção EMOCOES[k])
PASTAS = ["neutro", "alegre", "surpreso", "triste", "bravo", "nojo", "medo", "desprezo"]


def carregar_fotos(pasta=EMOCOES_TESTE_DIR):
    # Lê as fotos (48 x 48, em cinza) e devolve (fotos, rótulos); o rótulo é a posição da pasta em PASTAS
    fotos, rotulos = [], []
    for k, nome in enumerate(PASTAS):
        for arquivo in sorted(os.listdir(os.path.join(pasta, nome))):
            fotos.append(np.array(Image.open(os.path.join(pasta, nome, arquivo)).convert("L"), dtype=np.float64))
            rotulos.append(k)
    return fotos, np.array(rotulos)


def entrada_do_teste(foto):
    # Estas fotos já vêm recortadas no rosto (é o formato do FER2013), então não passam pelo nosso recorte:
    # só ampliamos de 48 x 48 para os 64 x 64 que o FER+ espera, mantendo os tons de 0 a 255
    rosto = ampliar_bilinear(foto, TAMANHO_EMOCAO)
    return rosto.reshape(1, 1, TAMANHO_EMOCAO, TAMANHO_EMOCAO).astype(np.float32)   # (lote, canais, altura, largura)


def prever(sessao, fotos):
    # Probabilidades das 8 emoções para cada foto (o FER+ aceita um rosto por vez)
    nome_entrada = sessao.get_inputs()[0].name
    return softmax(np.array([sessao.run(None, {nome_entrada: entrada_do_teste(f)})[0][0] for f in fotos]))


def desenhar_exemplos(fotos, rotulos, probs, por_emocao=3):
    # Para cada emoção, as fotos em que o FER+ deu a maior probabilidade para a emoção certa.
    # Título verde = acertou; vermelho = a mais provável foi outra emoção (mostramos mesmo assim).
    previsto = probs.argmax(axis=1)
    fig, eixos = plt.subplots(por_emocao, len(PASTAS), figsize=(2.0 * len(PASTAS), 2.3 * por_emocao))
    for k in range(len(PASTAS)):
        da_emocao = np.flatnonzero(rotulos == k)
        melhores = da_emocao[np.argsort(-probs[da_emocao, k])][:por_emocao]
        for linha, i in enumerate(melhores):
            eixo = eixos[linha, k]
            eixo.imshow(fotos[i], cmap="gray", vmin=0, vmax=255)
            eixo.set_title(f"{EMOCOES[previsto[i]]}\n{probs[i, previsto[i]]:.0%}", fontsize=9,
                           color="#2ca02c" if previsto[i] == k else "#d62728")
            eixo.axis("off")
        eixos[0, k].text(0.5, 1.45, EMOCOES[k], transform=eixos[0, k].transAxes, ha="center", fontsize=11,
                         fontweight="bold")
    fig.suptitle("Rótulo real (em negrito) e resposta do FER+ em cada foto", y=1.02)
    fig.savefig(os.path.join(IMAGENS_EMOCAO_DIR, "exemplos.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)


def salvar_miniaturas(fotos, rotulos, probs, tamanho=144):
    # Uma foto de cada emoção para a tabela "sinais típicos" do site: a que recebeu do FER+ a maior probabilidade
    # para a emoção certa, ampliada para ficar legível
    for k, nome in enumerate(PASTAS):
        da_emocao = np.flatnonzero(rotulos == k)
        melhor = da_emocao[np.argmax(probs[da_emocao, k])]
        ampliada = np.clip(ampliar_bilinear(fotos[melhor], tamanho), 0, 255).astype(np.uint8)
        Image.fromarray(ampliada).save(os.path.join(IMAGENS_EMOCAO_DIR, f"exemplo_{nome}.png"))


def desenhar_matriz(matriz):
    # Matriz de confusão: linhas = emoção real, colunas = resposta do FER+
    n = len(EMOCOES)
    fracao = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    fig = plt.figure(figsize=(8.5, 7.5))
    plt.imshow(fracao, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if matriz[i, j]:
                plt.text(j, i, str(matriz[i, j]), ha="center", va="center", color="white" if fracao[i, j] > 0.5 else "black")
    plt.xticks(range(n), EMOCOES, rotation=35, ha="right")
    plt.yticks(range(n), EMOCOES)
    plt.xlabel("resposta do FER+")
    plt.ylabel("emoção real (votos de 10 pessoas)")
    plt.title("FER+ nas fotos de emocoes_de_teste/")
    fig.savefig(os.path.join(IMAGENS_EMOCAO_DIR, "matriz.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    fotos, rotulos = carregar_fotos()
    probs = prever(carregar_modelo_emocao(), fotos)
    matriz = matriz_confusao(rotulos, probs.argmax(axis=1), len(EMOCOES))
    m = calcular_metricas(matriz)

    print(f"{len(fotos)} fotos, acurácia {m['acuracia']:.1%}\n")
    print("| Emoção | Fotos | Acertos | Recall | Precisão | Erro mais comum |")
    print("|---|---|---|---|---|---|")
    for k, nome in enumerate(EMOCOES):
        erros = matriz[k].copy()
        erros[k] = 0                                              # sobra só o que foi confundido com outra emoção
        erro = f"{EMOCOES[erros.argmax()]} ({erros.max()})" if erros.max() else "—"
        print(f"| {nome} | {matriz[k].sum()} | {m['VP'][k]} | {m['recall'][k]:.0%} | {m['precisao'][k]:.0%} | {erro} |")

    os.makedirs(IMAGENS_EMOCAO_DIR, exist_ok=True)
    desenhar_exemplos(fotos, rotulos, probs)
    salvar_miniaturas(fotos, rotulos, probs)
    desenhar_matriz(matriz)
    print("\nImagens gravadas em", IMAGENS_EMOCAO_DIR)


if __name__ == "__main__":
    main()
