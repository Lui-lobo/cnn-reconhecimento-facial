# Gera as imagens intermediárias (PNG) que aparecem na documentação do site, em web/img/resultados/.
# São os mesmos gráficos do notebook, mas salvos em arquivo em vez de exibidos na tela.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # desenha direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt
from tensorflow import keras

from src.config import IMAGENS_DIR
from src.dados import ler_imagem
from src.preprocessamento import preprocessar
from src.aumento import aumentar
from src.reconhecimento import criar_modelo_gradcam, grad_cam


def salvar(nome):
    # Salva a figura atual em web/img/resultados/<nome>.png e fecha a figura (libera memória)
    os.makedirs(IMAGENS_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGENS_DIR, nome + ".png"), dpi=90, bbox_inches="tight")
    plt.close("all")


def fotos_originais(classes, caminhos, rotulos):
    # 6 fotos originais de cada integrante, sem nenhum tratamento
    fig, eixos = plt.subplots(len(classes), 6, figsize=(12, 2.2 * len(classes)), squeeze=False)
    for linha, nome in enumerate(classes):
        fotos_da_pessoa = [c for c, r in zip(caminhos, rotulos) if r == linha]
        for coluna in range(6):
            ax = eixos[linha, coluna]
            ax.axis("off")
            if coluna < len(fotos_da_pessoa):
                ax.imshow(ler_imagem(fotos_da_pessoa[coluna * len(fotos_da_pessoa) // 6]))
        eixos[linha, 0].set_title(nome, loc="left")
    plt.suptitle("Fotos originais")
    plt.tight_layout()
    salvar("fotos_originais")


def etapas_preprocessamento(classes, caminhos, rotulos):
    # As 5 etapas do pré-processamento para a primeira foto de cada integrante, com o histograma antes/depois
    titulos = ["original", "1. recorte", "2. cinza", "3. 96x96", "4. equalizada"]
    fig, eixos = plt.subplots(len(classes), 6, figsize=(17, 3 * len(classes)), squeeze=False)
    for numero, nome in enumerate(classes):
        primeira_foto = caminhos[list(rotulos).index(numero)]
        _, etapas = preprocessar(ler_imagem(primeira_foto), mostrar_etapas=True)
        for ax, img, titulo in zip(eixos[numero], etapas, titulos):
            ax.imshow(img.astype(np.uint8), cmap="gray")
            ax.set_title(f"{nome}: {titulo}\n{img.shape[1]}x{img.shape[0]}", fontsize=9)
            ax.axis("off")
        # Histograma: quantos pixels há de cada tom antes e depois da equalização
        eixos[numero, 5].hist(etapas[3].ravel(), bins=50, alpha=0.6, label="antes")
        eixos[numero, 5].hist(etapas[4].ravel(), bins=50, alpha=0.6, label="equalizada")
        eixos[numero, 5].set_title("histograma dos tons", fontsize=9)
        eixos[numero, 5].legend(fontsize=8)
    plt.tight_layout()
    salvar("preprocessamento")


def exemplos_aumento(classes, X_treino, y_treino):
    # A mesma foto de treino modificada 6 vezes pelo aumento de dados
    fig, eixos = plt.subplots(len(classes), 7, figsize=(13, 2 * len(classes)), squeeze=False)
    for linha in range(len(classes)):
        original = X_treino[y_treino == linha][0]
        eixos[linha, 0].imshow(original[:, :, 0], cmap="gray", vmin=0, vmax=1)
        eixos[linha, 0].set_title(f"{classes[linha]} (original)", fontsize=9)
        for coluna in range(1, 7):
            eixos[linha, coluna].imshow(aumentar(original)[:, :, 0], cmap="gray", vmin=0, vmax=1)
            eixos[linha, coluna].set_title(f"aumentada {coluna}", fontsize=9)
    for ax in eixos.ravel():
        ax.axis("off")
    plt.tight_layout()
    salvar("aumento")


def curvas_treino(historico):
    # Curvas de aprendizado: erro e acurácia a cada época, no treino e na validação
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    eixos[0].plot(historico["loss"], label="treino")
    eixos[0].plot(historico["val_loss"], label="validação")
    eixos[0].set_title("Erro (entropia cruzada)")
    eixos[1].plot(historico["accuracy"], label="treino")
    eixos[1].plot(historico["val_accuracy"], label="validação")
    eixos[1].set_title("Acurácia")
    for ax in eixos:
        ax.set_xlabel("época")
        ax.legend()
        ax.grid(alpha=0.3)
    salvar("curvas")


def desenhar_matriz(matriz, classes):
    # Matriz de confusão: quantidade e porcentagem da linha em cada célula
    n = len(classes)
    porcentagem = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    tamanho = 2 + 1.2 * n
    plt.figure(figsize=(tamanho, tamanho))
    plt.imshow(porcentagem, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            cor = "white" if porcentagem[i, j] > 0.5 else "black"
            plt.text(j, i, f"{matriz[i, j]}\n({porcentagem[i, j]:.0%})", ha="center", va="center", color=cor)
    plt.xticks(range(n), classes, rotation=30)
    plt.yticks(range(n), classes)
    plt.xlabel("previsto")
    plt.ylabel("real")
    plt.title("Matriz de confusão (teste)")
    plt.colorbar(fraction=0.046)
    salvar("matriz_confusao")


def barras_metricas(m, classes):
    # Precisão, recall, F1 e especificidade de cada integrante em um gráfico de barras
    posicoes = np.arange(len(classes))
    plt.figure(figsize=(max(6, 2 * len(classes)), 4))
    plt.bar(posicoes - 0.3, m["precisao"], 0.2, label="Precisão")
    plt.bar(posicoes - 0.1, m["recall"], 0.2, label="Recall")
    plt.bar(posicoes + 0.1, m["f1"], 0.2, label="F1")
    plt.bar(posicoes + 0.3, m["especificidade"], 0.2, label="Especificidade")
    plt.xticks(posicoes, classes)
    plt.ylim(0, 1.1)
    plt.legend(ncol=4, loc="upper center")
    plt.title("Métricas por integrante (teste)")
    salvar("metricas")


def fotos_erradas(X_teste, y_teste, y_previsto, classes):
    # Até 8 fotos do teste que a rede errou, com o nome real e o previsto
    erros = np.where(y_previsto != y_teste)[0]
    if len(erros) == 0:
        plt.figure(figsize=(6, 1.2))
        plt.text(0.5, 0.5, "A rede não errou nenhuma foto do teste.", ha="center", va="center", fontsize=13)
        plt.axis("off")
    else:
        n = min(len(erros), 8)
        fig, eixos = plt.subplots(1, n, figsize=(2.3 * n, 2.8), squeeze=False)
        for ax, i in zip(eixos[0], erros[:n]):
            ax.imshow(X_teste[i][:, :, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"real: {classes[y_teste[i]]}\nprevisto: {classes[y_previsto[i]]}", fontsize=9)
            ax.axis("off")
    salvar("erros")


def mapas_caracteristicas(model, foto, titulo):
    # Saída das 4 camadas convolucionais para uma foto: os 8 primeiros mapas de cada camada.
    # Um "modelo espião" devolve a saída das camadas intermediárias.
    camadas = ["conv1", "conv2", "conv3", "conv4"]
    espiao = keras.Model(model.input, [model.get_layer(nome).output for nome in camadas])
    saidas = espiao.predict(foto[np.newaxis], verbose=0)    # np.newaxis: a rede espera um lote de fotos

    fig, eixos = plt.subplots(len(camadas), 9, figsize=(15.5, 7.5))
    for linha, (nome, mapas) in enumerate(zip(camadas, saidas)):
        eixos[linha, 0].axis("off")
        for coluna in range(8):
            eixos[linha, coluna + 1].imshow(mapas[0, :, :, coluna], cmap="viridis")
            eixos[linha, coluna + 1].axis("off")
        eixos[linha, 1].set_title(f"{nome}: {mapas.shape[1]}x{mapas.shape[2]}, {mapas.shape[3]} mapas",
                                  loc="left", fontsize=10)
    eixos[0, 0].imshow(foto[:, :, 0], cmap="gray")             # a foto de entrada fica no canto
    eixos[0, 0].set_title(f"entrada: {titulo}", fontsize=10)
    plt.tight_layout()
    salvar("mapas")


def gradcam_por_integrante(model, X_teste, y_teste, classes):
    # Uma foto de teste de cada integrante: imagem em cima, mapa de calor do Grad-CAM embaixo
    modelo_gradcam = criar_modelo_gradcam(model)
    n = len(classes)
    fig, eixos = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for c in range(n):
        img = X_teste[y_teste == c][0]
        calor, previsto = grad_cam(modelo_gradcam, img)
        eixos[0, c].imshow(img[:, :, 0], cmap="gray")
        eixos[0, c].set_title(f"real: {classes[c]}")
        eixos[1, c].imshow(img[:, :, 0], cmap="gray")
        eixos[1, c].imshow(calor, cmap="jet", alpha=0.45, interpolation="bilinear")   # calor por cima da foto
        eixos[1, c].set_title(f"previsto: {classes[previsto]}")
    for ax in eixos.ravel():
        ax.axis("off")
    plt.tight_layout()
    salvar("gradcam")
