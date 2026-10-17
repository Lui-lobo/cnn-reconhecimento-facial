# Script principal: executa o pipeline completo, na mesma ordem do notebook.
# Leitura -> pré-processamento -> divisão -> aumento -> CNN -> treino -> avaliação -> imagens intermediárias.
# Uso (na raiz do repositório):  python -m src.treinar
import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (DATASET_DIR, MODELO_DIR, CAMINHO_MODELO, CAMINHO_CLASSES, CAMINHO_RESULTADOS,
                        EPOCAS, BATCH, SEED, CLASSE_NINGUEM, FRACAO_DISTANCIA)
from src.dados import ler_imagem, listar_dataset
from src.preprocessamento import preprocessar
from src.divisao import dividir_dados
from src.ninguem import recorte_de_fundo, adicionar_ninguem
from src.robustez import teste_troca_de_fundo
from src.aumento import montar_treino_aumentado
from src.modelo import construir_cnn
from src.metricas import matriz_confusao, calcular_metricas, acuracia
from src import visualizacoes


def treinar_rede(model, X_treino_final, y_treino_final, X_val, y_val):
    # Treina a CNN. O EarlyStopping observa o erro na validação: se ele não melhorar por 10 épocas
    # seguidas, o treino para e a rede volta aos pesos da melhor época.
    parada_antecipada = keras.callbacks.EarlyStopping(
        monitor="val_loss",            # observa o erro na validação
        patience=10,                   # tolera 10 épocas sem melhora
        restore_best_weights=True,     # no fim, volta para os pesos da melhor época
    )
    historico = model.fit(
        X_treino_final, y_treino_final,       # dados de treino (com aumento)
        validation_data=(X_val, y_val),       # dados de validação (fotos reais)
        epochs=EPOCAS,
        batch_size=BATCH,
        callbacks=[parada_antecipada],
        verbose=2,                            # imprime uma linha por época
    )
    return {chave: [float(v) for v in valores] for chave, valores in historico.history.items()}


def resumo_camadas(model):
    # Lista cada camada com o formato da saída e o número de parâmetros (o mesmo que o model.summary() mostra)
    camadas = []
    for camada in model.layers:
        formato = list(camada.output.shape[1:])                 # tira o primeiro eixo (tamanho do lote)
        camadas.append({"nome": camada.name, "tipo": camada.__class__.__name__,
                        "saida": formato, "parametros": int(camada.count_params())})
    return camadas


def main():
    np.random.seed(SEED)          # mesma semente do notebook: o resultado se repete a cada execução
    tf.random.set_seed(SEED)

    # As linhas "Etapa N/8" mostram o andamento no terminal e na barra de progresso do site
    print("Etapa 1/8: leitura das fotos")
    classes, caminhos, rotulos = listar_dataset(DATASET_DIR)
    integrantes = [nome for nome in classes if nome != CLASSE_NINGUEM]
    assert len(integrantes) >= 2, f"São necessários pelo menos 2 integrantes com fotos em {DATASET_DIR}."
    print("Integrantes:")
    for numero, nome in enumerate(classes):
        print(f"  {numero} = {nome}: {np.sum(rotulos == numero)} fotos")

    # Pré-processamento de todas as fotos (mesma função usada no reconhecimento ao vivo)
    # De cada foto sai também um recorte do fundo (ao lado do rosto), usado como exemplo da classe "ninguém",
    # e uma versão "de outra distância" (quadrado maior ou menor), usada no aumento de dados
    print("Etapa 2/8: pré-processamento")
    gerador = np.random.default_rng(SEED)
    X, fundos, X_distancia = [], [], []
    for caminho in caminhos:
        img = ler_imagem(caminho)
        X.append(preprocessar(img))
        fundos.append(preprocessar(recorte_de_fundo(img, gerador)))
        X_distancia.append(preprocessar(img, fracao=gerador.uniform(*FRACAO_DISTANCIA)))
    X, fundos, X_distancia, y = np.array(X), np.array(fundos), np.array(X_distancia), rotulos
    print("X:", X.shape, "-> (quantidade de fotos, altura, largura, canais)")

    # Divisão em treino, validação e teste por blocos de fotos seguidas
    print("Etapa 3/8: divisão em treino, validação e teste")
    idx_treino, idx_val, idx_teste = dividir_dados(y, classes)

    # A classe "ninguém" entra no fim da lista se não houver a pasta dataset/ninguem/.
    # Cada conjunto recebe recortes de fundo das SUAS fotos: nenhum fundo do teste aparece no treino.
    pastas = classes                                   # as classes que têm pasta (e fotos) em dataset/
    if CLASSE_NINGUEM not in classes:
        classes = classes + [CLASSE_NINGUEM]
    ninguem = classes.index(CLASSE_NINGUEM)
    X_treino, y_treino = adicionar_ninguem(X[idx_treino], y[idx_treino], fundos[idx_treino], ninguem, len(classes), gerador)
    X_val,    y_val    = adicionar_ninguem(X[idx_val],    y[idx_val],    fundos[idx_val],    ninguem, len(classes), gerador)
    X_teste,  y_teste  = adicionar_ninguem(X[idx_teste],  y[idx_teste],  fundos[idx_teste],  ninguem, len(classes), gerador)

    # Imagens das fotos originais e das etapas do pré-processamento
    visualizacoes.fotos_originais(pastas, caminhos, rotulos)
    visualizacoes.etapas_preprocessamento(pastas, caminhos, rotulos)

    # Aumento de dados (só no treino)
    # Os exemplos de "ninguém" gerados (no fim de X_treino) não têm versão de outra distância: usam o próprio recorte
    print("Etapa 4/8: aumento de dados")
    X_distancia_treino = np.concatenate([X_distancia[idx_treino], X_treino[len(idx_treino):]])
    visualizacoes.exemplos_aumento(classes, X_treino, X_distancia_treino, y_treino, ninguem)
    X_treino_final, y_treino_final = montar_treino_aumentado(X_treino, X_distancia_treino, y_treino, ninguem)
    print(f"Treino: {len(X_treino)} fotos originais -> {len(X_treino_final)} com o aumento de dados")

    # Montagem da CNN e treinamento
    print("Etapa 5/8: montagem da CNN")
    model = construir_cnn(len(classes))
    model.summary()
    print(f"Etapa 6/8: treinamento (até {EPOCAS} épocas)")
    historico = treinar_rede(model, X_treino_final, y_treino_final, X_val, y_val)

    # Avaliação no conjunto de teste
    print("Etapa 7/8: avaliação no conjunto de teste")
    y_previsto = model.predict(X_teste, verbose=0).argmax(axis=1)
    matriz = matriz_confusao(y_teste, y_previsto, len(classes))
    m = calcular_metricas(matriz)
    acuracias = {nome: acuracia(yc, model.predict(Xc, verbose=0).argmax(axis=1))
                 for nome, Xc, yc in [("treino", X_treino, y_treino), ("validacao", X_val, y_val),
                                      ("teste", X_teste, y_teste)]}
    troca = teste_troca_de_fundo(model, X_teste, y_teste, ninguem, gerador)

    # Imagens intermediárias da avaliação e da rede
    print("Etapa 8/8: imagens intermediárias e gravação do modelo")
    visualizacoes.curvas_treino(historico)
    visualizacoes.desenhar_matriz(matriz, classes)
    visualizacoes.barras_metricas(m, classes)
    visualizacoes.fotos_erradas(X_teste, y_teste, y_previsto, classes)
    visualizacoes.mapas_caracteristicas(model, X_teste[0], classes[y_teste[0]])
    visualizacoes.gradcam_por_integrante(model, X_teste, y_teste, classes)
    visualizacoes.grafico_troca_de_fundo(troca, X_teste, y_teste, classes)

    # Salva os nomes dos integrantes, o modelo e todos os resultados (lidos pelo site).
    # Os nomes vão ANTES do modelo: o servidor recarrega o modelo quando o arquivo .keras muda,
    # e nessa hora os nomes novos já precisam estar prontos.
    os.makedirs(MODELO_DIR, exist_ok=True)
    with open(CAMINHO_CLASSES, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)
    model.save(CAMINHO_MODELO)

    contar = lambda yc: {nome: int(np.sum(yc == c)) for c, nome in enumerate(classes)}   # fotos por integrante
    resultados = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "fotos_por_conjunto": {"treino": contar(y_treino), "validacao": contar(y_val), "teste": contar(y_teste)},
        "totais": {"fotos": len(y), "treino": len(y_treino),
                   "treino_com_aumento": len(y_treino_final),
                   "validacao": len(y_val), "teste": len(y_teste)},
        "historico": historico,
        "epocas_treinadas": len(historico["loss"]),
        "melhor_epoca": int(np.argmin(historico["val_loss"])) + 1,
        "matriz_confusao": matriz.tolist(),
        "metricas": {chave: (valor.tolist() if isinstance(valor, np.ndarray) else float(valor))
                     for chave, valor in m.items()},
        "acuracias": acuracias,
        "troca_de_fundo": troca,
        "camadas": resumo_camadas(model),
        "parametros_total": int(model.count_params()),
    }
    with open(CAMINHO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Resumo no terminal
    print(f"\n{'integrante':>15} | precisão | recall |   F1  | especificidade | fotos no teste")
    for i, nome in enumerate(classes):
        print(f"{nome:>15} |   {m['precisao'][i]:.3f}  |  {m['recall'][i]:.3f} | {m['f1'][i]:.3f} |"
              f"     {m['especificidade'][i]:.3f}      | {matriz[i].sum()}")
    print(f"{'média macro':>15} |   {m['precisao'].mean():.3f}  |  {m['recall'].mean():.3f} |"
          f" {m['f1'].mean():.3f} |     {m['especificidade'].mean():.3f}")
    for nome, acc in acuracias.items():
        print(f"Acurácia em {nome}: {acc:.3f}")
    print(f"\nTroca de fundo: a rede seguiu o rosto em {troca['segue_o_rosto']:.1%} e o entorno em "
          f"{troca['segue_o_entorno']:.1%} (quanto menor, melhor). Fotos vazias -> ninguém: {troca['vazio_e_ninguem']:.1%}")
    print(f"\nModelo salvo em {CAMINHO_MODELO}")
    print(f"Resultados em {CAMINHO_RESULTADOS} e imagens em web/img/resultados/")


if __name__ == "__main__":
    main()
