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
                        EPOCAS, BATCH, SEED)
from src.dados import ler_imagem, listar_dataset
from src.preprocessamento import preprocessar
from src.divisao import dividir_dados
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

    # 1. Leitura das fotos
    classes, caminhos, rotulos = listar_dataset(DATASET_DIR)
    assert len(classes) >= 2, f"São necessários pelo menos 2 integrantes com fotos em {DATASET_DIR}."
    print("Integrantes:")
    for numero, nome in enumerate(classes):
        print(f"  {numero} = {nome}: {np.sum(rotulos == numero)} fotos")

    # 2. Pré-processamento de todas as fotos (mesma função usada no reconhecimento ao vivo)
    X = np.array([preprocessar(ler_imagem(c)) for c in caminhos])
    y = rotulos
    print("X:", X.shape, "-> (quantidade de fotos, altura, largura, canais)")

    # 3. Divisão em treino, validação e teste por blocos de fotos seguidas
    idx_treino, idx_val, idx_teste = dividir_dados(y, classes)
    X_treino, y_treino = X[idx_treino], y[idx_treino]
    X_val,    y_val    = X[idx_val],    y[idx_val]
    X_teste,  y_teste  = X[idx_teste],  y[idx_teste]

    # Imagens das fotos originais e das etapas do pré-processamento
    visualizacoes.fotos_originais(classes, caminhos, rotulos)
    visualizacoes.etapas_preprocessamento(classes, caminhos, rotulos)

    # 4. Aumento de dados (só no treino)
    visualizacoes.exemplos_aumento(classes, X_treino, y_treino)
    X_treino_final, y_treino_final = montar_treino_aumentado(X_treino, y_treino)
    print(f"Treino: {len(X_treino)} fotos originais -> {len(X_treino_final)} com o aumento de dados")

    # 5. Montagem da CNN e 6. Treinamento
    model = construir_cnn(len(classes))
    model.summary()
    historico = treinar_rede(model, X_treino_final, y_treino_final, X_val, y_val)

    # 7. Avaliação no conjunto de teste
    y_previsto = model.predict(X_teste, verbose=0).argmax(axis=1)
    matriz = matriz_confusao(y_teste, y_previsto, len(classes))
    m = calcular_metricas(matriz)
    acuracias = {nome: acuracia(yc, model.predict(Xc, verbose=0).argmax(axis=1))
                 for nome, Xc, yc in [("treino", X_treino, y_treino), ("validacao", X_val, y_val),
                                      ("teste", X_teste, y_teste)]}

    # 8. Imagens intermediárias da avaliação e da rede
    visualizacoes.curvas_treino(historico)
    visualizacoes.desenhar_matriz(matriz, classes)
    visualizacoes.barras_metricas(m, classes)
    visualizacoes.fotos_erradas(X_teste, y_teste, y_previsto, classes)
    visualizacoes.mapas_caracteristicas(model, X_teste[0], classes[y_teste[0]])
    visualizacoes.gradcam_por_integrante(model, X_teste, y_teste, classes)

    # Salva o modelo, os nomes dos integrantes e todos os resultados (lidos pelo site)
    os.makedirs(MODELO_DIR, exist_ok=True)
    model.save(CAMINHO_MODELO)
    with open(CAMINHO_CLASSES, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    contar = lambda yc: {nome: int(np.sum(yc == c)) for c, nome in enumerate(classes)}   # fotos por integrante
    resultados = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "fotos_por_conjunto": {"treino": contar(y_treino), "validacao": contar(y_val), "teste": contar(y_teste)},
        "totais": {"fotos": len(y), "treino": len(y_treino), "treino_com_aumento": len(y_treino_final),
                   "validacao": len(y_val), "teste": len(y_teste)},
        "historico": historico,
        "epocas_treinadas": len(historico["loss"]),
        "melhor_epoca": int(np.argmin(historico["val_loss"])) + 1,
        "matriz_confusao": matriz.tolist(),
        "metricas": {chave: (valor.tolist() if isinstance(valor, np.ndarray) else float(valor))
                     for chave, valor in m.items()},
        "acuracias": acuracias,
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
    print(f"\nModelo salvo em {CAMINHO_MODELO}")
    print(f"Resultados em {CAMINHO_RESULTADOS} e imagens em web/img/resultados/")


if __name__ == "__main__":
    main()
