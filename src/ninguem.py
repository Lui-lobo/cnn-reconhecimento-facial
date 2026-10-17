# Classe "ninguém": exemplos de fundo vazio, para a rede poder responder que não há ninguém na caixa.
# Sem ela, o softmax é obrigado a escolher um integrante até para uma cadeira vazia.
# Os exemplos vêm de duas fontes: fotos tiradas com a câmera vazia (pasta dataset/ninguem/, opcional) e
# recortes do fundo das próprias fotos dos integrantes (gerados aqui, sem precisar coletar nada).
import numpy as np

from src.preprocessamento import caixa_do_rosto


def recorte_de_fundo(img, gerador):
    # Recorta um quadrado ao lado da caixa do rosto (à esquerda ou à direita, sorteado), na metade de cima da foto:
    # é só fundo, porque fica inteiro fora da região onde a pessoa está.
    altura, largura = img.shape[:2]
    _, esquerda, _ = caixa_do_rosto(altura, largura)
    lado = max(1, min(esquerda, altura))                                     # o espaço que sobra ao lado do rosto
    x = 0 if gerador.random() < 0.5 else largura - lado                      # canto esquerdo ou direito
    y = int(gerador.integers(0, (altura - lado) // 2 + 1))                   # altura sorteada na metade de cima
    return img[y:y + lado, x:x + lado]


def adicionar_ninguem(X, y, fundos, indice_ninguem, n_classes, gerador):
    # Junta a um conjunto (treino, validação ou teste) recortes de fundo das MESMAS fotos desse conjunto,
    # com o rótulo "ninguém". Quantidade: a média de fotos por integrante, para a classe nova não dominar.
    fotos_por_classe = np.bincount(y, minlength=n_classes)
    quantidade = int(round(np.mean(np.delete(fotos_por_classe, indice_ninguem))))   # média só dos integrantes
    escolhidos = gerador.choice(len(fundos), size=min(quantidade, len(fundos)), replace=False)
    X_novo = np.concatenate([X, fundos[escolhidos]])
    y_novo = np.concatenate([y, np.full(len(escolhidos), indice_ninguem)])
    return X_novo, y_novo
