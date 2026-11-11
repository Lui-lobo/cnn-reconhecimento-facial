# Desconhecidos: como a rede aprende a ficar em DÚVIDA diante de quem não é do grupo, e como o limiar é escolhido.
# O softmax sempre distribui 100% entre as classes que conhece. Treinada só com os integrantes, a rede dava quase
# 100% de certeza para qualquer rosto novo, e nenhum limiar separava um estranho de um integrante.
# A solução: mostrar no treino rostos de pessoas de fora do grupo (pasta rostos_de_fora/) e cobrar que, para eles,
# a resposta fique dividida por igual entre as classes. Assim a confiança só fica alta para quem é do grupo,
# e o limiar de confiança passa a funcionar. Tudo com NumPy; a rede continua sendo treinada do zero.
import numpy as np

from src.config import LIMIARES_POSSIVEIS


def alvos(y, n_classes):
    # Transforma os rótulos (0, 1, 2...) no alvo do treino: uma probabilidade por classe.
    # Integrante ou "ninguém": 100% na classe certa. Rosto de fora (rótulo = n_classes): dividido por igual.
    alvo = np.full((len(y), n_classes), 1.0 / n_classes, dtype=np.float32)   # começa tudo "em dúvida"
    com_dono = y < n_classes                                                 # fotos que têm uma classe certa
    alvo[com_dono] = np.eye(n_classes, dtype=np.float32)[y[com_dono]]        # 1 na classe certa, 0 nas outras
    return alvo


def confianca_como_membro(probs, indice_ninguem):
    # A maior probabilidade que a rede deu a um INTEGRANTE (a coluna de "ninguém" não conta) em cada foto
    return np.delete(probs, indice_ninguem, axis=1).max(axis=1)


def escolher_limiar(conf_membros, conf_fora):
    # Escolhe o limiar que menos erra nas fotos de validação: integrante rejeitado (confiança abaixo do limiar)
    # ou rosto de fora aceito (confiança igual ou acima). Em geral vários limiares empatam; ficamos com o do meio,
    # que deixa folga para os dois lados (integrante com luz ruim, estranho parecido com alguém do grupo).
    erros = [np.mean(conf_membros < limiar) + np.mean(conf_fora >= limiar) for limiar in LIMIARES_POSSIVEIS]
    empatados = [limiar for limiar, erro in zip(LIMIARES_POSSIVEIS, erros) if erro <= min(erros) + 1e-9]
    return empatados[len(empatados) // 2]


def avaliar_desconhecidos(model, X_val, y_val, X_fora_val, X_teste, y_teste, X_fora_teste, indice_ninguem):
    # Escolhe o limiar na VALIDAÇÃO e mede no TESTE (pessoas de fora que a rede nunca viu) quantos integrantes
    # continuam aceitos e quantos rostos de fora são rejeitados. Devolve o resumo e as confianças do teste (para o gráfico).
    def confiancas(X, y, X_fora):
        membros = y != indice_ninguem
        probs = model.predict(X[membros], verbose=0)
        conf_membros = probs[np.arange(membros.sum()), y[membros]]          # confiança no integrante CERTO
        conf_fora = confianca_como_membro(model.predict(X_fora, verbose=0), indice_ninguem)
        return conf_membros, conf_fora

    limiar = escolher_limiar(*confiancas(X_val, y_val, X_fora_val))
    conf_membros, conf_fora = confiancas(X_teste, y_teste, X_fora_teste)
    resumo = {
        "limiar": float(limiar),
        "membros_aceitos": float(np.mean(conf_membros >= limiar)),          # quanto maior, melhor
        "fora_rejeitados": float(np.mean(conf_fora < limiar)),              # quanto maior, melhor
        "confianca_membros": float(np.median(conf_membros)),                # confiança típica para um integrante
        "confianca_fora": float(np.median(conf_fora)),                      # e para um rosto de fora
    }
    return resumo, conf_membros, conf_fora
