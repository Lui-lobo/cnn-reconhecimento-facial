# Testes da classe "ninguém" e do teste de troca de fundo.
# Rodar na raiz do repositório com:  pytest
import numpy as np

from src.config import IMG_SIZE
from src.preprocessamento import caixa_do_rosto
from src.ninguem import recorte_de_fundo, adicionar_ninguem
from src.robustez import trocar_fundo


def test_recorte_de_fundo_fica_fora_da_caixa_do_rosto():
    # Marca a caixa do rosto com 1 e o resto com 0: o recorte de fundo não pode conter nenhum pixel da caixa
    gerador = np.random.default_rng(0)
    for altura, largura in [(480, 640), (3088, 2316), (100, 100)]:
        img = np.zeros((altura, largura, 1))
        topo, esquerda, lado = caixa_do_rosto(altura, largura)
        img[topo:topo + lado, esquerda:esquerda + lado] = 1
        for _ in range(20):
            recorte = recorte_de_fundo(img, gerador)
            assert recorte.shape[0] == recorte.shape[1] > 0
            assert recorte.max() == 0


def test_adicionar_ninguem_usa_a_media_de_fotos_por_integrante():
    # 2 integrantes com 4 e 6 fotos -> entram 5 exemplos de "ninguém" (classe 2)
    y = np.array([0] * 4 + [1] * 6)
    X = np.zeros((10, IMG_SIZE, IMG_SIZE, 1), np.float32)
    fundos = np.ones((10, IMG_SIZE, IMG_SIZE, 1), np.float32)
    X_novo, y_novo = adicionar_ninguem(X, y, fundos, 2, 3, np.random.default_rng(0))
    assert len(X_novo) == len(y_novo) == 15
    assert (y_novo == 2).sum() == 5 and (X_novo[y_novo == 2] == 1).all()


def test_trocar_fundo_usa_o_rosto_de_uma_foto_e_o_entorno_da_outra():
    rosto = np.ones((3, IMG_SIZE, IMG_SIZE, 1), np.float32)      # foto toda branca
    entorno = np.zeros((3, IMG_SIZE, IMG_SIZE, 1), np.float32)   # foto toda preta
    montada = trocar_fundo(rosto, entorno)
    assert montada.shape == rosto.shape
    assert (montada[:, IMG_SIZE // 2, IMG_SIZE // 2] == 1).all()   # centro: vem do rosto
    assert (montada[:, 0, 0] == 0).all()                           # canto: vem do entorno
