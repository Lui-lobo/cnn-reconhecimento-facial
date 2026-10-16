# Testes do pré-processamento: formato e faixa da saída, equalização sem NaN e recorte quadrado.
# Rodar na raiz do repositório com:  pytest
import numpy as np
import pytest

from src.config import IMG_SIZE
from src.preprocessamento import preprocessar, equalizar_histograma, recorte_central


@pytest.mark.parametrize("altura, largura", [(480, 640), (640, 480), (30, 50), (2000, 3000), (96, 96)])
def test_preprocessar_devolve_96x96x1_float32_entre_0_e_1(altura, largura):
    # Imagens retangulares, pequenas e grandes devem sair todas no formato que a CNN espera
    img = np.random.default_rng(0).integers(0, 256, (altura, largura, 3)).astype(np.uint8)
    saida = preprocessar(img)
    assert saida.shape == (IMG_SIZE, IMG_SIZE, 1)
    assert saida.dtype == np.float32
    assert saida.min() >= 0 and saida.max() <= 1


def test_preprocessar_devolve_as_etapas():
    # Com mostrar_etapas=True, vêm também [original, recorte, cinza, 96x96, equalizada]
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, etapas = preprocessar(img, mostrar_etapas=True)
    assert len(etapas) == 5
    assert etapas[1].shape == (480, 480, 3)
    assert etapas[3].shape == (IMG_SIZE, IMG_SIZE)


def test_equalizar_imagem_constante_nao_gera_nan():
    # Em uma imagem de uma cor só, max - min do acumulado é 0: o max(..., 1) evita a divisão por zero
    resultado = equalizar_histograma(np.full((IMG_SIZE, IMG_SIZE), 128.0))
    assert not np.isnan(resultado).any()


@pytest.mark.parametrize("altura, largura", [(480, 640), (640, 480), (100, 100), (7, 3)])
def test_recorte_central_devolve_quadrado(altura, largura):
    recorte = recorte_central(np.zeros((altura, largura, 3)))
    assert recorte.shape[0] == recorte.shape[1] == min(altura, largura)
