# Testes do pré-processamento: formato e faixa da saída, equalização sem NaN, recorte do rosto e máscara oval.
# Rodar na raiz do repositório com:  pytest
import numpy as np
import pytest

from src.config import IMG_SIZE
from src.config import FRACAO_ROSTO
from src.preprocessamento import (preprocessar, equalizar_histograma, recorte_rosto, caixa_do_rosto,
                                  mascara_oval, aplicar_mascara_oval)


@pytest.mark.parametrize("altura, largura", [(480, 640), (640, 480), (30, 50), (2000, 3000), (96, 96)])
def test_preprocessar_devolve_96x96x1_float32_entre_0_e_1(altura, largura):
    # Imagens retangulares, pequenas e grandes devem sair todas no formato que a CNN espera
    img = np.random.default_rng(0).integers(0, 256, (altura, largura, 3)).astype(np.uint8)
    saida = preprocessar(img)
    assert saida.shape == (IMG_SIZE, IMG_SIZE, 1)
    assert saida.dtype == np.float32
    assert saida.min() >= 0 and saida.max() <= 1


def test_preprocessar_devolve_as_etapas():
    # Com mostrar_etapas=True, vêm também [original, recorte, cinza, 96x96, equalizada, com máscara]
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, etapas = preprocessar(img, mostrar_etapas=True)
    assert len(etapas) == 6
    assert etapas[1].shape == (int(480 * FRACAO_ROSTO),) * 2 + (3,)
    assert etapas[3].shape == etapas[5].shape == (IMG_SIZE, IMG_SIZE)


def test_equalizar_imagem_constante_nao_gera_nan():
    # Em uma imagem de uma cor só, max - min do acumulado é 0: o max(..., 1) evita a divisão por zero
    resultado = equalizar_histograma(np.full((IMG_SIZE, IMG_SIZE), 128.0))
    assert not np.isnan(resultado).any()


@pytest.mark.parametrize("altura, largura", [(480, 640), (640, 480), (100, 100), (7, 3)])
def test_recorte_rosto_devolve_quadrado_dentro_da_foto(altura, largura):
    topo, esquerda, lado = caixa_do_rosto(altura, largura)
    assert 0 <= topo and topo + lado <= altura and 0 <= esquerda and esquerda + lado <= largura
    recorte = recorte_rosto(np.zeros((altura, largura, 3)))
    assert recorte.shape[0] == recorte.shape[1] == lado == max(1, int(min(altura, largura) * FRACAO_ROSTO))


def test_mascara_oval_mantem_o_centro_e_apaga_os_cantos():
    m = mascara_oval(IMG_SIZE)
    assert m[IMG_SIZE // 2, IMG_SIZE // 2] == 1                 # centro (rosto) intacto
    assert m[0, 0] == m[0, -1] == m[-1, 0] == m[-1, -1] == 0    # cantos (fundo) apagados
    assert mascara_oval(IMG_SIZE, escala=0.7).sum() < m.sum()   # escala menor = oval menor
    img = np.full((IMG_SIZE, IMG_SIZE), 255.0)
    resultado = aplicar_mascara_oval(img)
    assert resultado[0, 0] == 127.5 and resultado[IMG_SIZE // 2, IMG_SIZE // 2] == 255
