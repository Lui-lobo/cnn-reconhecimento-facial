# Testes da parte de emoção (modelo pronto FER+). Não precisam do arquivo do modelo: testam só o que é nosso
# (a entrada no formato do FER+ e o softmax).
# Rodar na raiz do repositório com:  pytest
import numpy as np

from src.config import TAMANHO_EMOCAO, FRACAO_EMOCAO, FRACAO_ROSTO
from src.emocao import EMOCOES, entrada_emocao, softmax
from src.preprocessamento import caixa_do_rosto, preprocessar


def test_entrada_emocao_no_formato_do_ferplus():
    # Qualquer tamanho de foto vira (1, 64, 64) em float32, com tons de 0 a 255 (o FER+ não usa escala 0-1)
    gerador = np.random.default_rng(0)
    for altura, largura in [(480, 640), (3088, 2316), (100, 100)]:
        img = gerador.integers(0, 256, (altura, largura, 3)).astype(np.float64)
        entrada = entrada_emocao(img)
        assert entrada.shape == (1, TAMANHO_EMOCAO, TAMANHO_EMOCAO)
        assert entrada.dtype == np.float32
        assert 0 <= entrada.min() and entrada.max() <= 255


def test_recorte_da_emocao_fica_dentro_do_da_cnn_e_nao_muda_a_cnn():
    # O quadrado do FER+ é menor que o da CNN, com o mesmo centro, e a entrada da CNN continua com FRACAO_ROSTO
    for altura, largura in [(480, 640), (3088, 2316), (720, 1280)]:
        t_cnn, e_cnn, l_cnn = caixa_do_rosto(altura, largura, FRACAO_ROSTO)
        t_emo, e_emo, l_emo = caixa_do_rosto(altura, largura, FRACAO_EMOCAO)
        assert l_emo < l_cnn
        assert t_cnn <= t_emo and t_emo + l_emo <= t_cnn + l_cnn
        assert e_cnn <= e_emo and e_emo + l_emo <= e_cnn + l_cnn
        assert abs((t_emo + l_emo / 2) - (t_cnn + l_cnn / 2)) <= 1 and abs((e_emo + l_emo / 2) - (e_cnn + l_cnn / 2)) <= 1
    img = np.random.default_rng(1).integers(0, 256, (480, 640, 3)).astype(np.float64)
    assert np.array_equal(preprocessar(img), preprocessar(img, fracao=FRACAO_ROSTO))


def test_softmax_soma_1_e_mantem_a_ordem():
    pontuacoes = np.array([[2.0, 1.0, 0.1, -3.0, 0.0, 0.5, 1.5, -1.0], [1000.0] * 8])   # o 2º testa números grandes
    probs = softmax(pontuacoes)
    assert np.allclose(probs.sum(axis=1), 1)
    assert not np.isnan(probs).any()
    assert probs[0].argmax() == 0 and np.allclose(probs[1], 1 / 8)


def test_oito_emocoes_com_as_pedidas_pelo_trabalho():
    assert len(EMOCOES) == 8
    for pedida in ["alegre", "triste", "bravo(a)", "com medo"]:
        assert pedida in EMOCOES
