# Testes da detecção de desconhecidos: o alvo "em dúvida" dos rostos de fora, a escolha do limiar e as pastas.
# Rodar na raiz do repositório com:  pytest
import os
import numpy as np

from src.config import ROSTOS_DE_FORA_DIR, IMG_SIZE
from src.aumento import colar_no_cenario, variar_cena
from src.dados import listar_dataset, listar_rostos_de_fora
from src.desconhecidos import alvos, confianca_como_membro, escolher_limiar
from src.modelo import AcuraciaComResposta


def test_alvos_integrante_tem_classe_certa_e_rosto_de_fora_fica_dividido():
    # 3 classes (0, 1, 2); o rótulo 3 é um rosto de fora do grupo
    alvo = alvos(np.array([0, 2, 3]), 3)
    assert alvo.shape == (3, 3) and alvo.dtype == np.float32
    assert alvo[0].tolist() == [1, 0, 0]
    assert alvo[1].tolist() == [0, 0, 1]
    assert np.allclose(alvo[2], 1 / 3)                 # dividido por igual
    assert np.allclose(alvo.sum(axis=1), 1)            # toda linha é uma distribuição de probabilidades


def test_confianca_como_membro_ignora_a_classe_ninguem():
    probs = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    assert confianca_como_membro(probs, indice_ninguem=2).tolist() == [0.2, 0.6]


def test_escolher_limiar_fica_no_meio_da_folga():
    # Integrantes sempre com 99% e rostos de fora sempre com 30%: todos os limiares de 50% a 95% acertam tudo;
    # o escolhido é o do meio
    assert escolher_limiar(np.full(10, 0.99), np.full(10, 0.30)) == 0.75


def test_escolher_limiar_sobe_quando_os_de_fora_tem_confianca_alta():
    # Rostos de fora chegando a 80%: só os limiares acima disso rejeitam todos sem perder integrantes
    limiar = escolher_limiar(np.full(10, 0.99), np.array([0.3, 0.5, 0.8, 0.8]))
    assert 0.8 < limiar <= 0.95


def test_acuracia_do_treino_conta_so_as_fotos_com_classe_certa():
    # 2 fotos com dono (a rede acerta 1) e 1 rosto de fora (alvo dividido): acurácia = 1/2, e não 1/3 ou 2/3
    alvo = np.array([[1, 0], [0, 1], [0.5, 0.5]], np.float32)
    previsto = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4]], np.float32)
    acuracia = AcuraciaComResposta()
    acuracia.update_state(alvo, previsto)
    assert float(acuracia.result()) == 0.5
    # Um lote só com rostos de fora não muda o resultado (a conta é da época inteira, não a média dos lotes)
    acuracia.update_state(np.full((4, 2), 0.5, np.float32), np.full((4, 2), 0.5, np.float32))
    assert float(acuracia.result()) == 0.5


def test_colar_no_cenario_poe_o_rosto_no_meio_do_cenario_vazio():
    rosto = np.ones((IMG_SIZE, IMG_SIZE, 1), np.float32)       # "rosto" todo branco
    cenario = np.zeros((IMG_SIZE, IMG_SIZE, 1), np.float32)    # cenário todo preto
    colada = colar_no_cenario(rosto, cenario, escala=0.5, dy=0, dx=0)
    meio = IMG_SIZE // 2
    assert colada.shape == rosto.shape and colada.dtype == np.float32
    assert colada[meio, meio, 0] == 1                          # centro: o rosto
    assert colada[meio, meio - 30, 0] == 0                     # ao lado do rosto encolhido: o cenário
    assert colada[0, 0, 0] == 0.5                              # fora do oval da entrada: o cinza neutro


def test_variar_cena_cola_so_os_rostos_de_fora():
    # Classe 0 = integrante (branco), 1 = "ninguém" (preto), 2 = rosto de fora (branco)
    X = np.stack([np.ones((IMG_SIZE, IMG_SIZE, 1), np.float32), np.zeros((IMG_SIZE, IMG_SIZE, 1), np.float32),
                  np.ones((IMG_SIZE, IMG_SIZE, 1), np.float32)])
    y = np.array([0, 1, 2])
    np.random.seed(0)
    # Só a colagem deixa o canto cinza (0,5); sem ela, o canto continua branco ou preto
    coladas = sum(variar_cena(2, X, X, y, 1, 2)[0, 0, 0] == 0.5 for _ in range(200))
    assert 120 < coladas < 180                                 # perto de 75% das vezes
    assert all(variar_cena(0, X, X, y, 1, 2)[0, 0, 0] != 0.5 for _ in range(50))   # integrante: nunca é colado


def test_pasta_desconhecido_nao_vira_classe(tmp_path):
    # dataset/desconhecido/ são pessoas de fora: não entra na lista de classes, entra na lista de rostos de fora
    for pasta in ["ana", "bruno", "desconhecido"]:
        os.makedirs(tmp_path / pasta)
        (tmp_path / pasta / f"{pasta}_0000.jpg").write_bytes(b"")
    classes, caminhos, rotulos = listar_dataset(str(tmp_path))
    assert classes == ["ana", "bruno"] and len(caminhos) == 2 and rotulos.tolist() == [0, 1]
    de_fora = listar_rostos_de_fora(str(tmp_path))
    assert de_fora[-1] == str(tmp_path / "desconhecido" / "desconhecido_0000.jpg")


def test_rostos_publicos_estao_no_repositorio():
    # O treino depende deles para a rede aprender a ficar em dúvida
    fotos = [a for a in os.listdir(ROSTOS_DE_FORA_DIR) if a.endswith(".jpg")]
    assert len(fotos) >= 500
