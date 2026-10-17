# O oval-guia da coleta (web/js/coleta.js) precisa ficar exatamente onde o pré-processamento recorta o rosto.
# Os valores estão repetidos no JavaScript; este teste avisa se alguém mudar um lado e esquecer o outro.
# Rodar na raiz do repositório com:  pytest
import os
import re

from src.config import FRACAO_ROSTO, ALTURA_ROSTO

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_oval_da_coleta_usa_os_mesmos_valores_do_config():
    with open(os.path.join(RAIZ, "web", "js", "coleta.js"), encoding="utf-8") as f:
        texto = f.read()
    valor = lambda nome: float(re.search(rf"const {nome} = ([\d.]+);", texto).group(1))
    assert valor("FRACAO_ROSTO") == FRACAO_ROSTO
    assert valor("ALTURA_ROSTO") == ALTURA_ROSTO
