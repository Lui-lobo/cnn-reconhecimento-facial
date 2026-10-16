# Testes da documentação do site: cada explicação (<li data-trecho="...">) precisa apontar para uma linha que existe
# no código real da função. Se alguém mudar o código e o trecho sumir, este teste avisa.
# Rodar na raiz do repositório com:  pytest
import os
import re
from html.parser import HTMLParser

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LeitorDeNotas(HTMLParser):
    # Percorre o index.html e junta, para cada bloco de código, a função que ele mostra e os trechos das notas
    def __init__(self):
        super().__init__()
        self.blocos = []                # lista de (origem, [trechos])

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "pre" and ("data-codigo" in attrs or "data-js" in attrs):
            self.blocos.append((attrs.get("data-codigo") or "js:" + attrs["data-js"], []))
        if tag == "li" and "data-trecho" in attrs:
            self.blocos[-1][1].append(attrs["data-trecho"])


def extrair_funcao(texto, funcao, eh_python):
    # Mesma regra do servidor.js e do documentacao.js: Python vai até a próxima linha sem recuo; JS até a linha "}"
    linhas = texto.replace("\r\n", "\n").split("\n")
    comeco = re.compile(rf"^def {funcao}\(" if eh_python else rf"^(async )?function {funcao}\(")
    inicio = next(i for i, linha in enumerate(linhas) if comeco.match(linha))
    fim = inicio + 1
    while fim < len(linhas) and (
            (linhas[fim] == "" or linhas[fim][0].isspace()) if eh_python else linhas[fim - 1] != "}"):
        fim += 1
    return linhas[inicio:fim]


def codigo_do_bloco(origem):
    # Lê o arquivo certo para cada tipo de bloco: src/<modulo>.py, servidor.js ou web/js/<arquivo>
    if origem.startswith("js:"):
        arquivo, funcao = origem[3:].split("#")
        caminho, eh_python = os.path.join(RAIZ, "web", "js", arquivo), False
    else:
        modulo, funcao = origem.split("/")
        if modulo == "servidor":
            caminho, eh_python = os.path.join(RAIZ, "servidor.js"), False
        else:
            caminho, eh_python = os.path.join(RAIZ, "src", modulo + ".py"), True
    with open(caminho, encoding="utf-8") as f:
        return extrair_funcao(f.read(), funcao, eh_python)


def test_cada_nota_aponta_para_uma_linha_do_codigo():
    leitor = LeitorDeNotas()
    with open(os.path.join(RAIZ, "web", "index.html"), encoding="utf-8") as f:
        leitor.feed(f.read())
    assert leitor.blocos, "nenhum bloco de código encontrado no index.html"

    problemas = []
    for origem, trechos in leitor.blocos:
        linhas = codigo_do_bloco(origem)
        usadas = {}
        for trecho in trechos:
            n = next((i for i, linha in enumerate(linhas) if trecho in linha), None)   # a 1ª linha com o trecho
            if n is None:
                problemas.append(f"{origem}: trecho não encontrado: {trecho!r}")
            elif n in usadas:
                problemas.append(f"{origem}: {trecho!r} e {usadas[n]!r} caem na mesma linha")
            else:
                usadas[n] = trecho
    assert not problemas, "\n".join(problemas)
