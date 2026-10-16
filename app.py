# Servidor local (Flask): entrega o site da pasta web/ e as rotas da API usadas pelo JavaScript.
# Uso (na raiz do repositório):  python app.py   e abra http://localhost:5000
import os
import io
import json
import base64
import inspect
import importlib
import numpy as np
import matplotlib
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory

from src.config import RAIZ, CAMINHO_MODELO, CAMINHO_RESULTADOS
from src.dados import imagem_de_dataurl, nome_valido, salvar_fotos
from src.preprocessamento import preprocessar
from src.reconhecimento import carregar_modelo, reconhecer_quadro, criar_modelo_gradcam, grad_cam

PASTA_WEB = os.path.join(RAIZ, "web")
MSG_SEM_MODELO = "O modelo ainda não foi treinado: rode `python -m src.treinar` primeiro."

app = Flask(__name__, static_folder=PASTA_WEB, static_url_path="")   # arquivos de web/ servidos direto
cache = {"data": None}                                                # modelo carregado na memória


def obter_modelo():
    # Carrega o modelo do disco só quando precisa, e de novo se ele foi retreinado (o arquivo mudou)
    if not os.path.exists(CAMINHO_MODELO):
        return None
    data = os.path.getmtime(CAMINHO_MODELO)
    if cache["data"] != data:
        model, classes = carregar_modelo()
        cache.update(data=data, model=model, classes=classes, gradcam=criar_modelo_gradcam(model))
    return cache


def png_base64(matriz):
    # Converte uma matriz de pixels (0-255) em um PNG no formato dataURL, que o navegador exibe direto
    buffer = io.BytesIO()
    Image.fromarray(np.clip(matriz, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/")
def pagina():
    # Página única: aplicação ao vivo no topo e documentação embaixo
    return send_from_directory(PASTA_WEB, "index.html")


@app.post("/api/reconhecer")
def api_reconhecer():
    # Recebe um quadro da webcam, recorta 1 ou 2 regiões fixas e devolve as probabilidades de cada uma.
    # O servidor não guarda estado: a média dos últimos quadros é feita no JavaScript.
    m = obter_modelo()
    if m is None:
        return jsonify(erro=MSG_SEM_MODELO), 503
    dados = request.get_json()
    quadro = imagem_de_dataurl(dados["imagem"])
    n_pessoas = 2 if int(dados.get("n_pessoas", 1)) == 2 else 1
    return jsonify(caixas=reconhecer_quadro(m["model"], quadro, n_pessoas), classes=m["classes"])


@app.post("/api/etapas")
def api_etapas():
    # Botão "Ver o que a rede vê": devolve as 5 etapas do pré-processamento da região central,
    # o mapa de calor do Grad-CAM e a previsão
    m = obter_modelo()
    if m is None:
        return jsonify(erro=MSG_SEM_MODELO), 503
    quadro = imagem_de_dataurl(request.get_json()["imagem"])
    entrada, etapas = preprocessar(quadro, mostrar_etapas=True)       # etapas = [original, 1, 2, 3, 4]
    probs = m["model"](entrada[np.newaxis], training=False).numpy()[0]

    # Grad-CAM: mapa de calor colorido (escala "jet") misturado com a imagem de entrada em cinza
    calor, _ = grad_cam(m["gradcam"], entrada)
    colorido = matplotlib.colormaps["jet"](calor)[:, :, :3] * 255
    mistura = 0.55 * entrada * 255 + 0.45 * colorido                  # (96,96,1) + (96,96,3) -> (96,96,3)

    return jsonify(
        etapas=[png_base64(etapas[1]), png_base64(etapas[2]), png_base64(etapas[3]),
                png_base64(etapas[4]), png_base64(entrada[:, :, 0] * 255)],
        gradcam=png_base64(mistura),
        probabilidades=probs.tolist(),
        classes=m["classes"],
    )


@app.post("/api/fotos")
def api_fotos():
    # Recebe as fotos capturadas no navegador e salva em dataset/<nome>/, continuando a numeração
    dados = request.get_json()
    nome = str(dados.get("nome", "")).strip().lower()
    if not nome_valido(nome):
        return jsonify(erro="Nome inválido: use só letras minúsculas sem acento, números, _ ou -."), 400
    imagens = [imagem_de_dataurl(texto) for texto in dados.get("imagens", [])]
    if not imagens:
        return jsonify(erro="Nenhuma foto recebida."), 400
    pasta = salvar_fotos(nome, imagens)
    return jsonify(salvas=len(imagens), pasta=os.path.relpath(pasta, RAIZ).replace(os.sep, "/"))


@app.get("/api/resultados")
def api_resultados():
    # Devolve as métricas, a matriz de confusão e o histórico do treino (gerados por src/treinar.py)
    if not os.path.exists(CAMINHO_RESULTADOS):
        return jsonify(erro=MSG_SEM_MODELO), 503
    with open(CAMINHO_RESULTADOS, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.get("/api/codigo/<modulo>/<funcao>")
def api_codigo(modulo, funcao):
    # Devolve o código-fonte real de uma função de src/, para a documentação nunca ficar desatualizada.
    # Só aceita módulos que existem em src/ e funções definidas neles.
    modulos = [a[:-3] for a in os.listdir(os.path.join(RAIZ, "src")) if a.endswith(".py") and a != "__init__.py"]
    if modulo not in modulos:
        return jsonify(erro="Módulo não encontrado."), 404
    objeto = getattr(importlib.import_module(f"src.{modulo}"), funcao, None)
    if not inspect.isfunction(objeto) or objeto.__module__ != f"src.{modulo}":
        return jsonify(erro="Função não encontrada."), 404
    return jsonify(modulo=modulo, funcao=funcao, codigo=inspect.getsource(objeto))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)   # só nesta máquina (a webcam exige localhost ou HTTPS)
