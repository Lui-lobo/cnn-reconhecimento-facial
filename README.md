# Reconhecimento facial com CNN

Uma Rede Neural Convolucional (CNN) treinada **do zero** que reconhece os integrantes do grupo pelo rosto.
O repositório tem o pipeline completo em Python (pré-processamento, divisão, aumento de dados, CNN, treino,
métricas e imagens intermediárias) e um **site local** com:

- **no topo:** o reconhecimento **ao vivo pela webcam**, usando o modelo treinado;
- **logo abaixo:** a **documentação** passo a passo, com o código real de cada etapa, as imagens
  intermediárias e as métricas do último treino.

O notebook original do Google Colab está em [`notebooks/reconhecimento_facial_cnn.ipynb`](notebooks/reconhecimento_facial_cnn.ipynb)
e é a referência do pipeline: mesma lógica, mesmos parâmetros e mesmos nomes de funções.

## Estrutura

```
├── app.py                  servidor Flask: entrega o site e as rotas da API
├── dataset/<nome>/         fotos de cada integrante (o nome da pasta é o rótulo)
├── modelo/                 gerado pelo treino: modelo_cnn.keras, classes.json, resultados.json
├── src/
│   ├── config.py           todos os parâmetros (IMG_SIZE, EPOCAS, BATCH, SEED, LIMIAR...)
│   ├── dados.py            ler_imagem (com correção EXIF), listar_dataset, salvar_fotos
│   ├── preprocessamento.py recorte_central, para_cinza, redimensionar, equalizar_histograma, preprocessar
│   ├── divisao.py          treino/validação/teste por blocos de fotos seguidas
│   ├── aumento.py          aumentar (espelho, deslocamento, contraste, brilho, ruído)
│   ├── modelo.py           construir_cnn
│   ├── metricas.py         matriz de confusão, VP/FP/FN/VN, precisão, recall, F1, especificidade, acurácia
│   ├── visualizacoes.py    imagens intermediárias em PNG
│   ├── reconhecimento.py   regioes_da_tela, reconhecer_quadro, grad_cam (usado pelo servidor)
│   └── treinar.py          executa o pipeline completo
├── web/                    site: index.html, css/, js/, img/resultados/ (PNGs gerados pelo treino)
├── tests/                  testes com pytest
└── notebooks/              notebook original
```

## Como usar

### 1. Pré-requisitos e ambiente virtual

Python **3.10 a 3.13** (testado com Python 3.13 e TensorFlow 2.20). Na pasta do repositório:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Coletar as fotos

Cada integrante precisa de uma pasta `dataset/<nome>/` com as fotos dele. Há duas formas:

- **Pelo site:** rode `python app.py`, abra `http://localhost:5000` e use a seção
  **“Coletar fotos de um integrante para o dataset”**. Digite o nome, clique em “Iniciar captura”: depois de
  uma contagem de 3 s, a página tira N fotos (padrão 80, uma a cada 250 ms) e salva em `dataset/<nome>/`,
  continuando a numeração.
- **Copiando arquivos:** coloque fotos `.jpg`, `.jpeg` ou `.png` em `dataset/<nome>/`. Os arquivos são lidos em
  ordem alfabética, que deve ser a ordem em que foram tirados (ex.: `ana_0000.jpg`, `ana_0001.jpg`...).

Dicas para a rede aprender o rosto, e não o fundo:

- façam **3 ou mais sessões** por pessoa, em **lugares e iluminações diferentes**;
- mantenham o rosto centralizado e mexam levemente a cabeça durante a captura;
- juntem **cerca de 150 a 300 fotos por pessoa**, em **quantidades parecidas** entre as pessoas;
- nomes de pasta só com letras minúsculas sem acento, números, `_` ou `-` (pastas que começam com `.` ou `_` são ignoradas).

### 4. Treinar

```bash
python -m src.treinar
```

O script faz, na ordem: leitura → pré-processamento → divisão em treino/validação/teste (por blocos de fotos
seguidas) → aumento de dados → montagem da CNN → treino (até 60 épocas, com EarlyStopping) → avaliação no teste
→ imagens intermediárias. No fim, imprime as métricas no terminal e salva:

| Arquivo | Conteúdo |
|---|---|
| `modelo/modelo_cnn.keras` | a rede treinada |
| `modelo/classes.json` | os nomes dos integrantes, na ordem dos rótulos |
| `modelo/resultados.json` | métricas, matriz de confusão, histórico do treino, divisão dos dados e camadas |
| `web/img/resultados/*.png` | fotos originais, etapas do pré-processamento, aumento, curvas, matriz, métricas, erros, mapas de características e Grad-CAM |

Sempre que coletarem fotos novas, rodem o treino de novo. O servidor percebe o modelo novo sozinho.

### 5. Abrir o site

```bash
python app.py
```

Abra **http://localhost:5000**. Clique em “Ligar webcam” e permita o acesso à câmera.
O navegador só libera a webcam em `localhost` ou em HTTPS, por isso o servidor escuta apenas em `127.0.0.1`.

- **1 pessoa:** a caixa é o quadrado central, o mesmo recorte das fotos de treino (é quando a rede funciona melhor).
- **2 pessoas:** a tela é dividida em duas caixas lado a lado; cada pessoa fica dentro de uma.
- **Limiar:** abaixo dele (padrão 60%), a caixa fica vermelha com “desconhecido”.
- **Ver o que a rede vê:** congela o quadro e mostra as 5 etapas do pré-processamento, o Grad-CAM e as probabilidades.

Se o modelo ainda não foi treinado, o site avisa para rodar `python -m src.treinar` primeiro.

### 6. Testes

```bash
pytest
```

Testa o pré-processamento (formato `(96, 96, 1)` em `float32` entre 0 e 1, equalização sem NaN, recorte quadrado)
e as métricas (VP/FP/FN/VN, precisão, recall, F1, especificidade e acurácia contra valores calculados à mão).

## Regras do trabalho

Não usamos bibliotecas auxiliares para o modelo nem para os tratamentos: nada de detector de rosto pronto
(OpenCV, dlib, face_recognition, MediaPipe), modelo pré-treinado, transfer learning, `ImageDataGenerator`,
camadas de aumento de dados do Keras ou scikit-learn.

| Etapa | Ferramenta |
|---|---|
| Abrir e salvar o arquivo .jpg/.png | `PIL` (só transforma o arquivo em uma matriz de números) |
| Recorte, tons de cinza, redimensionamento, equalização, aumento de dados, divisão dos dados | **código próprio com NumPy** |
| Montar e treinar a CNN | TensorFlow/Keras, **sem modelo pré-treinado** |
| Matriz de confusão e métricas | **código próprio com NumPy** |
| Gráficos e imagens intermediárias | `matplotlib` |
| Servidor local | `Flask` |
| Site e captura da webcam | HTML, CSS e JavaScript puros (`navigator.mediaDevices.getUserMedia`) |

## Privacidade

Este repositório guarda **fotos de rostos** dos integrantes. Recomendamos deixá-lo **privado** ou ter o
consentimento de todos antes de publicá-lo. Se a pasta `dataset/` passar de ~100 MB, usem
[Git LFS](https://git-lfs.com/) para as imagens (`git lfs track "dataset/**/*.jpg"`).
