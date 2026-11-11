# Configurações gerais do projeto (é só aqui que precisamos mexer).
# Os valores são os mesmos do notebook original.
import os

# Pasta raiz do repositório (a pasta acima de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Onde ficam as fotos: uma subpasta por integrante.
# A variável de ambiente DATASET_DIR permite testar com outra pasta sem mexer no código.
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(RAIZ, "dataset"))

# Rostos públicos de pessoas de FORA do grupo (uma foto por pessoa): ensinam a rede a ficar em dúvida diante de um estranho.
ROSTOS_DE_FORA_DIR = os.path.join(RAIZ, "rostos_de_fora")

# MODELO_DIR e IMAGENS_DIR também aceitam variável de ambiente: dá para treinar um modelo de teste sem tocar no oficial.
MODELO_DIR     = os.environ.get("MODELO_DIR", os.path.join(RAIZ, "modelo"))   # modelo treinado e resultados
CAMINHO_MODELO = os.path.join(MODELO_DIR, "modelo_cnn.keras")
CAMINHO_CLASSES = os.path.join(MODELO_DIR, "classes.json")
CAMINHO_LIMIAR = os.path.join(MODELO_DIR, "limiar.json")              # limiar de confiança escolhido pelo treino
CAMINHO_RESULTADOS = os.path.join(MODELO_DIR, "resultados.json")
IMAGENS_DIR    = os.environ.get("IMAGENS_DIR", os.path.join(RAIZ, "web", "img", "resultados"))   # PNGs da documentação

IMG_SIZE          = 96    # toda imagem vira 96 x 96 pixels
FRACAO_ROSTO      = 0.6   # o quadrado do rosto tem 60% do menor lado da foto (corta a maior parte do fundo)
ALTURA_ROSTO      = 0.45  # o centro desse quadrado fica a 45% da altura (o rosto fica um pouco acima do meio)
FRACAO_DISTANCIA  = (0.45, 0.9)  # aumento: quadrado de 45% a 90% do menor lado = pessoa mais perto ou mais longe
CLASSE_NINGUEM    = "ninguem"   # classe extra para "não há ninguém na caixa" (fundo vazio)
PASTA_DESCONHECIDO = "desconhecido"   # pasta opcional dataset/desconhecido/: pessoas de fora do grupo fotografadas na
                                      # webcam. Não vira classe: junta-se aos rostos públicos de rostos_de_fora/
FOTOS_POR_PESSOA  = 80    # fotos tiradas a cada captura pela webcam
COPIAS_AUMENTADAS = 3     # quantas versões modificadas de cada foto de treino vamos criar
EPOCAS            = 60    # número máximo de passadas pelos dados de treino
BATCH             = 32    # quantas fotos a rede vê antes de cada ajuste dos pesos
SEED              = 42    # semente aleatória: garante o mesmo resultado a cada execução
LIMIAR            = 0.6   # abaixo dessa confiança, a pessoa é "desconhecida" (vale se o treino não gravou um limiar)
LIMIARES_POSSIVEIS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]   # valores que o treino testa (os do site)

# Emoção (triste, alegre, bravo, com medo...): é a ÚNICA parte que usa um modelo pronto, o FER+ (Microsoft, formato
# ONNX), como o trabalho permite. Quem é a pessoa continua sendo decidido só pela nossa CNN.
CAMINHO_EMOCAO = os.path.join(MODELO_DIR, "emocao_ferplus.onnx")     # baixado sozinho no primeiro uso (~35 MB)
URL_EMOCAO     = ("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
                  "emotion_ferplus/model/emotion-ferplus-8.onnx")
EMOCOES_TESTE_DIR = os.path.join(RAIZ, "emocoes_de_teste")           # fotos públicas rotuladas para testar o FER+
IMAGENS_EMOCAO_DIR = os.path.join(RAIZ, "web", "img", "emocao")     # figuras desse teste (ficam no repositório)
TAMANHO_EMOCAO = 64       # o FER+ recebe um rosto 64 x 64 em tons de cinza
FRACAO_EMOCAO  = 0.45     # quadrado mais fechado que o da CNN (0.6), no mesmo centro: o FER+ foi treinado com
                          # rostos recortados da testa ao queixo. Só vale para a emoção; a CNN continua com FRACAO_ROSTO
