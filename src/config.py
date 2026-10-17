# Configurações gerais do projeto (é só aqui que precisamos mexer).
# Os valores são os mesmos do notebook original.
import os

# Pasta raiz do repositório (a pasta acima de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Onde ficam as fotos: uma subpasta por integrante.
# A variável de ambiente DATASET_DIR permite testar com outra pasta sem mexer no código.
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(RAIZ, "dataset"))

MODELO_DIR     = os.path.join(RAIZ, "modelo")                        # modelo treinado e resultados
CAMINHO_MODELO = os.path.join(MODELO_DIR, "modelo_cnn.keras")
CAMINHO_CLASSES = os.path.join(MODELO_DIR, "classes.json")
CAMINHO_RESULTADOS = os.path.join(MODELO_DIR, "resultados.json")
IMAGENS_DIR    = os.path.join(RAIZ, "web", "img", "resultados")      # PNGs exibidos na documentação

IMG_SIZE          = 96    # toda imagem vira 96 x 96 pixels
FRACAO_ROSTO      = 0.6   # o quadrado do rosto tem 60% do menor lado da foto (corta a maior parte do fundo)
ALTURA_ROSTO      = 0.45  # o centro desse quadrado fica a 45% da altura (o rosto fica um pouco acima do meio)
FRACAO_DISTANCIA  = (0.45, 0.9)  # aumento: quadrado de 45% a 90% do menor lado = pessoa mais perto ou mais longe
CLASSE_NINGUEM    = "ninguem"   # classe extra para "não há ninguém na caixa" (fundo vazio)
FOTOS_POR_PESSOA  = 80    # fotos tiradas a cada captura pela webcam
COPIAS_AUMENTADAS = 3     # quantas versões modificadas de cada foto de treino vamos criar
EPOCAS            = 60    # número máximo de passadas pelos dados de treino
BATCH             = 32    # quantas fotos a rede vê antes de cada ajuste dos pesos
SEED              = 42    # semente aleatória: garante o mesmo resultado a cada execução
LIMIAR            = 0.6   # abaixo de 60% de confiança, a pessoa é "desconhecida"
