# Montagem da CNN com a API funcional do Keras, treinada DO ZERO (sem modelo pré-treinado).
# Arquitetura igual à do notebook: 4 blocos convolução + pooling, dropout, densa e softmax.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE


class AcuraciaComResposta(keras.metrics.Metric):
    # Acurácia só das fotos que têm resposta certa (integrantes e "ninguém"). Os rostos de fora do grupo têm o alvo
    # dividido por igual entre as classes (ver src/desconhecidos.py), então não existe "acerto" para eles.
    # Soma acertos e fotos ao longo da época inteira (um lote só com rostos de fora não entra na conta).
    # O nome "accuracy" mantém as chaves "accuracy" e "val_accuracy" do histórico do Keras.
    def __init__(self, name="accuracy", **opcoes):
        super().__init__(name=name, **opcoes)
        self.acertos = self.add_variable(shape=(), initializer="zeros", name="acertos")
        self.fotos = self.add_variable(shape=(), initializer="zeros", name="fotos")

    def update_state(self, alvo, previsto, sample_weight=None):
        tem_resposta = tf.reduce_max(alvo, axis=-1) > 0.99
        acertou = tf.equal(tf.argmax(alvo, axis=-1), tf.argmax(previsto, axis=-1))
        self.acertos.assign_add(tf.reduce_sum(tf.cast(acertou & tem_resposta, tf.float32)))
        self.fotos.assign_add(tf.reduce_sum(tf.cast(tem_resposta, tf.float32)))

    def result(self):
        return self.acertos / tf.maximum(self.fotos, 1.0)


def construir_cnn(n_classes):
    # Monta e compila a rede. Cada linha recebe a saída da linha anterior (x) e passa adiante.
    entrada = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="entrada")                    # imagem 96x96, 1 canal

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(entrada)      # detecta bordas simples
    x = layers.MaxPooling2D(2, name="pool1")(x)                                             # 96 -> 48

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)            # combina bordas em texturas
    x = layers.MaxPooling2D(2, name="pool2")(x)                                             # 48 -> 24

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)           # partes do rosto
    x = layers.MaxPooling2D(2, name="pool3")(x)                                             # 24 -> 12

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv4")(x)           # combinações de partes
    x = layers.MaxPooling2D(2, name="pool4")(x)                                             # 12 -> 6

    x = layers.Flatten(name="flatten")(x)                                                   # 6x6x128 -> vetor de 4608
    x = layers.Dropout(0.5, name="dropout")(x)                                              # combate o overfitting
    x = layers.Dense(128, activation="relu", name="densa")(x)                               # combina as características
    x = layers.Dense(n_classes, name="pontuacoes")(x)                                       # 1 pontuação por integrante
    saida = layers.Softmax(name="saida")(x)                                                 # pontuações -> probabilidades

    model = keras.Model(entrada, saida)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),   # algoritmo que ajusta os pesos
        loss="categorical_crossentropy",                        # erro: entropia cruzada (alvo = 1 probabilidade por classe)
        metrics=[AcuraciaComResposta()],                           # acurácia durante o treino (só das fotos com resposta certa)
    )
    return model
