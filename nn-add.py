import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    inp = keras.Input(shape=(2,))
    x = layers.Dense(8, activation='sigmoid')(inp)
    x = layers.Dense(1)(x)
    model = keras.Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def get_data(max_n=10):
    x = []
    y = []
    for _ in range(10000):
        a = np.random.randint(0, max_n)
        b = np.random.randint(0, max_n)
        x.append([a, b])
        y.append(a + b)
    return x, y


def run():
    model = create_model()
    max_n = 10
    x, y = get_data(max_n)
    model.fit(x, y, epochs=100)
    print(f"loss: {model.evaluate(x, y)}")
    for i in range(5):
        a = np.random.randint(0, max_n)
        b = np.random.randint(0, max_n)
        p = np.squeeze(model.predict([[a, b]]))
        print(f"{a} + {b} = {int(np.round(p))} (error {np.around(np.abs(p - (a + b)), 2)})")


if __name__ == "__main__":
    run()
