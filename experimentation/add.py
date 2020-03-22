# Notebook-style Python script for exploring addition with neural networks.

# %%
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Even if it's unused, it needs to be imported
from scipy.special import expit
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers

# %%
def create_model(n_nodes=8, n_depth=1, use_bias=False):
    inp = keras.Input(shape=(2,))
    x = inp
    for _ in range(n_depth):
        x = layers.Dense(n_nodes, activation="sigmoid", use_bias=use_bias)(inp)
    x = layers.Dense(1, use_bias=use_bias)(x)
    model = keras.Model(inputs=inp, outputs=x)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def differentiable_round(x):
    x = tf.maximum(x - 0.499, 0)
    x = x * 10000
    return tf.minimum(x, 1)


def get_max_number(bits):
    return 2 ** (bits - 1)


def get_train_data(n=10000, bits=4):
    # The bits determine the maximum sum that will be handled by the adder.
    max_number = get_max_number(bits)
    x = []
    y = []
    for _ in range(n):
        a = np.random.randint(0, max_number)
        b = np.random.randint(0, max_number)
        x.append([a, b])
        y.append(a + b)
    return np.array(x), np.array(y)


def predict_numbers(model, number_pairs, actual):
    number_pairs = np.array(number_pairs)
    actual = np.array(actual)

    predicted = np.squeeze(model.predict(number_pairs))
    predicted_rounded = np.round(predicted)
    df = pd.DataFrame(
        {
            "a": number_pairs[:, 0],
            "b": number_pairs[:, 1],
            "actual": actual,
            "predicted": predicted,
            "predicted_rounded": predicted_rounded,
            "error": np.abs(actual - predicted),
            "error_rounded": np.abs(actual - predicted_rounded),
        }
    )
    return df


def evaluate_model(model, bits, extend_range=False):
    max_number = get_max_number(bits)
    number_pairs = []
    actual = []

    numbers = list(range(max_number + 1))
    if extend_range:
        # Add 20% extra numbers
        range_add = int(max_number * 0.2)
        numbers += list(range(max_number + 1, max_number + range_add + 1))

    for _ in range(10000):
        a = np.random.choice(numbers)
        b = np.random.choice(numbers)
        number_pairs.append([a, b])
        actual.append(a + b)

    return predict_numbers(model, number_pairs, actual)


# %%
histories = collections.defaultdict(dict)

bit_options = [4, 8, 16]
layer_options = [1, 2, 4, 8, 16]

# %%
def train():
    for n_bits in bit_options:
        for n_nodes in layer_options:
            csv_filename = f"logs/bits_{n_bits}_nodes_{n_nodes}.csv"
            model_filename = f"models/bits_{n_bits}_nodes_{n_nodes}.h5"
            model = create_model(n_nodes=n_nodes)
            x, y = get_train_data(bits=n_bits)
            history = model.fit(
                x,
                y,
                epochs=1000,
                validation_split=0.2,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, verbose=1
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=model_filename,
                        save_best_only=True,
                        monitor="val_loss",
                        verbose=1,
                    ),
                    keras.callbacks.CSVLogger(csv_filename),
                ],
            )
            histories[n_bits][n_nodes] = history


def evaluate():
    fig, ax = plt.subplots(len(bit_options), 1, figsize=(7, 4 * len(bit_options)))

    for i, n_bits in enumerate(bit_options):
        history_df = None
        for j, n_nodes in enumerate(layer_options):
            csv_filename = f"logs/bits_{n_bits}_nodes_{n_nodes}.csv"
            model_filename = f"models/bits_{n_bits}_nodes_{n_nodes}.h5"
            df = pd.read_csv(csv_filename)
            df["dataset"] = f"bits_{n_bits}_nodes_{n_nodes}"
            if history_df is None:
                history_df = df
            else:
                history_df = history_df.append(df)

            m = keras.models.load_model(model_filename)
            metrics_df = evaluate_model(m, n_bits)

            print(f"*** Max bits: {n_bits}, hidden nodes: {n_nodes} ***")
            mse = metrics.mean_squared_error(metrics_df.actual, metrics_df.predicted)
            print(f"Mean squared error: {mse}")
            print(
                f"Mean absolute error: {metrics.mean_absolute_error(metrics_df.actual, metrics_df.predicted)}"
            )
            accuracy = metrics.accuracy_score(
                metrics_df.actual, metrics_df.predicted_rounded
            )
            print(f"Accuracy (rounded): {accuracy:,.1%}")
            print()

        sns.lineplot(data=history_df, x="epoch", y="val_loss", hue="dataset", ax=ax[i])
        ax[i].set(title=f"Maximum bits {n_bits} ({2 ** n_bits})")

    plt.tight_layout()
    plt.show()


# %%
# Load the traine models
b4_l1 = keras.models.load_model("models/bits_4_nodes_1.h5")
b4_l2 = keras.models.load_model("models/bits_4_nodes_2.h5")
b4_l4 = keras.models.load_model("models/bits_4_nodes_4.h5")
b4_l8 = keras.models.load_model("models/bits_4_nodes_8.h5")
b4_l16 = keras.models.load_model("models/bits_4_nodes_16.h5")
b8_l1 = keras.models.load_model("models/bits_8_nodes_1.h5")
b8_l2 = keras.models.load_model("models/bits_8_nodes_2.h5")
b8_l4 = keras.models.load_model("models/bits_8_nodes_4.h5")
b8_l8 = keras.models.load_model("models/bits_8_nodes_8.h5")
b8_l16 = keras.models.load_model("models/bits_8_nodes_16.h5")
b16_l1 = keras.models.load_model("models/bits_16_nodes_1.h5")
b16_l2 = keras.models.load_model("models/bits_16_nodes_2.h5")
b16_l4 = keras.models.load_model("models/bits_16_nodes_4.h5")
b16_l8 = keras.models.load_model("models/bits_16_nodes_8.h5")
b16_l16 = keras.models.load_model("models/bits_16_nodes_16.h5")

# %%
# Choose the model
chosen_network = b4_l16
network = "16-node hidden layer model, maximum trained sum 16"
# network = '16-node hidden layer model, maximum trained sum 65536'
chosen_bits = 4

# %%
evaluate()

# %%
# Evaluation and plotting of model
df = evaluate_model(chosen_network, chosen_bits, extend_range=True)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_trisurf(df.a, df.b, df.error, cmap=cm.coolwarm)
ax.set(
    xlabel="a",
    ylabel="b",
    zlabel="error",
    title=f"Absolute error for predicted a + b\n{network}",
)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots()
ax.hist(df.actual, bins=20)
ax.set(xlabel="sum", title="Distribution of numbers")
plt.tight_layout()
plt.show()

df.sort_values(by="error").head(5)

# %%
chosen_network.summary()

# %%
numbers = [[2, 2]]
print(f"Prediction (sum) {chosen_network.predict(numbers)}")
print()

for layer in chosen_network.layers:
    print(f'Layer "{layer.name}" weights:')
    print(np.round(np.squeeze(layer.get_weights()), 3))
print()

inp = chosen_network.layers[0](np.array(numbers))
hidden = chosen_network.layers[1](inp)
prediction = chosen_network.layers[2](hidden)
print(f"Hidden layer vector: {hidden}")
print(f"Prediction (sum): {prediction}")
print()

# Manually calculate the sigmoids for the nodes in the hidden layer, just to
# compare.
hidden_values = np.squeeze(chosen_network.layers[1].get_weights()) * np.array(
    numbers
).reshape(2, 1)
hidden_sum = np.sum(hidden_values, axis=0)
hidden_sigmoid = expit(hidden_sum)
print(f"Hidden values before sum {hidden_values}")
print(f"Hidden sum: {np.round(hidden_sum, 2)}")
print(f"Hidden sigmoid: {np.round(hidden_sigmoid, 2)}")

# %%
# Show error plot for specific sum
number_pairs = []
actual = []
the_sum = 5000
for a in range(0, the_sum):
    b = the_sum - a
    number_pairs.append([a, b])
    actual.append(a + b)

df = predict_numbers(chosen_network, number_pairs, actual)
df["x_name"] = df.apply(lambda x: f"{x.a:.0f}+{x.b:.0f}", axis=1)
ax = df.plot(kind="line", x="x_name", y="error")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.set(
    xlabel="",
    ylabel="Error",
    title=f"Absolute error for pairs of numbers that sum to {the_sum}\n{network}",
)
ax.legend("")
plt.tight_layout()
plt.show()

# %%
number_pairs = []
actual = []
for a in range(0, 10000, 100):
    for b in range(0, 10000, 100):
        number_pairs.append([a, b])
        actual.append(a + b)
df = predict_numbers(chosen_network, number_pairs, actual)
df.describe()
