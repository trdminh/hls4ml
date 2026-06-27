import csv
import os
import numpy as np
import tensorflow as tf

class Utils:
    def __init__(self, path="cpp/"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = path if os.path.isabs(path) else os.path.join(self.base_dir, path)

        self.num_classes = 6
        self.input_dim = 276
        self.dims = [276, 128, 64, 32, 6]

        self.input_scale = 64
        self.weight_scale = 64
        self.activation_scale = 64

        self.log_scales = [6, 6, 6, 6]

        self.load_data()
    def load_data(self):
        X = np.load(os.path.join(self.base_dir, "X.npy"))
        Y = np.load(os.path.join(self.base_dir, "Y.npy"))

        X_test = np.load(os.path.join(self.base_dir, "x_test.npy"))
        Y_test = np.load(os.path.join(self.base_dir, "y_test.npy"))

        Y = Y[:, 0] - 1
        Y_test = Y_test[:, 0] - 1
        self.x_mean = np.mean(X)
        self.x_std = np.std(X) + 1e-8

        X = (X - self.x_mean) / self.x_std
        X_test = (X_test - self.x_mean) / self.x_std

        Y = tf.keras.utils.to_categorical(Y, self.num_classes)
        Y_test = tf.keras.utils.to_categorical(Y_test, self.num_classes)

        self.x_train = X.astype(np.float32)
        self.y_train = Y.astype(np.float32)
        self.x_test = X_test.astype(np.float32)
        self.y_test = Y_test.astype(np.float32)
    def train(self, train=False, model_path='model.h5'):
        if train or not os.path.exists(model_path):
            inputs = tf.keras.Input(shape=(self.input_dim,))
            x = tf.keras.layers.Dense(128, activation="relu", name="dense_0")(inputs)
            x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
            x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
            outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="dense_3")(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(
                loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"],
            )
            
            model.fit(
                self.x_train,
                self.y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=1
            )
            
            score = model.evaluate(self.x_test, self.y_test, verbose=0)
            print("Train Accuracy:", score[1])
            model.save(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
            score = model.evaluate(self.x_test, self.y_test, verbose=0)
            print("Train accuracy:", score[1])
        return model