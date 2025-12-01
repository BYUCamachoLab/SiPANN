import pickle
from itertools import combinations_with_replacement as comb_w_r

import numpy as np
import tensorflow as tf

# Move eager execution disable to module level to ensure it runs once upon import.
# This is required for TensorFlow 2.x to handle the v1 graphs properly.
try:
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass # Already disabled or not available

class TensorMinMax:
    """Copy of sklearn's MinMaxScaler implemented to work with tensorflow."""

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy
        self.min_ = None
        self.scale_ = None
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        self.data_min = np.amin(X, axis=0)
        self.data_max = np.amax(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max - self.data_min
        )
        self.min_ = self.feature_range[0] - self.data_min * self.scale_

    def transform(self, X, mode="numpy"):
        if mode == "numpy":
            X *= self.scale_
            X += self.min_
        elif mode == "tensor":
            X = X * tf.constant(self.scale_, tf.float32) + tf.constant(
                self.min_, tf.float32
            )
        return X

    def inverse_transform(self, X, mode="numpy"):
        if mode == "numpy":
            X -= self.min_
            X /= self.scale_
        elif mode == "tensor":
            X = (X - tf.constant(self.min_, tf.float32)) / tf.constant(
                self.scale_, tf.float32
            )
        return X


class ImportNN:
    """Class to import trained NN."""

    def __init__(self, directory):
        # import all graph info
        with open(f"{directory}/Import.pkl", "rb") as file:
            dict_ = pickle.load(file)
            self.normX = dict_["normX"]
            self.normY = dict_["normY"]
            self.s_data = dict_["s_data"]

        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            # Import graph
            imported_meta = tf.compat.v1.train.import_meta_graph(
                f"{directory}/model.meta"
            )
            imported_meta.restore(self.sess, f"{directory}/model")

            # get all tensor names
            self.output_tf = self.graph.get_tensor_by_name("OUTPUT:0")
            self.input_tf = self.graph.get_tensor_by_name("INPUT:0")
            self.input_tf_parts = [
                self.graph.get_tensor_by_name(f"INPUT_{i}:0")
                for i in range(self.s_data[0])
            ]

            self.keep_prob = self.graph.get_tensor_by_name("KEEP_PROB:0")
        # Removed disable_eager_execution from here

    def validate_input(self, input):
        input = np.array(input)
        if len(input.shape) == 1:
            input = np.expand_dims(input, axis=1).T
        if input.shape[1] != self.s_data[0]:
            raise ValueError("Data is the wrong size")
        return input

    def output(self, input, kp=1):
        input = self.validate_input(input)
        return self.sess.run(
            self.normY.inverse_transform(self.output_tf),
            feed_dict={self.input_tf: input, self.keep_prob: kp},
        )

    def differentiate(self, input, d, kp=1):
        input = self.validate_input(input)
        fd = {self.keep_prob: kp}
        for i in range(self.s_data[0]):
            fd[self.input_tf_parts[i]] = input[:, i : i + 1]
        deriv = tf.gradients(
            self.normY.inverse_transform(self.output_tf)[:, d[0] : d[0] + 1],
            self.input_tf_parts[d[1]],
        )[0]
        for _ in range(1, d[2]):
            deriv = tf.gradients(deriv, self.input_tf_parts[d[1]])[0]

        return self.sess.run(deriv, feed_dict=fd)

    def rel_error(self, input, output, kp=1):
        input = self.validate_input(input)
        output_nn = self.output(input, kp)
        mask = ~np.isin(output, 0)
        re = np.abs((output[mask] - output_nn[mask]) / output[mask])
        return re.mean()


class ImportLR:
    """Class to import trained Linear Regression."""

    def __init__(self, directory):
        with open(directory, "rb") as file:
            dict_ = pickle.load(file)
            self.coef_ = dict_["coef_"]
            self.degree_ = dict_["degree_"]
            self.s_data = dict_["s_data"]

    def make_combos(self, X):
        combos = []
        for i in range(self.degree_ + 1):
            combos += list(comb_w_r(range(self.s_data[0]), i))

        n = len(X)
        polyCombos = np.ones((n, len(combos)))
        for j, c in enumerate(combos):
            if c == ():
                polyCombos[:, j] = 1
            else:
                for k in c:
                    polyCombos[:, j] *= X[:, k]
        return polyCombos

    def validate_input(self, input):
        input = np.array(input)
        if len(input.shape) == 1:
            input = np.expand_dims(input, axis=1).T
        if input.shape[1] != self.s_data[0]:
            raise ValueError("Data is the wrong size")
        return input

    def predict(self, X):
        X = self.validate_input(X)
        Xcombo = self.make_combos(X)
        return Xcombo @ (self.coef_.T)