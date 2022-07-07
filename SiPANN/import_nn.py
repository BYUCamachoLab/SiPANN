import pickle
from itertools import combinations_with_replacement as comb_w_r

import numpy as np
import tensorflow as tf


class TensorMinMax:
    """Copy of sklearn's MinMaxScaler implemented to work with tensorflow.

    When used, tensorflow is able to take gradients on the transformation as
    well as on the network itself, allowing for gradient-based optimization in
    inverse design problems.

    Parameters
    ----------
        feature_range : 2-tuple, optional
            Desired range of transformed data. Defaults to (0, 1)
        copy : bool, optional
            Set to false to perform inplace operations. Defaults to True.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy
        self.min_ = None
        self.scale_ = None
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        """Fits the transfomer to the data.

        Essentially finds original min and max of data to be able to shift the data.

        Parameters
        ----------
        X : tensor or ndarray
            Data to fit
        """
        self.data_min = np.amin(X, axis=0)
        self.data_max = np.amax(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max - self.data_min
        )
        self.min_ = self.feature_range[0] - self.data_min * self.scale_

    def transform(self, X, mode="numpy"):
        """Actually does the transorm.

        Parameters
        ----------
        X : tensor or ndarray
            Data to transform
        mode : {'numpy' or 'tensor'}, optional
            Whether to use numpy or tensorflow operations.

        Returns
        -------
        X : tensor or ndarray
            Transformed data
        """
        if mode == "numpy":
            X *= self.scale_
            X += self.min_
        elif mode == "tensor":
            X = X * tf.constant(self.scale_, tf.float32) + tf.constant(
                self.min_, tf.float32
            )

        return X

    def inverse_transform(self, X, mode="numpy"):
        """Undo the transorm.

        Parameters
        ----------
        X : tensor or ndarray
            Data to inverse transform
        mode : {'numpy' or 'tensor'}, optional
            Whether to use numpy or tensorflow operations.

        Returns
        -------
        X : tensor or ndarray
            Inverse transformed data
        """
        if mode == "numpy":
            X -= self.min_
            X /= self.scale_
        elif mode == "tensor":
            X = (X - tf.constant(self.min_, tf.float32)) / tf.constant(
                self.scale_, tf.float32
            )

        return X


class ImportNN:
    """Class to import trained NN.

    This the way we've been saving and using our neural networks. After saving them
    we can simply import them using this class and it keeps them open for as many
    operations as we desire.

    Attributes
    ----------
        normX : TensorMinMax
            Norm of the inputs
        normY: TensorMinMax)
            Norm of the outputs
        s_data : 2-tuple
            Dimensions (size) of input and outputs

    Parameters
    ----------
        directory : str
            The directory where the model has been stored
    """

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
        tf.compat.v1.disable_eager_execution()

    def validate_input(self, input):
        """Used to check for valid input.

        If it is only a single data point, expands the dimensions so it fits properly

        Parameters
        -----------
        input : ndarray
            Numpy array with width s_data[0] (hopefully)

        Returns
        --------
        input : ndarray
            Numpy array with width s_data[0] (hopefully) and height 1
        """
        # validate and prepare data
        input = np.array(input)
        # make sure it's 2-dimensional
        if len(input.shape) == 1:
            input = np.expand_dims(input, axis=1).T
        # make sure it's the right size
        if input.shape[1] != self.s_data[0]:
            raise ValueError("Data is the wrong size")

        return input

    def output(self, input, kp=1):
        """Runs input through neural network.

        Parameters
        ----------
        input : ndarray
            Numpy array with width s_data[0]
        kp : int, optional
                Value from 0 to 1, 1 refers to not performing any dropout on nodes, 0 drops all of them. Defaults to 1.

        Returns
        ----------
        output: ndarray
            numpy array with width s_data[1]
        """
        # validate data
        input = self.validate_input(input)
        # return the outputs
        return self.sess.run(
            self.normY.inverse_transform(self.output_tf),
            feed_dict={self.input_tf: input, self.keep_prob: kp},
        )

    def differentiate(self, input, d, kp=1):
        """Returns partial derivative of neural network.

        Parameters
        ----------
        input : ndarray
            numpy array with width s_data[0]
        d : 3-tuple of ints
            Refers to partial of first element wrt second element to the order of third element
        kp : int, optional
            Value from 0 to 1, 1 refers to not performing any dropout on nodes, 0 drops all of them. Defaults to 1.

        Returns
        ----------
            output : ndarray
                numpy array with width s_data[1]
        """
        # validate data
        input = self.validate_input(input)
        # make feed dict
        fd = {self.keep_prob: kp}
        for i in range(self.s_data[0]):
            fd[self.input_tf_parts[i]] = input[:, i : i + 1]
        # take first derivatives, then the rest
        deriv = tf.gradients(
            self.normY.inverse_transform(self.output_tf)[:, d[0] : d[0] + 1],
            self.input_tf_parts[d[1]],
        )[0]
        for _ in range(1, d[2]):
            deriv = tf.gradients(deriv, self.input_tf_parts[d[1]])[0]

        return self.sess.run(deriv, feed_dict=fd)

    def rel_error(self, input, output, kp=1):
        """Returns relative error of network.

        Parameters
        ----------
        input : ndarray
            Numpy array with width s_data[0]
        output : ndarray
            Numpy array with width s_data[1]
        kp : int, optional
            Value from 0 to 1, 1 refers to not performing any dropout on nodes, 0 drops all of them. Defaults to 1.

        Returns
        ----------
        relative error : scalar
            The relative error of inputs/outputs
        """
        # validate data
        input = self.validate_input(input)
        # get output
        output_nn = self.output(input, kp)
        # get rid of any possible divide by 0's)
        mask = ~np.isin(output, 0)
        # make relative error
        re = np.abs((output[mask] - output_nn[mask]) / output[mask])
        return re.mean()


class ImportLR:
    """Class to import trained Linear Regression.

    To remove independence on sklearn and it's updates, we manually implement an sklearn
    Pipeline that includes (PolynomialFeatures, LinearRegression). We use the actual sklearn
    implementation to train, save the coefficients, and then proceed to implement it here.
    To see how to save a pipeline like above to be used here see SiPANN/LR/regress.py

    Attributes
    -----------
        coef_ : ndarray
            Linear Regression Coefficients
        degree_ : float
            Degree to be used in PolynomialFeatures.
        s_data : 2-tuple
            Dimensions of inputs and outputs

    Parameters
    ----------
        directory : str
            The directory where the model has been stored
    """

    def __init__(self, directory):
        # import all graph info
        with open(directory, "rb") as file:
            dict_ = pickle.load(file)
            self.coef_ = dict_["coef_"]
            self.degree_ = dict_["degree_"]
            self.s_data = dict_["s_data"]

    def make_combos(self, X):
        """Duplicates Polynomial Features.

        Takes in an input X, and makes all possibly combinations of it using
        polynomials of specified degree.

        Parameters
        -----------
        X : ndarray
            Numpy array of size (N, s_data[0])

        Returns
        --------
        polyCombos : ndarray
            Numpy array of size (N, )
        """
        combos = []
        for i in range(self.degree_ + 1):
            combos += list(comb_w_r(range(self.s_data[0]), i))

        # make matrix of all combinations
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
        """Used to check for valid input.

        If it is only a single data point, expands the dimensions so it fits properly

        Parameters
        -----------
        input : ndarray
            Numpy array with width s_data[0] (hopefully)

        Returns
        --------
        input : ndarray
            Numpy array with width s_data[0] (hopefully) and height 1
        """
        # validate and prepare data
        input = np.array(input)
        # make sure it's 2-dimensional
        if len(input.shape) == 1:
            input = np.expand_dims(input, axis=1).T
        # make sure it's the right size
        if input.shape[1] != self.s_data[0]:
            raise ValueError("Data is the wrong size")

        return input

    def predict(self, X):
        """Predict values.

        Runs X through Pipeline to make prediction

        Parameters
        -----------
        X : ndarray
            Numpy array of size (N, s_data[0])

        Returns
        --------
        polyCombos : ndarray
            Numpy array of size (N, )
        """
        X = self.validate_input(X)
        Xcombo = self.make_combos(X)
        return Xcombo @ (self.coef_.T)
