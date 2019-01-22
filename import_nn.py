import pickle
from history import TensorMinMax
import tensorflow as tf
import numpy as np

class ImportNN():
    """
    Class to import trained NN

    Attribute:
        normX (TensorMinMax): Norm of the inputs
        normY (TensorMinMax): Norm of the outputs
        s_data (2-tuple): dimensions of input and outputs
    """
    def __init__(self, directory):
        """Intialize sess and all tensor names

        Parameters:
            directory (str): the directory where the model is stored
        """
        #import all graph info
        with open(directory + 'Import.pkl', 'rb') as file:
            self.dict = pickle.load(file)
            self.normX = self.dict['normX']
            self.normY = self.dict['normY']
            self.s_data = self.dict['s_data']
            self.b_epoch = self.dict['b_epoch']

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            #Import graph
            imported_meta = tf.train.import_meta_graph(directory + "model_iter-" + str(self.b_epoch) + ".meta")
            imported_meta.restore(self.sess, directory +  "model_iter-" + str(self.b_epoch))

            #get all tensor names
            self.output_tf = self.graph.get_tensor_by_name('OUTPUT:0')
            self.input_tf = self.graph.get_tensor_by_name('INPUT:0')
            self.input_tf_parts = []
            for i in range(self.s_data[0]):
                 self.input_tf_parts.append(self.graph.get_tensor_by_name('INPUT_{}:0'.format(i)))
            self.keep_prob = self.graph.get_tensor_by_name('KEEP_PROB:0')

    def validate_input(self, input):
        """Used to check for valid input

        Parameters:
            input (numpy array): numpy array with width s_data[0]
        """
        #validate and prepare data
        input = np.array(input)
        #make sure it's 2-dimensional
        if len(input.shape) == 1:
            input = np.expand_dims(input, axis=1).T
        #make sure it's the right size
        if input.shape[1] != self.s_data[0]:
            raise ValueError("Data is the wrong size")

        return input

    def output(self, input, kp=1):
        """Runs input through neural network

        Parameters:
            input (numpy array): numpy array with width s_data[0]
            kp (int): value from 0 to 1, 1 refers to keeping all nodes, 0 none of them

        Returns:
            output (numpy array): numpy array with width s_data[1]
        """
        #validate data
        input = self.validate_input(input)
        #return the outputs
        return self.sess.run(self.normY.inverse_transform(self.output_tf), feed_dict={self.input_tf: input, self.keep_prob: kp})

    def differentiate(self, input, d, kp=1):
        """Returns derivative of neural network

        Parameters:
            input (numpy array): numpy array with width s_data[0]
            deriv (3-tuple of ints): first refers to output, 2nd input, 3rd the order of derivative
            kp (int): value from 0 to 1, 1 refers to keeping all nodes, 0 none of them

        Returns:
            output (numpy array): numpy array with width s_data[1]
        """
        #validate data
        input = self.validate_input(input)
        #make feed dict
        fd = {self.keep_prob: kp}
        for i in range(self.s_data[0]):
            fd[self.input_tf_parts[i]] = input[:,i:i+1]
        #take first derivatives, then the rest
        deriv = tf.gradients(self.normY.inverse_transform(self.output_tf)[:,d[0]:d[0]+1], self.input_tf_parts[d[1]])[0]
        for i in range(1,d[2]):
            deriv = tf.gradients(deriv, self.input_tf_parts[d[1]])[0]

        return self.sess.run(deriv, feed_dict=fd)

    def rel_error(self, input, output, kp=1):
        """Returns relative error of network

        Parameters:
            input (numpy array): numpy array with width s_data[0]
            output (numpy array): numpy array with width s_data[1]
            kp (int): value from 0 to 1, 1 refers to keeping all nodes, 0 none of them

        Returns:
            relative error (scalar): the relative error of values
        """
        #validate data
        input = self.validate_input(input)
        #get output
        output_nn = self.output(input, kp)
        #get rid of any possible divide by 0's)
        mask = ~np.isin(output, 0)
        #make relative error
        re = np.abs( (output[mask] - output_nn[mask]) / output[mask] )
        return re.mean()
