import tensorflow as tf
import numpy as np
import pickle, os
#from history import TensorMinMax
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def nnimport(data, directory=""):
    """Runs data through previously trained NN

    Parameters:
        data (np.array) : data to be ran through network. Each row is a data point,
                            make sure it has the right width
        directory (str) : dir to where nn is stored. make sure to include last /

    Returns:
        output (np.array): returns the output of the neural network, with rows
                            corresponding to each data point
        gradients (np.array):  3 dimensional array. Rows are each output, while each column
                            corresponds to a input, and depth is each data point"""
    #get the norm
    with open(directory + 'Import.pkl', 'rb') as file:
        dict = pickle.load(file)
        normX = dict['normX']
        normY = dict['normY']
        s_data = dict['s_data']
        b_epoch = dict['b_epoch']

    #validate and prepare data
    data = np.array(data)
    #make sure it's 2-dimensional
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1).T
    #make sure it's the right size
    if data.shape[1] != s_data[0]:
        raise ValueError("Data is the wrong size")
    #put in zeros for output for denormalizing
    data = np.hstack(( data, np.zeros((data.shape[0], s_data[1])) ))

    #get graph ready
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(directory + "model_iter-" + str(b_epoch) + ".meta")

    #actually restore graph and get data
    with tf.Session() as sess:
        imported_meta.restore(sess, directory +  "model_iter-" + str(b_epoch))
        graph = tf.get_default_graph()

        keep_prob = graph.get_tensor_by_name('KEEP_PROB:0')
        output_tf = graph.get_tensor_by_name('OUTPUT:0')
        input_tf = graph.get_tensor_by_name('INPUT:0')

        #get output
        output = sess.run(normY.inverse_transform(output_tf), feed_dict={input_tf: data[:,0:s_data[0]], keep_prob: 1}).astype('float32')

        #make Jacobian
        jacob = []
        for i in range(s_data[1]):
            jacob.append(tf.gradients(normY.inverse_transform(output_tf)[:,i:i+1], input_tf)[0])

        j_values = []
        for i in range(s_data[1]):
            j_values.append(sess.run(jacob[i], feed_dict={input_tf: data[:,0:s_data[0]], keep_prob: 1}))

    d = np.dstack(j_values)
    d = np.transpose(d, (2,1,0))

    return output, d

if __name__ == "__main__":
    print(nnimport([[1,2,3,4],[5,6,7,8],[9,10,11,12]], 'GAP-NN-Reals/'))
