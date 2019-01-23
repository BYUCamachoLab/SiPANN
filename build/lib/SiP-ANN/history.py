import numpy as np
import tensorflow as tf

class History():
    def __init__(self, n_batch, n_layers, n_nodes, keep_rate, l_rate, a_func, job_id, s_data):
        self.epoch = []
        self.loss_tr = []
        self.loss_val = []
        self.r_tr = []
        self.r_val = []
        self.pred_tr = []
        self.pred_val =[]
        self.n_batch = n_batch
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.keep_rate = keep_rate
        self.l_rate = l_rate
        self.a_func = a_func
        self.job_id = job_id
        self.normX = None
        self.normY = None
        self.s_data = s_data

    def add_norm(self, normX, normY=None):
        self.normX = normX
        self.normY = normY

    def epoch_end(self, loss_tr, loss_val, r_tr, r_val, epoch=None):
        if epoch == None:
            self.epoch.append(len(self.epoch))
        else:
            self.epoch.append(epoch)
        self.loss_tr.append(loss_tr)
        self.loss_val.append(loss_val)
        self.r_tr.append(r_tr)
        self.r_val.append(r_val)

    def import_values(self):
        idx = self.loss_val.index(min(self.loss_val))
        dict = {'normX': self.normX,
                'normY': self.normY,
                's_data': self.s_data,
                'b_epoch': idx}
        return dict

    def values(self):
        idx = self.loss_val.index(min(self.loss_val))
        dict = {'loss': self.loss_val[idx],
                'r': self.r_val[idx],
                'n_batch': self.n_batch,
                'n_layers': self.n_layers,
                'keep_rate': self.keep_rate,
                'n_nodes': self.n_nodes,
                'l_rate': self.l_rate,
                'a_func': self.a_func,
                'job_id': self.job_id,
                'b_epoch': self.epoch[idx]}
        return dict

class TensorMinMax():
    def __init__(self, feature_range=(0,1), copy=True):
        self.feature_range = feature_range
        self.copy = copy
        self.min_ = None
        self.scale_ = None
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        self.data_min = np.amin(X, axis=0)
        self.data_max = np.amax(X, axis=0)
        self.scale_ = ((self.feature_range[1] - self.feature_range[0]) / (self.data_max - self.data_min))
        self.min_ = self.feature_range[0] - self.data_min * self.scale_

    def transform(self, X, mode='numpy'):
        if mode == 'numpy':
            X *= self.scale_
            X += self.min_
        elif mode == 'tensor':
            X = X * tf.constant(self.scale_, tf.float32) + tf.constant(self.min_, tf.float32)

        return X

    def inverse_transform(self, X, mode='numpy'):
        if mode == 'numpy':
            X -= self.min_
            X /= self.scale_
        elif mode == 'tensor':
            X = (X -  tf.constant(self.min_, tf.float32)) /  tf.constant(self.scale_, tf.float32)

        return X
