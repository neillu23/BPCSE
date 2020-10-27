#!/usr/local/bin/python3.6

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 


os.environ["CUDA_VISIBLE_DEVICES"]="4"

tf.logging.set_verbosity(tf.logging.ERROR)

train_list = "/home/neillu/BPCSE/train_spec_noisy_list_train.txt"
dev_list = "/home/neillu/BPCSE/train_spec_noisy_list_dev.txt"
test_list = "/mnt/Data/user_vol_2/user_neillu/BPCSE/ts_timit"

ppg_path = "/mnt/Data/user_vol_2/user_neillu/ppg_feat/broad_manner/"
btn_path = "/mnt/Data/user_vol_2/user_neillu/btn_feat/"
pretrain_name = "broad_manner_savemodel/"
result_name = "broad_manner_savemodel/"

result_path = os.path.join(btn_path,result_name)
save_path = os.path.join(btn_path,"models/",result_name)

pretrain_path = os.path.join(btn_path,"models/",pretrain_name) + "model-0"
pretrain = False

if not os.path.exists(result_path):
    os.makedirs(result_path)
class Autoencoder:

    def __init__(self, n_features, learning_rate=0.5, n_hidden=[1000, 500, 250, 2], alpha=0.0):
        self.n_features = n_features

        self.weights = None
        self.biases = None

        self.graph = tf.Graph()  # initialize new grap
        self.build(n_features, learning_rate, n_hidden, alpha)  # building graph
        self.sess = tf.Session(graph=self.graph)  # create session by the graph

    def build(self, n_features, learning_rate, n_hidden, alpha):
        with self.graph.as_default():
            ### Input
            self.train_features = tf.placeholder(tf.float32, shape=(None, n_features))
            self.train_targets = tf.placeholder(tf.float32, shape=(None, n_features))

            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss, _ = self.structure(
                                               features=self.train_features,
                                               targets=self.train_targets,
                                               n_hidden=n_hidden)

            # regularization loss
            # weight elimination L2 regularizer
            self.regularizer = \
                tf.reduce_sum([tf.reduce_sum(
                        tf.pow(w, 2)/(1+tf.pow(w, 2))) for w in self.weights.values()]) \
                / tf.reduce_sum(
                    [tf.size(w, out_type=tf.float32) for w in self.weights.values()])

            # total loss
            self.loss = self.original_loss + alpha * self.regularizer

            # define training operation
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None, n_features))
            self.new_targets = tf.placeholder(tf.float32, shape=(None, n_features))
            self.new_y_, self.new_original_loss, self.new_encoder = self.structure(
                                                          features=self.new_features,
                                                          targets=self.new_targets,
                                                          n_hidden=n_hidden)
            self.new_loss = self.new_original_loss + alpha * self.regularizer

            ### Initialization
            self.init_op = tf.global_variables_initializer()

    def structure(self, features, targets, n_hidden):
        ### Variable
        if (not self.weights) and (not self.biases):
            self.weights = {}
            self.biases = {}

            n_encoder = [self.n_features]+n_hidden
            for i, n in enumerate(n_encoder[:-1]):
                self.weights['encode{}'.format(i+1)] = \
                    tf.Variable(tf.truncated_normal(
                        shape=(n, n_encoder[i+1]), stddev=0.1), dtype=tf.float32)
                self.biases['encode{}'.format(i+1)] = \
                    tf.Variable(tf.zeros(shape=(n_encoder[i+1])), dtype=tf.float32)

            n_decoder = list(reversed(n_hidden))+[self.n_features]
            for i, n in enumerate(n_decoder[:-1]):
                self.weights['decode{}'.format(i+1)] = \
                    tf.Variable(tf.truncated_normal(
                        shape=(n, n_decoder[i+1]), stddev=0.1), dtype=tf.float32)
                self.biases['decode{}'.format(i+1)] = \
                    tf.Variable(tf.zeros(shape=(n_decoder[i+1])), dtype=tf.float32)

        ### Structure
        activation = tf.nn.relu

        encoder = self.get_dense_layer(features,
                                       self.weights['encode1'],
                                       self.biases['encode1'],
                                       activation=activation)

        for i in range(1, len(n_hidden)-1):
            encoder = self.get_dense_layer(
                encoder,
                self.weights['encode{}'.format(i+1)],
                self.biases['encode{}'.format(i+1)],
                activation=activation,
            )

        encoder = self.get_dense_layer(
            encoder,
            self.weights['encode{}'.format(len(n_hidden))],
            self.biases['encode{}'.format(len(n_hidden))],
        )

        decoder = self.get_dense_layer(encoder,
                                       self.weights['decode1'],
                                       self.biases['decode1'],
                                       activation=activation)

        for i in range(1, len(n_hidden)-1):
            decoder = self.get_dense_layer(
                decoder,
                self.weights['decode{}'.format(i+1)],
                self.biases['decode{}'.format(i+1)],
                activation=activation,
            )

        y_ = self.get_dense_layer(
            decoder,
            self.weights['decode{}'.format(len(n_hidden))],
            self.biases['decode{}'.format(len(n_hidden))],
            activation=tf.nn.sigmoid,
        )

        loss = tf.reduce_mean(tf.pow(targets - y_, 2))

        return (y_, loss, encoder)

    def get_dense_layer(self, input_layer, weight, bias, activation=None):
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x

    def fit(self, X, Y, epochs=10, validation_data=None, test_data=None, batch_size=None):
        X = self._check_array(X)
        Y = self._check_array(Y)

        N = X.shape[0]
        random.seed(9000)
        if not batch_size:
            batch_size = N

        self.sess.run(self.init_op)
        with self.graph.as_default():
            self.saver = tf.train.Saver()
        for epoch in range(epochs):
            print('Epoch %2d/%2d: ' % (epoch+1, epochs))
            start_time = time.time()

            # mini-batch gradient descent
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index) > 0:
                index_size = len(index)
                batch_index = [index.pop() for _ in range(min(batch_size, index_size))]

                feed_dict = {self.train_features: X[batch_index, :],
                             self.train_targets: Y[batch_index, :]}
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                print('[%d/%d] loss = %9.4f     ' % (N-len(index), N, loss), end='\r')

            # evaluate at the end of this epoch
            msg_valid = ''
            if validation_data is not None:
                val_loss = self.evaluate(validation_data[0], validation_data[1])
                msg_valid = ', val_loss = %9.4f' % (val_loss)

            train_loss = self.evaluate(X, Y)
            print('[%d/%d] %ds loss = %9.4f %s' % (N, N, time.time()-start_time,
                                                   train_loss, msg_valid))

            # save at the end of this epoch
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.saver.save(self.sess, save_path + 'model', global_step=0)

        if test_data is not None:
            test_loss = self.evaluate(test_data[0], test_data[1])
            print('test_loss = %9.4f' % (test_loss))

    def encode(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_encoder, feed_dict={self.new_features: X})

    def predict(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})

    def evaluate(self, X, Y):
        X = self._check_array(X)
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X,
                                                       self.new_targets: Y})

    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1:
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray

def load_ppg_spectrum(ppg_path):
    #TODO: read name ppg_data in ppg_path files
    ppg_data = {}
    ppg_dim = 0
    for root, dirs, files in os.walk(ppg_path):
        for ppg_file in files:
            if ".scp" in ppg_file:
                continue
            with open(os.path.join(root,ppg_file),'r') as fp:
                lines = fp.readlines()
                name = ""
                start_id = 0
                end_id =0
                for idx, line in enumerate(tqdm(lines)):
                    if '[' in line:
                        name = line.split()[0]
                        start_id = idx + 1
                        if ppg_dim == 0:
                            ppg_dim = len(lines[start_id].split(" ")[2:-1])
                    if start_id != None and "]" in line:
                        end_id = idx
                        ppg_spec = [[float(ele) for ele in line.split(" ")[2:-1]] for line in lines[start_id:end_id+1]]
                        # ppg_spec = np.array([np.array([float(ele) for ele in line.split(" ")[2:-1]]) for line in lines[start_id:end_id+1]])
                        # ppg_spec = ppg_spec.astype(np.float32)
                        ppg_data[name] = ppg_spec
        return ppg_data, ppg_dim

def make_ppg(n_list,name_type,ppg_data):
    n_files = np.array([x[:-1] for x in open(n_list).readlines()])
    # result_array = np.empty((0, ppg_dim))
    result_array = []
    #use n_files to append data
    for i,n_ in enumerate(tqdm(n_files)):
        if name_type == "train":
            # name = n_.split('/')[-1].split('_')[0] + '_' + n_.split('/')[-1].split('_')[1]
            # ppg_spec =  ppg_data[name.upper()]
            name = n_.split('/')[-1].split('.')[0] + '_' + n_.split('/')[-3].split('.')[0] + '_' + n_.split('/')[-2]
            ppg_spec =  ppg_data[name]
        else:
            name = n_.split('/')[-1].split('.')[0] + '_' + n_.split('/')[-3] + '_' + n_.split('/')[-2]
            ppg_spec =  ppg_data[name]
        
        result_array += ppg_spec
        # result_array = np.append(result_array, ppg_spec, axis=0)
    return np.array(result_array)

#TODO: MAKE SURE DATA IS IN CORRECT FILE
def write_btn(encode_data,n_list,name_type,ppg_data,btn_path):
    n_files = np.array([x[:-1] for x in open(n_list).readlines()])
    current_index = 0
    #use n_files to load data
    for i,n_ in enumerate(tqdm(n_files)):
        if name_type == "train":
            name = n_.split('/')[-1].split('_')[0] + '_' + n_.split('/')[-1].split('_')[1]
            cur_len = len(ppg_data[name.upper()])
            btn_file = n_.split('/')[-1].split('.')[0]  + ".txt"
        else:
            name = n_.split('/')[-1].split('.')[0] + '_' + n_.split('/')[-3] + "_" + n_.split('/')[-2]
            cur_len = len(ppg_data[name])
            btn_file = name + ".txt"
        
        np.savetxt(os.path.join(btn_path,btn_file), encode_data[current_index: current_index + cur_len])
        current_index += cur_len
    if encode_data.shape[0] != current_index:
        print("encode_data num:",encode_data.shape[0],"writed data:",current_index)
    assert(encode_data.shape[0] == current_index)





if __name__ == '__main__':
    print('Loading ppg Data ...')
    ppg_data, ppg_dim = load_ppg_spectrum(ppg_path)    
    print('Making ppg Data ...')

    train_data = make_ppg(train_list,"train",ppg_data)
    valid_data = make_ppg(dev_list,"train",ppg_data)
    test_data = make_ppg(test_list,"test",ppg_data)
    print('Start training ...')

    model_2 = Autoencoder(
        n_features=ppg_dim,
        learning_rate=0.0005,
        n_hidden=[512,256,96],
        alpha=0.0001,
    )

    # #load
    if pretrain == True:
        with model_2.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(model_2.sess, pretrain_path)   
    else:
        model_2.fit(
            X=train_data,
            Y=train_data,
            epochs=20,
            validation_data=(valid_data, valid_data),
            test_data=(test_data, test_data),
            batch_size=8,
        )


    ### get code
    train_encode = model_2.encode(train_data)
    valid_encode = model_2.encode(valid_data)
    test_encode = model_2.encode(test_data)

    if tf.gfile.Exists(result_path):
        print('Folder already exists: {}\n'.format(result_path))
    else:
        tf.gfile.MkDir(result_path)
    print('Write bottleneck ...')
    write_btn(train_encode,train_list,"train",ppg_data,result_path)
    write_btn(valid_encode,dev_list,"train",ppg_data,result_path)
    write_btn(test_encode,test_list,"test",ppg_data,result_path)
