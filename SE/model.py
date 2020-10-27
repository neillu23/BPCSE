from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
import re
from tensorflow.keras.layers import Conv1D as conv1d
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from hyperparameters import Hyperparams as hp
### data_format = 'NCHW'
### Spectrogram shape = [N, 1, F, T]

class SEmodel_Base(object):
    def __init__(self, name=None, frame_size=None):
        self.name = name
        self.frame_size = frame_size
        self.layers = []

    def __call__(self, x=None, reuse=True, is_training=True):
        raise NotImplementedError

    def print_layers(self):
        print(self.name)
        for l in self.layers:
            print(l.get_shape())

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name == var.name.split('/')[0]]

class spec_CNN(SEmodel_Base):
    ''' Multiple frames to multiple frames
    input  = [N, 1, F, T]
    output = [N, 1, F, T]
    '''
    def __init__(self, name="spec_CNN"):
        super().__init__(name)
        self.weights = []

        for ngf in hp.num_layers:
            conv = conv1d(filters=ngf, kernel_size=7, strides=1, padding='causal', data_format='channels_first')
            self.weights.append(conv)

        out = conv1d(filters=hp.f_bin, kernel_size=1, strides=1, padding='causal', data_format='channels_first')
        self.weights.append(out)


    def __call__(self, x, reuse=True, is_training=True):
        norm_mode   = hp.normalization_mode
        act_func    = hp.activation
        self.layers.append(x)

        with tf.variable_scope(self.name, reuse=reuse) as vs:
            for i, ngf in enumerate(hp.num_layers):
                with tf.variable_scope("layer_%d" % (i)):
                    if i == 0:
                        input_ = tf.squeeze(x, 1)
                    else:
                        input_ = self.layers[-1]

                    conv = self.weights[i](input_)
                    conv = activation(conv, act_func, 'act')                 
                    conv = normalization(conv, mode=norm_mode, name='norm', is_training=is_training)
                    self.layers.append(conv)            

            with tf.variable_scope("output_layer"):
                out = self.weights[-1](conv)
                out = activation(out, hp.final_act, 'final_act') 
                out = tf.expand_dims(out, 1)  
                self.layers.append(out)

        if reuse == False:
            self.print_layers()

        return self.layers[-1]

class framewise_LSTM(SEmodel_Base):
    ''' One frame predict one frame LSTM
    input  = [N, 1, F, T]
    output = [N, 1, F, T]
    '''
    def __init__(self, name="framewise_LSTM"):
        super().__init__(name)        
        units = hp.n_units
        self.weights = []

        # Modify for BLSTM
        #h1 = Bidirectional(LSTM(units=units//2, return_sequences=True, unroll=False))
        #h2 = Bidirectional(LSTM(units=units//2, return_sequences=True, unroll=False))
        h1 = LSTM(units=units, return_sequences=True, unroll=False)
        h2 = LSTM(units=units, return_sequences=True, unroll=False)
        h3 = tf.keras.layers.Dense(units=hp.f_bin, activation=None)
        self.weights.append(h1)
        self.weights.append(h2)
        self.weights.append(h3)

    def __call__(self, x, reuse=True, is_training=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x = tf.squeeze(tf.transpose(x, [0,3,2,1]), -1)
            self.layers.append(x)

            h = self.weights[0](x)
            self.layers.append(h)

            h = self.weights[1](h)
            self.layers.append(h)

            h = self.weights[2](h)
            h = activation(h, 'relu', 'final_act')
            self.layers.append(h)

            h = tf.expand_dims(tf.transpose(h, [0,2,1]), axis=1)
            self.layers.append(h)

        if reuse == False:
            self.print_layers()

        return self.layers[-1]

class spec_CNNLSTM(SEmodel_Base):
    ''' Multiple frames to multiple frames
    input  = [N, 1, F, T]
    output = [N, 1, F, T]
    '''
    def __init__(self, name="spec_CNNLSTM"):
        super().__init__(name)
        self.weights = []
        self.num_layers = [2048,1024,512,256,128] 

        for ngf in self.num_layers:
            conv = conv1d(filters=ngf, kernel_size=7, strides=1, padding='causal', data_format='channels_first')
            self.weights.append(conv)

        self.h1 = LSTM(units=256, return_sequences=True, unroll=True)
        self.h2 = LSTM(units=512, return_sequences=True, unroll=True)
        self.h3 = tf.keras.layers.Dense(units=hp.f_bin, activation=None)

    def __call__(self, x, reuse=True, is_training=True):
        norm_mode   = hp.normalization_mode
        act_func    = hp.activation
        self.layers.append(x)

        with tf.variable_scope(self.name, reuse=reuse) as vs:
            for i, ngf in enumerate(self.num_layers):
                with tf.variable_scope("layer_%d" % (i)):
                    if i == 0:
                        input_ = tf.squeeze(x, 1)
                    else:
                        input_ = self.layers[-1]

                    conv = self.weights[i](input_)
                    conv = activation(conv, act_func, 'act')                 
                    conv = normalization(conv, mode=norm_mode, name='norm', is_training=is_training)
                    self.layers.append(conv)            

            with tf.variable_scope("LSTM"):
                transposed = tf.transpose(conv, [0,2,1])

                h = self.h1(transposed)
                self.layers.append(h)

                h = self.h2(h)
                self.layers.append(h)

                h = self.h3(h)
                h = activation(h, 'relu', 'final_act')
                self.layers.append(h)  

                h = tf.expand_dims(tf.transpose(h, [0,2,1]), axis=1)
                self.layers.append(h)

        if reuse == False:
            self.print_layers()

        return self.layers[-1]


class transformer(SEmodel_Base):
    ''' transformer SE
    input  = [N, 1, F, T]
    output = [N, 1, F, T]
    '''
    def __init__(self, name="BERT"):
        super().__init__(name)     
        self.weights = []
        self.num_convs = [1024,512,128,256]

        for ngf in self.num_convs:
            conv = conv1d(filters=ngf, kernel_size=3, strides=1, padding='causal', data_format='channels_first')
            self.weights.append(conv)

    def __call__(self, x, reuse=True, is_training=True):
        ### x = [N, 1, F, T]
        num_hidden_layers   = 8
        hidden_size         = 256
        intermediate_size   = 512
        intermediate_act    = hp.activation
        dropout_rate        = 0.1
        num_heads           = 8
        size_per_head       = 64
        norm_mode           = hp.normalization_mode
        act_func            = hp.activation
        self.attn_hists     = []

        with tf.variable_scope(self.name, reuse=reuse) as vs:
            with tf.variable_scope("causal_conv_preencoder"):
                for i, ngf in enumerate(self.num_convs):
                    with tf.variable_scope("layer_%d" % (i)):
                        if i == 0:
                            input_ = tf.squeeze(x, 1)
                        else:
                            input_ = self.layers[-1]

                        conv = self.weights[i](input_)
                        conv = activation(conv, act_func, 'act')                 
                        # conv = normalization(conv, mode=norm_mode, name='norm', is_training=is_training)
                        self.layers.append(conv)  
                ### Transpose F and T
                trans = tf.transpose(conv, [0,2,1])
                self.layers.append(trans)

            with tf.variable_scope("transformer"):
                for layer_idx in range(num_hidden_layers):
                    with tf.variable_scope("layer_%d" % layer_idx):
                        layer_input = self.layers[-1]

                        with tf.variable_scope("attention"):
                            attention_output, attention_hist = masked_multihead_attention(
                                ### [N, t, nef]
                                queries=layer_input,
                                ### [N, T, hp.token_emb_size]
                                keys=layer_input, 
                                num_units=size_per_head*num_heads, 
                                num_heads=num_heads, 
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                scope="multihead_attention", 
                                )
                            self.attn_hists.append(attention_hist)
                            # Run a linear projection of `hidden_size` then add a residual
                            # with `layer_input`.
                            with tf.variable_scope("output"):
                                attention_output = tf.layers.dense(attention_output, units=hidden_size)
                                attention_output = tf.layers.dropout(attention_output, rate=dropout_rate, training=is_training)
                                attention_output = tf.contrib.layers.layer_norm(attention_output + layer_input, begin_norm_axis=-1, begin_params_axis=-1, scope=None)
                                self.layers.append(attention_output)


                        # The activation is only applied to the "intermediate" hidden layer.
                        with tf.variable_scope("intermediate"):
                            intermediate_output = tf.layers.dense(attention_output, units=intermediate_size)
                            intermediate_output = activation(intermediate_output, intermediate_act)
                            self.layers.append(intermediate_output)

                        # Down-project back to `hidden_size` then add the residual.
                        with tf.variable_scope("output"):
                            layer_output = tf.layers.dense(intermediate_output, units=hidden_size)
                            layer_output = tf.layers.dropout(layer_output, rate=dropout_rate, training=is_training)
                            layer_output = tf.contrib.layers.layer_norm(layer_output + attention_output, begin_norm_axis=-1, begin_params_axis=-1, scope=None)
                            self.layers.append(layer_output)
                #Modify for changing frame size
                linear_proj = tf.layers.dense(self.layers[-1], units=hp.f_bin)
                # linear_proj = tf.layers.dense(self.layers[-1], units=257)
                linear_proj = activation(linear_proj, 'relu', 'final_act')
                linear_proj = tf.expand_dims(tf.transpose(linear_proj, [0,2,1]), axis=1)
                self.layers.append(linear_proj)


        if reuse == False:
            self.print_layers()

        return self.layers[-1]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name == var.name.split('/')[0]]
