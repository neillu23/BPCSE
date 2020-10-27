import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np


def _pad_numbers(in_width, filter_size, stride):
    if stride == 2:
        out_width = np.ceil(float(in_width) / float(stride))
    else:
        out_width = in_width
    p = int(max(stride*(out_width-1)-in_width+filter_size, 0))
    if p%2==0:
        return [p//2, p//2]
    else:
        return [(p//2)+1, p//2]

def _pad_numbers_plus(in_width, filter_size, stride):
    if stride == 2:
        out_width = np.ceil(float(in_width) / float(stride))
    else:
        out_width = in_width-1
    p = int(max(stride*(out_width-1)-in_width+filter_size, 0))
    if p%2==0:
        return [p//2, p//2]
    else:
        return [(p//2)+1, p//2]

def _lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def _selu(x, name="selu"):
    """ When using SELUs you have to keep the following in mind:
    # (1) scale inputs to zero mean and unit variance
    # (2) use SELUs
    # (3) initialize weights with stddev sqrt(1/n)
    # (4) use SELU dropout
    """
    with tf.name_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def _prelu(x, name='prelu'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # make one alpha per feature
        alpha = tf.get_variable('alpha', in_shape[-1],
                                initializer=tf.constant_initializer(0.),
                                dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
    return pos + neg

def activation(x,name='relu', act_name='activation'):
    with tf.variable_scope(act_name):
        if name == 'prelu':
            return _prelu(x)
        elif name == 'lrelu':
            return _lrelu(x,0.2)
        elif name == 'tanh':
            return tf.nn.tanh(x)
        elif name == 'relu' :
            return tf.nn.relu(x)
        elif name == 'sigmoid':
            return tf.nn.sigmoid(x)
        else:
            return x

def normalization(x, mode='layer_norm', name='normalization', is_training=True):
    # Data format = NCHW
    with tf.variable_scope(name):
        if mode == 'layer_norm':
            return _layernorm(x, axis=[1, 2, 3], name='norm')
        elif mode == 'instance_norm':
            return  tf.contrib.layers.instance_norm(x, data_format="NCHW")
        elif mode == 'batch_norm':
            return tf.layers.batch_normalization(x, axis=1, training=is_training)
        else:
            return x

def _layernorm(x, axis, name):
    '''
    Layer normalization (Ba, 2016)
    J: Z-normalization using all nodes of the layer on a per-sample basis.
    Input:
        `x`: channel_first/NCHW format! (or fully-connected)
        `axis`: list
        `name`: must be assigned
    
    Example:
        # axis = [1, 2, 3]
        # x = tf.random_normal([64, 3, 10, 10])
        # name = 'D_layernorm'
    
    Return:
        (x - u)/s * scale + offset
    Source: 
        https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/layernorm.py
    '''
    mean, var = tf.nn.moments(x, axis, keep_dims=True)
    n_neurons = x.get_shape().as_list()[axis[0]]
    offset = tf.get_variable(
        name+'.offset',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.zeros_initializer
    )
    scale = tf.get_variable(
        name+'.scale',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.ones_initializer
    )
    return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-5)

def gru(inputs, bidirection=True, num_units=512):
    cell = tf.contrib.rnn.GRUCell(num_units)
    if bidirection:
        cell_bw = tf.contrib.rnn.GRUCell(num_units)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell_bw, inputs,
                        dtype=tf.float32
                    )
        return tf.concat(outputs, 2), tf.concat(state, 1)
    else:
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        return outputs, state

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def check_dir(path_name):
    if not tf.gfile.Exists(path_name):
        tf.gfile.MkDir(path_name)

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

### ref: https://github.com/Kyubyong/transformer/blob/master/modules.py
def masked_multihead_attention(queries, 
                                keys, 
                                num_units=None, 
                                num_heads=8, 
                                dropout_rate=0.,
                                is_training=True,
                                scope="multihead_attention", 
                                ):
    '''Applies masked multihead self-attention.
    TODO: Right now the attention is sum of the tokens.
          Maybe use hard attention in the future?
    Args:
      queries: A 3d tensor with shape of [N, T, C].
      keys: A 3d tensor with shape of [N, n_tokens, token_emb_size/h].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.

        
    Returns
      A 3d tensor with shape of (N, T_q, token_emb_size)  
    '''
    assert num_units % num_heads == 0

    with tf.variable_scope(scope):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T, num_units)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, t, num_units)
        V = tf.layers.dense(keys, num_units, activation=None) # (N, t, num_units)        
        # expand for later use
        Q_ = tf.expand_dims(Q, axis=1) # (N, 1, T, num_units)
        K_ = tf.expand_dims(K, axis=1) # (N, 1, t, num_units)
        V_ = tf.expand_dims(V, axis=1) # (N, 1, t, token_emb_size/h)
        # Split and concat
        Q_ = tf.concat(tf.split(Q_, num_heads, axis=-1), axis=1) # (N, h, T, num_units/h) 
        K_ = tf.concat(tf.split(K_, num_heads, axis=-1), axis=1) # (N, h, t, num_units/h) 
        V_ = tf.concat(tf.split(V_, num_heads, axis=-1), axis=1) # (N, h, t, num_units/h) 

        def _dot_product(Q_, K_, V_):
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 1, 3, 2])) # (N, h, T, t)        
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            return outputs # (N, h, T, t)

        weights = _dot_product(Q_, K_, V_)


        def _masking(inputs):
            padding_num = -2 ** 32 + 1
            N,H,T_q,T_k = shape_list(inputs)

            diag_vals = tf.ones_like(inputs[0, 0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T, t)
            future_masks = tf.tile(tf.expand_dims(tf.expand_dims(tril, 0), 0), [N, H, 1, 1])  # (N, h, T, t)

            paddings = tf.ones_like(future_masks) * padding_num
            outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)

            return outputs

        ### Against the rule!!!
        def _lookahead_masking(inputs):
            padding_num = -2 ** 32 + 1
            N,H,T_q,T_k = shape_list(inputs)

            diag_vals = tf.ones([T_q+1, T_k+1])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T, t)
            tril = tril[1:, :-1]
            future_masks = tf.tile(tf.expand_dims(tf.expand_dims(tril, 0), 0), [N, H, 1, 1])  # (N, h, T, t)

            paddings = tf.ones_like(future_masks) * padding_num
            outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)

            return outputs

        weights = _masking(weights)
        weights = tf.nn.softmax(weights)           
        ### Dropouts
        drop_weights = tf.layers.dropout(weights, rate=dropout_rate, training=is_training)
               
        ### Weighted sum
        outputs = tf.matmul(drop_weights, V_) # ( N, h, T, num_units/h)
        
        ### Restore shape
        outputs = tf.squeeze(tf.concat(tf.split(outputs, num_heads, axis=1), axis=-1), axis=1)# (N, T, num_units)

        # _, _, num_units = shape_list(queries)        
        # outputs = tf.layers.dense(outputs, units=num_units, activation=None)
        ### Residual connection (Should we use this??)
        # outputs += queries
              
        ### Normalize (Should we use this??)
        # outputs = _ln(outputs, scope='layer_norm') # (N, T, token_emb_size)
 
    return outputs, weights
