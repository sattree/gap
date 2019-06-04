import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import contextlib
import math
from timeit import default_timer as timer
from datetime import datetime, timedelta


from sklearn.metrics import classification_report, log_loss
from sklearn.externals.joblib import Parallel, delayed
from attrdict import AttrDict
from sklearn.model_selection import KFold

from ..base_dataloader import DataLoader
from ..mean_pool_model import MeanPoolModel
from .coarse_grain_model import CoarseGrainModel

from swa_tf.stochastic_weight_averaging import StochasticWeightAveraging

import logging
import sys

logger = logging.getLogger('Training')
syslog = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(syslog)

def gelu_fast(_x):
    return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))


def selfattn(X, seq_lens, n_hidden_x, seed):
    dense_init = tf.keras.initializers.glorot_normal(seed=seed)
    
    a_s = tf.keras.layers.Dense(n_hidden_x, activation='tanh', kernel_initializer=dense_init)(X)
    a_s = tf.keras.layers.Dense(1, activation=None, kernel_initializer=dense_init)(a_s)
    
    attn_masker = MaskForAttn()
    a_s = attn_masker.mask_pads(a_s, seq_lens)
    
    a_s = tf.nn.softmax(a_s, axis=1)
    a_s = attn_masker.restore(a_s, seq_lens)
    
    G_s = tf.matmul(a_s, X, transpose_a=True, name='G_s_matmul')
    
    G_s = tf.reshape(G_s, [-1, n_hidden_x])
    
    return G_s

def coattn(X, Q, X_lens, Q_lens, n_hidden_x, seed, name=''):
    dense_init = tf.keras.initializers.glorot_normal(seed=seed)
    
    A = tf.matmul(X, Q, transpose_b=True)
    
    attn_masker = MaskForAttn()

    A = attn_masker.mask_pads(A, X_lens, Q_lens)

    S_s = tf.matmul(tf.nn.softmax(A, axis=2), Q)
    S_s = attn_masker.restore(S_s, X_lens)
    
    
    S_q = tf.matmul(tf.nn.softmax(A, axis=1), X, transpose_a=True)
    S_q = attn_masker.restore(S_q, Q_lens)

    C_s = tf.matmul(tf.nn.softmax(A, axis=2), S_q)
    C_s = attn_masker.restore(C_s, X_lens)

    gru_init = tf.keras.initializers.glorot_uniform(seed=seed)
    gru_init2 = tf.keras.initializers.orthogonal(seed=seed)
    C_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                kernel_initializer=gru_init,
                                                                recurrent_initializer=gru_init2,
                                                                return_sequences=True))(C_s)
    # C_s = transformer_encoder(C_s, X_lens, 2*n_hidden_x, seed, name=name)

    U_s = tf.concat([C_s, S_s], axis=-1, name='document_context')
    
    U_s = tf.keras.layers.Dense(2*n_hidden_x, activation='tanh', kernel_initializer=dense_init)(U_s)
    
    return U_s

def pool_and_attend(inputs, seq_lens, n_hidden, seed):
    dense_init = tf.keras.initializers.glorot_normal(seed=seed)
    batch_size = tf.shape(inputs)[0]

    with tf.name_scope("sm_pooling-layer"):
        masks = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
        masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, tf.shape(inputs)[2]])

        with tf.name_scope("sm_max-pooling-layer"):
            paddings = tf.ones_like(inputs)*(-2**32+1)
            masked_outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
            max_pool = tf.reduce_max(masked_outputs, axis=-2)

        with tf.name_scope("sm_min-pooling-layer"):
            paddings = tf.ones_like(inputs)*(+2**32+1)
            masked_outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
            min_pool = tf.reduce_min(masked_outputs, axis=-2)

        with tf.name_scope("sm_mean-pooling-layer"):
            paddings = tf.zeros_like(inputs)
            masked_outputs = tf.where(tf.equal(masks, 0), paddings, inputs)            
            mean_pool = tf.reduce_sum(masked_outputs, axis=-2) / tf.reduce_sum(masks, axis=-2)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        with tf.name_scope("sm_attentiion-pooling-layer"):
            # a_s = tf.keras.layers.Dense(n_hidden, activation='tanh', kernel_initializer=dense_init)(inputs)
            # self_attentive = tf.squeeze(tf.keras.layers.Dense(1, activation=None, kernel_initializer=dense_init)(a_s))

            # masks = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
            # paddings = tf.ones_like(self_attentive)*(-2**32+1)
            # self_attentive = tf.where(tf.equal(masks, 0), paddings, self_attentive)
            # self_attentive = tf.nn.softmax(self_attentive)

            # masks = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
            # masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, tf.shape(inputs)[2]])
            # paddings = tf.zeros_like(inputs)
            # masked_outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
            # self_attentive_pool = tf.reduce_sum(masked_outputs * tf.expand_dims(self_attentive, -1), axis=-2)
            self_attentive_pool = selfattn(inputs, seq_lens, n_hidden, seed)

        outputs = tf.concat([max_pool, min_pool, mean_pool, self_attentive_pool], axis=1)

        outputs = tf.keras.layers.Dense(n_hidden, activation=None, kernel_initializer=dense_init)(outputs)

    return outputs

def add_timing_signal(x, scope='', min_timescale=1.0, max_timescale=1.0e4):
        with tf.name_scope(scope, values=[x]):
            length = tf.shape(x)[1]
            channels = tf.shape(x)[2]
            position = tf.to_float(tf.range(length))
            num_timescales = channels // 2

            log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
            )
            inv_timescales = min_timescale * tf.exp(
                tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
            )

            scaled_time = (tf.expand_dims(position, 1) *
                           tf.expand_dims(inv_timescales, 0))
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
            signal = tf.reshape(signal, [1, length, channels])

            return x + signal

def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        seq_len=None,
                        seq_len_q=None,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        seed=7,
                        scope="multihead_attention_att", 
                        reuse=None):

    with tf.variable_scope(scope, reuse=False):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, use_bias=False) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, use_bias=False) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.cast(tf.sequence_mask(seq_len), tf.float32)#tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        weights = outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.cast(tf.sequence_mask(seq_len_q), tf.float32)#tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.keras.layers.Dropout(dropout_rate, seed=seed)(outputs)
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
        
        outputs = tf.layers.dense(outputs, num_units, activation=None, use_bias=False)
        
        outputs = tf.contrib.layers.layer_norm(tf.add(queries, outputs))

    return outputs, weights

def feedforward(inputs, 
            num_units=[2048, 512],
            scope="multihead_attention_ff", 
            reuse=False,
           residual=True):

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        
        outputs = tf.contrib.layers.layer_norm(tf.add(inputs, outputs))
    return outputs

def transformer_encoder(outputs, seq_lens, n_hidden, seed, name=''):
    num_blocks = 2
    num_heads = 8
    d_dropout = 0.2

    # outputs = add_timing_signal(outputs, scope=name)

    for i in range(num_blocks):
      with tf.variable_scope("{}_num_blocks_{}".format(name, i)):
          outputs, _ = multihead_attention(queries=outputs, 
                                          keys=outputs,
                                          seq_len=seq_lens,
                                          seq_len_q=seq_lens,
                                          num_units=n_hidden, 
                                          num_heads=num_heads, 
                                          dropout_rate=d_dropout,
                                          is_training=True,
                                          seed=seed,
                                          scope='{}_mh{}'.format(name, i))

          outputs = feedforward(outputs, num_units=[2*n_hidden, n_hidden], scope='{}_ff{}'.format(name, i))
    return outputs

class MaskForAttn():
    def __init__(self):
        pass

    def mask_pads(self, X, X_lens, Q_lens=None):
        if Q_lens is None:
            masks_x = tf.cast(tf.sequence_mask(X_lens), tf.float32)
            masks_x = tf.expand_dims(masks_x, axis=-1)
            
            paddings = tf.ones_like(X)*(-2**32+1)
            X = tf.where(tf.equal(masks_x, 0), paddings, X)
            return X

        masks_x = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks_q = tf.cast(tf.sequence_mask(Q_lens), tf.float32)
        masks_x = tf.expand_dims(masks_x, axis=-1)
        masks_q = tf.expand_dims(masks_q, axis=-1)
        masks = tf.matmul(masks_x, masks_q, transpose_b=True)
        
        paddings = tf.ones_like(X)*(-2**32+1)

        X = tf.where(tf.equal(masks, 0), paddings, X)

        return X

    def restore(self, X, seq_lens):
        masks = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        X = X * masks

        return X

class CoarseGrainModelV2(CoarseGrainModel):
        
    def create_graph(self,
                     batch_size=None, 
                     n_dim_x=1024,
                     n_dim_x_p=1024, 
                     n_dim_x_a=1024,
                     n_dim_x_b=1024,
                     n_dim_c=1024,
                     n_dim_p=1024,
                     n_dim_q=1024,
                     n_hidden_x=1024,
                     n_hidden_x_p=1024, 
                     n_hidden_x_a=1024,
                     n_hidden_x_b=1024,
                     n_hidden_c=1024,
                     n_hidden_p=1024,
                     n_hidden_q=1024,
                     n_hidden_ff=1024,
                     dropout_rate_x=0.5, 
                     dropout_rate_p=0.5, 
                     dropout_rate_c=0.5,
                     dropout_rate_q=0.5,
                     dropout_rate_ff=0.5,
                     dropout_rate_fc=0.5,
                     reg_x=0.01,
                     reg_p=0.01,
                     reg_c=0.01,
                     reg_q=0.01,
                     reg_ff=0.01,
                     reg_fc=0.01,
                     label_smoothing=0.1,
                     data_type=tf.float32, 
                     activation='tanh',
                    use_pretrained=False,
                    use_plus_features=False,
                    use_context=False,
                     use_swa=True,
                    seed=None,
                    device='GPU:0'):
        
        activ = tf.nn.relu
#         activ = gelu_fast
        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        with tf.Graph().as_default() as graph:
            with tf.device('/device:{}'.format(device)):
                d_X = tf.placeholder(data_type, [batch_size, None, n_dim_x])
                d_X_p = tf.placeholder(data_type, [batch_size, None, n_dim_x_p])
                d_X_a = tf.placeholder(data_type, [batch_size, None, n_dim_x_a])
                d_X_b = tf.placeholder(data_type, [batch_size, None, n_dim_x_b])
                d_P = tf.placeholder(data_type, [batch_size, n_dim_p])
                d_C = tf.placeholder(data_type, [batch_size, n_dim_c])
                d_Q = tf.placeholder(data_type, [batch_size, n_dim_q])

                d_y = tf.placeholder(tf.int32, [batch_size, 3])

                d_seq_lens_x = tf.placeholder(tf.int32, [batch_size])
                d_seq_lens_x_p = tf.placeholder(tf.int32, [batch_size])
                d_seq_lens_x_a = tf.placeholder(tf.int32, [batch_size])
                d_seq_lens_x_b = tf.placeholder(tf.int32, [batch_size])

                with tf.name_scope('dense_concat_layers'):
                    dense_init = tf.keras.initializers.glorot_normal(seed=seed)

                    gru_init = tf.keras.initializers.glorot_uniform(seed=seed)
                    gru_init2 = tf.keras.initializers.orthogonal(seed=seed)
                    
#                     d_X = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init)(d_X)
                    X = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                               kernel_initializer=gru_init,
                                                                               recurrent_initializer=gru_init2,
                                                                               return_sequences=True))(d_X)
                    X_s, X = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X)

#                     d_X_p = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init)(d_X_p)
                    X_p = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                                 kernel_initializer=gru_init,
                                                                                 recurrent_initializer=gru_init2,
                                                                                 return_sequences=True))(d_X_p)
                    X_p_s, X_p = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X_p)

#                     d_X_a = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init)(d_X_a)
                    X_a = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                                 kernel_initializer=gru_init,
                                                                                 recurrent_initializer=gru_init2,
                                                                                 return_sequences=True))(d_X_a)
                    X_a_s, X_a = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X_a)

#                     d_X_b = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init)(d_X_b)
                    X_b = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                                 kernel_initializer=gru_init,
                                                                                 recurrent_initializer=gru_init2,
                                                                                 return_sequences=True))(d_X_b)
                    X_b_s, X_b = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X_b)

                    # n_hidden_x, n_hidden_x_p, n_hidden_x_a, n_hidden_x_b = n_dim_x, n_dim_x_p, n_dim_x_a, n_dim_x_b

                    # X_s = transformer_encoder(d_X, d_seq_lens_x, n_hidden_x, seed, name='X')
                    # X_p_s = transformer_encoder(d_X_p, d_seq_lens_x_p, n_hidden_x_p, seed, name='X_p')
                    # X_a_s = transformer_encoder(d_X_a, d_seq_lens_x_a, n_hidden_x_a, seed, name='X_a')
                    # X_b_s = transformer_encoder(d_X_b, d_seq_lens_x_b, n_hidden_x_b, seed, name='X_b')


                    G_s = coattn(X_s, X_p_s, d_seq_lens_x, d_seq_lens_x_p, n_hidden_x, seed, name='1')

                    A_s = coattn(X_s, X_a_s, d_seq_lens_x, d_seq_lens_x_a, n_hidden_x, seed, name='2')
                    A_s = pool_and_attend(A_s, d_seq_lens_x, 2*n_hidden_x, seed)
    #                 A_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
    #                                                                              kernel_initializer=gru_init,
    #                                                                              recurrent_initializer=gru_init2,
    #                                                                              return_sequences=False))(A_s)

                    B_s = coattn(X_s, X_b_s, d_seq_lens_x, d_seq_lens_x_b, n_hidden_x, seed, name='3')
                    B_s = pool_and_attend(B_s, d_seq_lens_x, 2*n_hidden_x, seed)
    #                 B_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
    #                                                                              kernel_initializer=gru_init,
    #                                                                              recurrent_initializer=gru_init2,
    #                                                                              return_sequences=False))(B_s)

                    G_s = pool_and_attend(G_s, d_seq_lens_x, 2*n_hidden_x, seed)
    #                 G_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
    #                                                                              kernel_initializer=gru_init,
    #                                                                              recurrent_initializer=gru_init2,
    #                                                                              return_sequences=False))(G_s)


                    P = tf.keras.layers.Dropout(dropout_rate_p, seed=seed)(d_P)

                    P = tf.keras.layers.Dense(n_hidden_p, activation=None, kernel_initializer=dense_init,
                                              kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_p))(P)
                    P = tf.layers.batch_normalization(P)
                    P = activ(P)

                    C = tf.keras.layers.Dropout(dropout_rate_p, seed=seed)(d_C)

                    C = tf.keras.layers.Dense(n_hidden_c, activation=None, kernel_initializer=dense_init,
                                              kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_p))(C)
                    C = tf.layers.batch_normalization(C)
                    C = activ(C)

                    Q = tf.keras.layers.Dropout(dropout_rate_p, seed=seed)(d_Q)

                    Q = tf.keras.layers.Dense(n_hidden_q, activation=None, kernel_initializer=dense_init,
                                              kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_p))(Q)
                    Q = tf.layers.batch_normalization(Q)
                    Q = activ(Q)

                    X = [G_s, A_s, B_s]
                    # X = tf.concat(X, axis=-1, name='agg_feats')
                    # X = tf.keras.layers.Dropout(dropout_rate_x, seed=7)(X)

                    # X = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init,
                    #                           kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_x))(X)
                    # X = tf.layers.batch_normalization(X)
                    # X = activ(X)

                    feats = X

                    X = [P, C, Q]
                    # X = tf.concat(X, axis=-1, name='agg_feats')
                    # X = tf.keras.layers.Dropout(dropout_rate_x, seed=7)(X)

                    # X = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init,
                    #                           kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_x))(X)
                    # X = tf.layers.batch_normalization(X)
                    # X = activ(X)

                    feats += X

                    X = tf.concat(feats, axis=-1, name='agg_feats')

                    X = tf.keras.layers.Dropout(dropout_rate_fc, seed=seed)(X)

                    y_hat = tf.keras.layers.Dense(3, name = 'output', kernel_initializer=dense_init,
                                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_fc))(X)

                with tf.name_scope('loss'):
                    probs = tf.nn.softmax(y_hat, axis=-1)
                    # label smoothing works
                    loss = tf.losses.softmax_cross_entropy(d_y, logits=y_hat, label_smoothing=label_smoothing)
                    loss = tf.reduce_mean(loss)

                with tf.name_scope('optimizer'):
                    global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
                    learning_rate = 0.001
                    # learning_rate = tf.train.cosine_decay_restarts(
                    #             learning_rate,
                    #             global_step,
                    #             500,
                    #             t_mul=2,
                    #             m_mul=.6,
                    #             alpha=0.01,
                    #             name=None
                    #         )

                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
                    # train_op = optimizer.minimize(loss, global_step=global_step)
            
            # get the trainable variables
            model_vars = tf.trainable_variables()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops = tf.group(*update_ops)

            with tf.control_dependencies([update_ops,]):
                train_op = optimizer.minimize(loss, global_step=global_step, var_list=model_vars)

            # create an op that combines the SWA formula for all trainable weights 
            swa = StochasticWeightAveraging()
            swa_op = swa.apply(var_list=model_vars)

            # now you can train you model, and EMA will be used, but not in your built network ! 
            # accumulated weights are stored in ema.average(var) for a specific 'var'
            # so you will evaluate your model with the classical weights, not with EMA weights
            # trick : create backup variables to store trained weights, and operations to set weights use in the network to weights from EMA

            # Make backup variables
            with tf.variable_scope('BackupVariables'), tf.device('/cpu:0'):
                # force tensorflow to keep theese new variables on the CPU ! 
                backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                               initializer=var.initialized_value())
                               for var in model_vars]

            # operation to assign SWA weights to model
            swa_to_weights = tf.group(*(tf.assign(var, swa.average(var).read_value()) for var in model_vars))
            # operation to store model into backup variables
            save_weight_backups = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))
            # operation to get back values from backup variables to model
            restore_weight_backups = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

  
            saver = tf.train.Saver()
            init = tf.initializers.global_variables()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options)
            sess = tf.Session(graph=graph, config = config)

        return AttrDict(locals())
    
    @staticmethod
    def mean_featurizer(X_bert, X_plus, X_pretrained, y_true, pooling_fn=None):
        
        batch_x = []
        batch_x_p = []
        batch_x_a = []
        batch_x_b = []
        batch_c = []
        batch_p = []
        batch_q = []
        
        seq_lens_x = []
        seq_lens_p = []
        seq_lens_a = []
        seq_lens_b = []
        batch_y = []
        
        max_len = 512
        max_len_tok = 20
        for idx, row in tqdm(X_bert.iterrows(), total=len(X_bert)):
            seq_len = len(row.tokens)

            x = pooling_fn(np.array(row.bert), axis=0)[:seq_len]
            p_vec = x[row.pronoun_offset_token:row.pronoun_offset_token+1]
            a_vec = x[row.a_span[0]:row.a_span[1]+1]
            b_vec = x[row.b_span[0]:row.b_span[1]+1]
            
            # x = np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1]))))
            # x_p = np.vstack((p_vec, np.zeros((max_len_tok-p_vec.shape[0], p_vec.shape[1]))))
            # x_a = np.vstack((a_vec, np.zeros((max_len_tok-a_vec.shape[0], a_vec.shape[1]))))
            # x_b = np.vstack((b_vec, np.zeros((max_len_tok-b_vec.shape[0], b_vec.shape[1]))))

            batch_x.append(x)
            batch_x_p.append(p_vec)
            batch_x_a.append(a_vec)
            batch_x_b.append(b_vec)

            seq_lens_x.append(seq_len)
            seq_lens_p.append(p_vec.shape[0])
            seq_lens_a.append(a_vec.shape[0])
            seq_lens_b.append(b_vec.shape[0])
            
            c = X_pretrained.loc[idx].pretrained
            batch_c.append(c)
            
            plus_row = X_plus.loc[idx]
            p = np.array(plus_row.plus)
            pronoun_vec = p[plus_row.pronoun_offset_token:plus_row.pronoun_offset_token+1]
            a_vec = p[plus_row.a_span[0]:plus_row.a_span[0]+1]
            b_vec = p[plus_row.b_span[0]:plus_row.b_span[0]+1]
            p = np.hstack((np.mean(pronoun_vec, axis=0), np.mean(a_vec, axis=0), np.mean(b_vec, axis=0))).reshape(-1)
            batch_p.append(p)
            
            q = pooling_fn(np.array(row.cls), axis=0).reshape(-1)
            batch_q.append(q)
            
            y = y_true.loc[idx].values
            
            batch_y.append(y)

        return pd.DataFrame([batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, \
            batch_q, batch_y, seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b]).T