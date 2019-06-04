import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import contextlib
from timeit import default_timer as timer
from datetime import datetime, timedelta


from sklearn.metrics import classification_report, log_loss
from sklearn.externals.joblib import Parallel, delayed
from attrdict import AttrDict
from sklearn.model_selection import KFold

from ..base_dataloader import DataLoader
from ..mean_pool_model import MeanPoolModel

from swa_tf.stochastic_weight_averaging import StochasticWeightAveraging

import logging
import sys

logger = logging.getLogger('Training')
syslog = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(syslog)
logger.propagate = False

class CoarseGrainModel():
    def __init__(self, X, name, device='GPU:0', use_pretrained=False, 
                **model_params):
                # **architecture_config,
                # **training_config):
        self.ckpt_path = name
        self.init_graph(X, device, use_pretrained, **model_params)

    def init_graph(self, X, device, use_pretrained, restore_path=None, **model_params):
#         logger.info('[{}] Creating model graph with params: {}'.format(self.ckpt_path, model_params))

        n_dim_x = len(X[0].values[0][0])
        n_dim_x_p = len(X[1].values[0][0])
        n_dim_x_a = len(X[2].values[0][0])
        n_dim_x_b = len(X[3].values[0][0])
        n_dim_c = len(X[4].values[0])
        n_dim_p = len(X[5].values[0])
        n_dim_q = len(X[6].values[0])
        
        for key, val in model_params.items():
            if key.startswith('n_hidden'):
                model_params[key] = int(model_params[key])
        
        self.model = self.create_graph( 
                                       None, 
                                        device=device,
                                        use_pretrained=use_pretrained,
                                       n_dim_x=n_dim_x,
                                       n_dim_x_p=n_dim_x_p, 
                                       n_dim_x_a=n_dim_x_a,
                                       n_dim_x_b=n_dim_x_b,
                                        n_dim_c=n_dim_c,
                                        n_dim_p=n_dim_p,
                                        n_dim_q=n_dim_q,
                                        **model_params)
    
        
        if restore_path is not None:
            self.model.saver.restore(self.model.sess, '{}.ckpt'.format(restore_path))
            
    def fit(self, 
            X, 
            y=None, 
            X_val=None, 
            y_val=None, 
            batch_size=32, 
            n_epochs=50, 
            patience=10,
            verbose=1,
            use_swa=False,
            seed=21):
                
        sess = self.model.sess
#         np.random.seed(0)
#         tf.set_random_seed(0)
        sess.run(self.model.init)
                        
        train_dl = DataLoader(X, batch_size, shuffle=True, seed=seed)
        test_dl = DataLoader(X_val, batch_size, shuffle=False)
        
        model = self.model
        best_score = np.inf
        best_score_epoch = 0
        best_probs = None
        since_best = 0
        for epoch in range(n_epochs):
            start = timer()
            pbar = tqdm(desc='Trn', total=len(X)+len(X_val)) if verbose > 1 else contextlib.suppress()
            with pbar:
                if epoch > 0:
                    sess.run(model.restore_weight_backups) 
                
                loss = []
                for idx, (batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, batch_q, batch_y, 
                          seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b) in enumerate(train_dl):
#                     print('batch data load time: ', timer()-start)
#                     start = timer()
                    loss_, _ = sess.run([model.loss,
                                        model.train_op],
                                        feed_dict={model.d_X: batch_x,
                                                    model.d_X_p: batch_x_p,
                                                    model.d_X_a: batch_x_a,
                                                    model.d_X_b: batch_x_b,
                                                   model.d_C: batch_c,
                                                   model.d_P: batch_p,
                                                   model.d_Q: batch_q,
                                                    model.d_y: batch_y,
                                                    model.d_seq_lens_x: seq_lens_x,
                                                    model.d_seq_lens_x_p: seq_lens_p,
                                                    model.d_seq_lens_x_a: seq_lens_a,
                                                    model.d_seq_lens_x_b: seq_lens_b})

                    loss.append(loss_)
                    if verbose > 1:
                        pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, np.mean(loss), np.inf))
                        pbar.update(len(batch_x))
#                     print('batch compute time: ', timer()-start)
#                     start = timer()
                # at the end of the epoch, you can run the SWA op which apply the formula defined above
#                 if use_swa:
                sess.run(model.swa_op)
                sess.run(model.save_weight_backups)
                sess.run(model.swa_to_weights)

                trn_loss = np.mean(loss)
                loss, y_true, probs = self.predict(test_dl, epoch=epoch, trn_loss=trn_loss, pbar=pbar)

                score = log_loss(np.array(y_true), np.array(probs))

                if score < best_score:
                    best_score = score
                    best_score_epoch = epoch
                    best_probs = probs
                    since_best = 0
                    model.saver.save(sess, '{}.ckpt'.format(self.ckpt_path))
                else:
                    since_best += 1
                
                if verbose > 1:
                    pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, trn_loss, score))
                    pbar.update(len(X_val))
              
                if verbose < 2:
                    logger.info('[{}] Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f} , Elapsed={}'.format(self.ckpt_path, 
                                                                                    epoch, 
                                                                                    trn_loss, 
                                                                                    score, 
                                                                                    timedelta(seconds=int(timer()-start))))

                if since_best > patience:
                    break                  
  
        if verbose:
            logger.info('[{}] Best score on validation set: {} (epoch {})'.format(self.ckpt_path, best_score, best_score_epoch))
        
        self.best_score = best_score
        self.best_score_epoch = best_score_epoch

        # restore the model state to best checkpoint
        # model.saver.restore(model.sess, '{}.ckpt'.format(self.ckpt_path))
        
        return self

        
    def predict(self, 
                X, 
                y=None, 
                epoch=None, 
                trn_loss=None, 
                pbar=None, 
                verbose=1,
                batch_size=32,
                ckpt_path='best_model',
                seed=21,
                **parameters):            

        if trn_loss is None:
            test_dl = DataLoader(X, batch_size, shuffle=False)
        else:
            test_dl = X
        loss, y_true, probs = [], [], []
        model = self.model
        sess = self.model.sess
#         if pbar is None:
#             pbar = tqdm(desc='Predict', total=len(X)) if verbose and not parallel else contextlib.suppress()
#         with pbar:
        for idx, (batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, batch_q, batch_y, 
                  seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b) in enumerate(test_dl):
            loss_, y_true_, probs_ = sess.run([model.loss,
                                                    model.d_y,
                                                    model.probs],
                                                    feed_dict={model.d_X: batch_x,
                                                                model.d_X_p: batch_x_p,
                                                                model.d_X_a: batch_x_a,
                                                                model.d_X_b: batch_x_b,
                                                               model.d_C: batch_c,
                                                               model.d_P: batch_p,
                                                               model.d_Q: batch_q,
                                                                model.d_y: batch_y,
                                                                model.d_seq_lens_x: seq_lens_x,
                                                                model.d_seq_lens_x_p: seq_lens_p,
                                                                model.d_seq_lens_x_a: seq_lens_a,
                                                                model.d_seq_lens_x_b: seq_lens_b})

            loss.append(loss_)
            y_true += y_true_.tolist()
            probs += probs_.tolist()
                
#                 if verbose and not parallel:
#                     if epoch is not None:
#                         pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, trn_loss, np.mean(loss)))
#                     else:
#                         pbar.set_description('Predict Tst-Loss={:.3f}'.format(np.mean(loss)))
#                     pbar.update(len(batch_x))
            
        # if trn_loss is None:
            # tf.reset_default_graph()
            
        return loss, np.array(y_true), probs
        
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
        
        def gelu_fast(_x):
            return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
        
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

    #                 fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='fw_x')
    #                 bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='bw_x')
    #                 _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
    #                                                                       d_X, sequence_length=d_seq_lens_x, dtype=tf.float32)
    #                 X = tf.reshape(tf.concat([fw_state, bw_state], axis=-1), (-1, 2*n_hidden_x))

    #                 X = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x))(d_X)

    #                 fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='fw_p')
    #                 bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='bw_p')
    #                 _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
    #                                                                       d_X_p, sequence_length=d_seq_lens_x_p, dtype=tf.float32)
    #                 X_p = tf.reshape(tf.concat([fw_state, bw_state], axis=-1), (-1, 2*n_hidden_x_p))

    #                 fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='fw_a')
    #                 bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='bw_a')
    #                 _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
    #                                                                       d_X_a, sequence_length=d_seq_lens_x_a, dtype=tf.float32)
    #                 X_a = tf.reshape(tf.concat([fw_state, bw_state], axis=-1), (-1, 2*n_hidden_x_a))

    #                 fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='fw_b')
    #                 bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden_x, name='bw_b')
    #                 _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
    #                                                                       d_X_b, sequence_length=d_seq_lens_x_b, dtype=tf.float32)
    #                 X_b = tf.reshape(tf.concat([fw_state, bw_state], axis=-1), (-1, 2*n_hidden_x_b))
                    gru_init = tf.keras.initializers.glorot_uniform(seed=seed)
                    gru_init2 = tf.keras.initializers.orthogonal(seed=seed)

                    X = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                               kernel_initializer=gru_init,
                                                                               recurrent_initializer=gru_init2,
                                                                               return_sequences=True))(d_X)
                    X_s, X = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X)

                    X_p = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                                 kernel_initializer=gru_init,
                                                                                 recurrent_initializer=gru_init2,
                                                                                 return_sequences=True))(d_X_p)
                    X_p_s, X_p = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X_p)

                    X_a = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                                 kernel_initializer=gru_init,
                                                                                 recurrent_initializer=gru_init2,
                                                                                 return_sequences=True))(d_X_a)
                    X_a_s, X_a = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X_a)

                    X_b = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                                 kernel_initializer=gru_init,
                                                                                 recurrent_initializer=gru_init2,
                                                                                 return_sequences=True))(d_X_b)
                    X_b_s, X_b = tf.keras.layers.Lambda(lambda t: [t, t[:,-1]])(X_b)

                    G_s = self.coattn(X_s, X_p_s, d_seq_lens_x, d_seq_lens_x_p, n_hidden_x, seed)

                    A_s = self.coattn(X_s, X_a_s, d_seq_lens_x, d_seq_lens_x_a, n_hidden_x, seed)
                    A_s = self.selfattn(A_s, d_seq_lens_x, 2*n_hidden_x, seed)
    #                 A_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
    #                                                                              kernel_initializer=gru_init,
    #                                                                              recurrent_initializer=gru_init2,
    #                                                                              return_sequences=False))(A_s)

                    B_s = self.coattn(X_s, X_b_s, d_seq_lens_x, d_seq_lens_x_b, n_hidden_x, seed)
                    B_s = self.selfattn(B_s, d_seq_lens_x, 2*n_hidden_x, seed)
    #                 B_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
    #                                                                              kernel_initializer=gru_init,
    #                                                                              recurrent_initializer=gru_init2,
    #                                                                              return_sequences=False))(B_s)

                    G_s = self.selfattn(G_s, d_seq_lens_x, 2*n_hidden_x, seed)
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

                    feats = [G_s, A_s, B_s]
                    if use_pretrained:
                        feats += [P, C, Q]

                    X = tf.concat(feats, axis=-1, name='agg_feats')

    #                 if use_pretrained:
    #                     X = tf.keras.layers.Dropout(dropout_rate_ff, seed=7)(X)

    #                     X = tf.keras.layers.Dense(n_hidden_ff, activation=None, kernel_initializer=dense_init,
    #                                               kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_ff))(X)
    #                     X = tf.layers.batch_normalization(X)
    #                     X = activ(X)

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
    
    def coattn(self, X, Q, X_lens, Q_lens, n_hidden_x, seed=None):
        dense_init = tf.keras.initializers.glorot_normal(seed=seed)
        
        A = tf.matmul(X, Q, transpose_b=True)
        
        masks_x = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks_q = tf.cast(tf.sequence_mask(Q_lens), tf.float32)
        masks_x = tf.expand_dims(masks_x, axis=-1)
        masks_q = tf.expand_dims(masks_q, axis=-1)
        masks = tf.matmul(masks_x, masks_q, transpose_b=True)
        
        paddings = tf.ones_like(A)*(-2**32+1)
        A_ = tf.where(tf.equal(masks, 0), paddings, A) 

        S_s = tf.matmul(tf.nn.softmax(A_, axis=2), Q)
        masks = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        S_s = S_s * masks
        
        
        S_q = tf.matmul(tf.nn.softmax(A_, axis=1), X, transpose_a=True)
        masks = tf.cast(tf.sequence_mask(Q_lens), tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        S_q = S_q * masks

        C_s = tf.matmul(tf.nn.softmax(A_, axis=2), S_q)
        masks = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        C_s = C_s * masks

        gru_init = tf.keras.initializers.glorot_uniform(seed=seed)
        gru_init2 = tf.keras.initializers.orthogonal(seed=seed)
        
        C_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x, 
                                                                     kernel_initializer=gru_init,
                                                                     recurrent_initializer=gru_init2,
                                                                     return_sequences=True))(C_s)

        U_s = tf.concat([C_s, S_s], axis=-1, name='document_context')
        
        U_s = tf.keras.layers.Dense(2*n_hidden_x, activation='tanh', kernel_initializer=dense_init)(U_s)
        
        return U_s
    
    def selfattn(self, X, X_lens, n_hidden_x, seed=None):
        dense_init = tf.keras.initializers.glorot_normal(seed=seed)
        
        a_s = tf.keras.layers.Dense(n_hidden_x, activation='tanh', kernel_initializer=dense_init)(X)
        a_s = tf.keras.layers.Dense(1, activation=None, kernel_initializer=dense_init)(a_s)
        
        masks_x = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks_x = tf.expand_dims(masks_x, axis=-1)
        
        paddings = tf.ones_like(a_s)*(-2**32+1)
        a_s = tf.where(tf.equal(masks_x, 0), paddings, a_s)
        
        a_s = tf.nn.softmax(a_s, axis=1)
        masks = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        a_s = a_s * masks
        
        G_s = tf.matmul(a_s, X, transpose_a=True, name='G_s_matmul')
        
        G_s = tf.reshape(G_s, [-1, n_hidden_x])
        
        return G_s
    
    @staticmethod
    def mean_featurizer(X_bert, X_plus, X_pretrained, y_true, *args, **kwargs):
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

            x = np.array(row.bert)[:seq_len]
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
            
            q = np.array(row.cls).reshape(-1)
            batch_q.append(q)
            
            y = y_true.loc[idx].values
            
            batch_y.append(y)

        return pd.DataFrame([batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, \
            batch_q, batch_y, seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b]).T