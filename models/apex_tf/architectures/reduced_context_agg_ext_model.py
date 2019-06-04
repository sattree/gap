import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import contextlib
from timeit import default_timer as timer

from sklearn.metrics import classification_report, log_loss
from attrdict import AttrDict

from ..base_dataloader import DataLoader
from ..mean_pool_model import MeanPoolModel

class ReducedContextAggExtModel(MeanPoolModel):
    def init_graph(self, X, batch_size, ckpt=None, device='gpu', **model_params):
        self.batch_size = batch_size

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
                                       n_dim_x=n_dim_x,
                                       n_dim_x_p=n_dim_x_p, 
                                       n_dim_x_a=n_dim_x_a,
                                       n_dim_x_b=n_dim_x_b,
                                        n_dim_c=n_dim_c,
                                        n_dim_p=n_dim_p,
                                        n_dim_q=n_dim_q,
                                        device=device,
                                        **model_params)
    
        self.init = tf.initializers.global_variables()
        TF_CONFIG = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1),
              allow_soft_placement=True)
        self.sess = sess = tf.Session(config=TF_CONFIG)
        self.saver = tf.train.Saver()
        if ckpt is not None:
            self.saver.restore(self.sess, 'tmp/{}.ckpt'.format(self.ckpt_path))
            
    def fit(self, 
            X, 
            y=None, 
            X_val=None, 
            y_val=None, 
            batch_size=32, 
            n_epochs=50, 
            patience=10,
            verbose=1, 
            ckpt_path='best_model',
            **model_params):
                
        sess = self.sess
#         np.random.seed(0)
#         tf.set_random_seed(0)
        sess.run(self.init)
                        
        train_dl = DataLoader(X, batch_size, shuffle=True, seed=21)
        test_dl = DataLoader(X_val, batch_size, shuffle=False, seed=21)
        
        model = self.model
        best_score = np.inf
        best_probs = None
        since_best = 0
        start = timer()
        for epoch in range(n_epochs):
            pbar = tqdm(desc='Trn', total=len(X)+len(X_val)) if verbose else contextlib.suppress()
            with pbar:
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
                    if verbose:
                        pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, np.mean(loss), np.inf))
                        pbar.update(len(batch_x))
#                     print('batch compute time: ', timer()-start)
#                     start = timer()
                                        
                trn_loss = np.mean(loss)
                loss, y_true, probs = self.predict(test_dl, epoch=epoch, trn_loss=trn_loss, pbar=pbar)

                score = log_loss(np.array(y_true), np.array(probs))

                if score < best_score:
                    best_score = score
                    best_probs = probs
                    since_best = 0
                    self.saver.save(sess, 'tmp/{}.ckpt'.format(self.ckpt_path))
                else:
                    since_best += 1
                
                if verbose:
                    pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, trn_loss, score))
                    pbar.update(len(X_val))
                    
                if since_best > patience:
                    break
                    
              
                
        if verbose:
            print('Best score on validation set: ', best_score)
        
        self.best_score = best_score
        tf.reset_default_graph()
        
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
                **parameters):
        if trn_loss is None:
            self.init_graph(X, batch_size, ckpt=self.ckpt_path, device='cpu', **parameters)

        if trn_loss is None:
            test_dl = DataLoader(X, self.batch_size)
        else:
            test_dl = X
        loss, y_true, probs = [], [], []
        model = self.model
        for idx, (batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, batch_q, batch_y, 
                  seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b) in enumerate(test_dl):
            loss_, y_true_, probs_ = self.sess.run([self.model.loss,
                                                    self.model.d_y,
                                                    self.model.probs],
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
            
        if trn_loss is None:
            tf.reset_default_graph()
            
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
                    seed=None,
                    device='gpu'):
        
        def gelu_fast(_x):
            return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
        
        activ = tf.nn.relu
#         activ = gelu_fast
            #         np.random.seed(0)
    #         tf.set_random_seed(0)
        tf.reset_default_graph()
        with tf.device('/{}:0'.format(device)):
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
                dense_init = tf.keras.initializers.glorot_normal(seed=21)

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
                gru_init = tf.keras.initializers.glorot_uniform(seed=9)
                gru_init2 = tf.keras.initializers.orthogonal(seed=7)
    
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

                
#                 A = tf.matmul(X_s, X_p_s, transpose_b=True)
#                 S_s = tf.matmul(tf.nn.softmax(A, axis=1), X_p_s)
#                 S_q = tf.matmul(tf.nn.softmax(A, axis=2), X_s, transpose_a=True)
                
                
#                 C_s = tf.matmul(tf.nn.softmax(A, axis=1), S_q)
#                 C_s = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(n_hidden_x//2, return_sequences=True))(C_s)
                
#                 U_s = tf.concat([C_s, S_s], axis=-1, name='document_context')
                
# #                 a_s = tf.keras.layers.Dense(n_hidden_x, activation='tanh', kernel_initializer=dense_init)(U_s)
#                 a_s = tf.keras.layers.Dense(1, activation=None, kernel_initializer=dense_init)(U_s)
#                 a_s = tf.nn.softmax(a_s, axis=1)
#                 G_s = tf.matmul(a_s, U_s, transpose_a=True, name='G_s_matmul')
#                 G_s = tf.reshape(G_s, [-1, 4*n_hidden_x])
                
                P = tf.keras.layers.Dropout(dropout_rate_p, seed=7)(d_P)

                P = tf.keras.layers.Dense(n_hidden_p, activation=None, kernel_initializer=dense_init,
                                          kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_p))(P)
                P = tf.layers.batch_normalization(P)
                P = activ(P)

                C = tf.keras.layers.Dropout(dropout_rate_c, seed=7)(d_C)

                C = tf.keras.layers.Dense(n_hidden_c, activation=None, kernel_initializer=dense_init,
                                          kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_c))(C)
                C = tf.layers.batch_normalization(C)
                C = activ(C)
                
                Q = tf.keras.layers.Dropout(dropout_rate_q, seed=7)(d_Q)

                Q = tf.keras.layers.Dense(n_hidden_q, activation=None, kernel_initializer=dense_init,
                                          kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_q))(Q)
                Q = tf.layers.batch_normalization(Q)
                Q = activ(Q)
                
                feats = [X, X_p, X_a, X_b]
                if use_pretrained:
                    feats += [P, C, Q]
                    
                X = tf.concat(feats, axis=-1, name='agg_feats')
                
                if use_pretrained:
                    X = tf.keras.layers.Dropout(dropout_rate_ff, seed=7)(X)

                    X = tf.keras.layers.Dense(n_hidden_ff, activation=None, kernel_initializer=dense_init,
                                              kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_ff))(X)
                    X = tf.layers.batch_normalization(X)
                    X = activ(X)
                    
                X = tf.keras.layers.Dropout(dropout_rate_fc, seed=7)(X)

                y_hat = tf.keras.layers.Dense(3, name = 'output', kernel_initializer=dense_init,
                                              kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_fc))(X)

            with tf.name_scope('loss'):
                probs = tf.nn.softmax(y_hat, axis=-1)
                # label smoothing works
                loss = tf.losses.softmax_cross_entropy(d_y, logits=y_hat, label_smoothing=label_smoothing)
                loss = tf.reduce_mean(loss)

            with tf.name_scope('optimizer'):
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
                learning_rate = 0.005
                learning_rate = tf.train.cosine_decay_restarts(
                            learning_rate,
                            global_step,
                            500,
                            t_mul=1,
                            m_mul=1,
                            alpha=0.01,
                            name=None
                        )

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss, global_step=global_step)

        return AttrDict(locals())
    
    @staticmethod
    def mean_featurizer(X_bert, X_plus, X_pretrained, y_true):
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
            x = np.array(row.bert)
            p_vec = x[row.pronoun_offset_token:row.pronoun_offset_token+1]
            a_vec = x[row.a_span[0]:row.a_span[1]+1]
            b_vec = x[row.b_span[0]:row.b_span[1]+1]
            
            x = np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1]))))
            x_p = np.vstack((p_vec, np.zeros((max_len_tok-p_vec.shape[0], p_vec.shape[1]))))
            x_a = np.vstack((a_vec, np.zeros((max_len_tok-a_vec.shape[0], a_vec.shape[1]))))
            x_b = np.vstack((b_vec, np.zeros((max_len_tok-b_vec.shape[0], b_vec.shape[1]))))

            batch_x.append(x)
            batch_x_p.append(x_p)
            batch_x_a.append(x_a)
            batch_x_b.append(x_b)
            
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
            
            seq_len = len(row.tokens)

            seq_lens_x.append(seq_len)
            seq_lens_p.append(p_vec.shape[0])
            seq_lens_a.append(a_vec.shape[0])
            seq_lens_b.append(b_vec.shape[0])
            
            y = y_true.loc[idx].values
            
            batch_y.append(y)

        return pd.DataFrame([batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, batch_q, batch_y, seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b]).T