import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, log_loss

from ..base_dataloader import DataLoader
from ..mean_pool_model import MeanPoolModel

class ContextAggModel(MeanPoolModel):
    def init_graph(self, X, batch_size, **model_params):
        self.batch_size = batch_size

        n_dim_x_p = len(X[0].values[0][0])
        n_dim_x_a = len(X[1].values[0][0])
        n_dim_x_b = len(X[2].values[0][0])
        n_dim_q = len(X[3].values[0])
        n_dim_p = len(X[4].values[0])
        n_dim_c = len(X[5].values[0])
#         n_dim_c = len(X[3].values[0][0])
        
        for key, val in model_params.items():
            if key.startswith('n_hidden'):
                model_params[key] = int(model_params[key])
        
        self.model = self.create_graph( 
                                       None, 
                                       n_dim_x_p=n_dim_x_p, 
                                        n_dim_x_a=n_dim_x_a,
                                        n_dim_x_b=n_dim_x_b,
                                       n_dim_q=n_dim_q, 
                                       n_dim_p=n_dim_p,
                                       n_dim_c=n_dim_c,
                                        **model_params)
    
        self.init = tf.initializers.global_variables()
        self.sess = sess = tf.Session()
        self.saver = tf.train.Saver()
        
    def fit(self, 
            X, 
            y=None, 
            X_val=None, 
            y_val=None, 
            batch_size=32, 
            n_epochs=30, 
            patience=5,
            verbose=1, 
            **model_params):
        
        start = timer()
        
        sess = self.sess
#         np.random.seed(0)
#         tf.set_random_seed(0)
        sess.run(self.init)
        
        self.saver = tf.train.Saver()
                
        train_dl = DataLoader(X, batch_size, shuffle=True, seed=21)
        test_dl = DataLoader(X_val, batch_size, shuffle=False, seed=21)
        
        model = self.model
        best_score = 2
        best_probs = None
        since_best = 0
        for epoch in range(n_epochs):
            pbar = tqdm(desc='Trn', total=len(X)+len(X_val)) if verbose else contextlib.suppress()
            with pbar:
                loss = []
                for idx, (batch_x_p, batch_x_a, batch_x_b, batch_q, batch_p, batch_c, batch_y, seq_lens, seq_lens_p, seq_lens_a, seq_lens_b) in enumerate(train_dl):
                    if len(batch_x_p) == 1:
                        continue
                    loss_, _ = sess.run([model.loss,
                                        model.train_op],
                                        feed_dict={model.d_X_p: batch_x_p,
                                                   model.d_X_a: batch_x_a,
                                                   model.d_X_b: batch_x_b,
                                                    model.d_Q: batch_q,
                                                    model.d_P: batch_p,
                                                    model.d_C: batch_c,
                                                    model.d_y: batch_y,
                                                    model.d_seq_lens: seq_lens,
                                                  model.d_seq_lens_p: seq_lens_p,
                                                  model.d_seq_lens_a: seq_lens_a,
                                                  model.d_seq_lens_b: seq_lens_b})

                    loss.append(loss_)
                    if verbose:
                        pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, np.mean(loss), np.inf))
                        pbar.update(len(batch_x_p))
                
                trn_loss = np.mean(loss)
                loss, y_true, probs = self.predict(test_dl, epoch=epoch, trn_loss=trn_loss, pbar=pbar)

                score = log_loss(np.array(y_true), np.array(probs))

                if score < best_score:
                    best_score = score
                    best_probs = probs
                    since_best = 0
                    self.saver.save(sess, 'tmp/best_model.ckpt')
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
        self.saver.restore(self.sess, 'tmp/best_model.ckpt')
        
        return self

        
    def predict(self, 
                X, 
                y=None, 
                epoch=None, 
                trn_loss=None, 
                pbar=None, 
                verbose=1):
        # if verbose and pbar is None:
        #     pbar_ = pbar = tqdm(desc='Predict', total=len(X))
        # else:
        #    pbar_ = contextlib.suppress()
            
        # with pbar_:
        if trn_loss is None:
            test_dl = DataLoader(X, self.batch_size)
        else:
            test_dl = X
        loss, y_true, probs = [], [], []
        for idx, (batch_x_p, batch_x_a, batch_x_b, batch_q, batch_p, batch_c, batch_y, seq_lens, seq_lens_p, seq_lens_a, seq_lens_b) in enumerate(test_dl):
            if len(batch_x_p) == 1:
                        continue
            loss_, y_true_, probs_ = self.sess.run([self.model.loss,
                                                    self.model.d_y,
                                                    self.model.probs],
                                                    feed_dict={self.model.d_X_p: batch_x_p,
                                                   self.model.d_X_a: batch_x_a,
                                                   self.model.d_X_b: batch_x_b,
                                                                self.model.d_Q: batch_q,
                                                                self.model.d_P: batch_p,
                                                                self.model.d_C: batch_c,
                                                                self.model.d_y: batch_y,
                                                                self.model.d_seq_lens: seq_lens,
                                                  self.model.d_seq_lens_p: seq_lens_p,
                                                  self.model.d_seq_lens_a: seq_lens_a,
                                                  self.model.d_seq_lens_b: seq_lens_b})

            loss.append(loss_)
            y_true += y_true_.tolist()
            probs += probs_.tolist()

            # if verbose:
            #    if trn_loss is not None:
            #        pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, trn_loss, np.mean(loss)))
            #    else:
            #        pbar.set_description('Predict, Loss={:.3f}'.format(np.mean(loss)))
            #
            #    pbar.update(len(batch_x))
            
        return loss, np.array(y_true), probs
    
    def create_graph(self,
                     batch_size=None, 
                     seq_len=None,
                     n_dim_x_p=1024,
                     n_dim_x_a=1024,
                     n_dim_x_b=1024,
                     n_dim_q=1024, 
                     n_dim_p=1024,
                     n_dim_c=1024,
                     n_hidden_x=1024, 
                     n_hidden_x_p=1024,
                     n_hidden_x_a=1024,
                     n_hidden_x_b=1024,
                     n_hidden_q=1024, 
                     n_hidden_p=1024,
                     n_hidden_c=1024,
                     dropout_rate_x=0.5, 
                     dropout_rate_p=0.5, 
                     dropout_rate_q=0.5, 
                     dropout_rate_c=0.5,
                     dropout_rate_fc=0.5,
                     reg_x=0.01,
                     reg_q=0.01,
                     reg_p=0.01,
                     reg_c=0.01,
                     reg_fc=0.01,
                     label_smoothing=0.1,
                     data_type=tf.float32, 
                     activation='tanh'):
        
        def gelu_fast(_x):
            return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
        
        activ = tf.nn.relu
#         activ = gelu_fast
        
#         np.random.seed(0)
#         tf.set_random_seed(0)
        tf.reset_default_graph()
        
#         d_train_c = tf.constant(X_train, shape=[2000, 512, 1024])

        d_X_p = tf.placeholder(data_type, [batch_size, None, n_dim_x_p])
        d_X_a = tf.placeholder(data_type, [batch_size, None, n_dim_x_a])
        d_X_b = tf.placeholder(data_type, [batch_size, None, n_dim_x_b])
        d_Q = tf.placeholder(data_type, [batch_size, n_dim_q])
        d_P = tf.placeholder(data_type, [batch_size, n_dim_p])
        d_C = tf.placeholder(data_type, [batch_size, n_dim_c])
#         d_C = tf.placeholder(data_type, [batch_size, None, n_dim_c])

        d_y = tf.placeholder(tf.int32, [batch_size, 3])
        d_seq_lens = tf.placeholder(tf.int32, [batch_size])
        d_seq_lens_p = tf.placeholder(tf.int32, [batch_size])
        d_seq_lens_a = tf.placeholder(tf.int32, [batch_size])
        d_seq_lens_b = tf.placeholder(tf.int32, [batch_size])

#         d_dropout_l0 = tf.placeholder(tf.float32, None)
#         d_dropout_l1 = tf.placeholder(tf.float32, None)
#         d_dropout_l2 = tf.placeholder(tf.float32, None)
#         d_dropout_l3 = tf.placeholder(tf.float32, None)

        with tf.name_scope('dense_concat_layers'):
#             n_hidden = 1024
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_x_p, name='fw1')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_x_p, name='bw1')
            _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, 
                                                                      d_X_p, sequence_length=d_seq_lens_p, dtype=tf.float32)
            X_p = tf.reshape(tf.concat([fw_state.h, bw_state.h], axis=-1), (-1, 2*n_hidden_x_p))
            
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_x_a, name='fw2')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_x_a, name='bw2')
            _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, 
                                                                      d_X_a, sequence_length=d_seq_lens_a, dtype=tf.float32)
            X_a = tf.reshape(tf.concat([fw_state.h, bw_state.h], axis=-1), (-1, 2*n_hidden_x_a))
            
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_x_b, name='fw3')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_x_b, name='bw3')
            _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, 
                                                                      d_X_b, sequence_length=d_seq_lens_b, dtype=tf.float32)
            X_b = tf.reshape(tf.concat([fw_state.h, bw_state.h], axis=-1), (-1, 2*n_hidden_x_b))

            dense_init = tf.keras.initializers.glorot_normal(seed=21)
            

#             X_p = TransformerCorefModel().pool_and_attend(d_X_p, d_seq_lens_p, n_dim_x_p, n_hidden_x_p, 20)[0]
#             X_a = TransformerCorefModel().pool_and_attend(d_X_a, d_seq_lens_a, n_dim_x_a, n_hidden_x_a, 20)[0]
#             X_b = TransformerCorefModel().pool_and_attend(d_X_b, d_seq_lens_b, n_dim_x_b, n_hidden_x_b, 20)[0]
            
            X = tf.concat([X_p, X_a, X_b], axis=-1)
            X = tf.keras.layers.Dropout(dropout_rate_x, seed=7)(X)

            X = tf.keras.layers.Dense(n_hidden_x, activation=None, kernel_initializer=dense_init,
                                      kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_x))(X)
            X = tf.layers.batch_normalization(X)
            X = activ(X)

            Q = tf.keras.layers.Dropout(dropout_rate_q, seed=7)(d_Q)
            
            Q = tf.keras.layers.Dense(n_hidden_q, activation=None, kernel_initializer=dense_init,
                                      kernel_regularizer = tf.contrib.layers.l1_regularizer(reg_q))(Q)
            Q = tf.layers.batch_normalization(Q)
            Q = activ(Q)
            
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
            
#             C = TransformerCorefModel().pool_and_attend(d_C, d_seq_lens, 1024, n_hidden_c, 512)[0]
            
            X = tf.concat([X, P, Q, C], axis=-1)
            X = tf.keras.layers.Dropout(dropout_rate_fc, seed=7)(X)

#             X = tf.layers.dense(X, 128, activation=None)
#             X = tf.layers.batch_normalization(X)
#             X = tf.nn.relu(X)
#             X = tf.keras.layers.Dropout(0.5)(X)

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
        batch_x_p = []
        batch_x_a = []
        batch_x_b = []
        batch_q = []
        batch_p = []
        batch_c = []
        seq_lens = []
        seq_lens_p = []
        seq_lens_a = []
        seq_lens_b = []
        batch_y = []
        max_len = 512
        max_len_tok = 20
        for idx, row in tqdm(X_bert.iterrows(), total=len(X_bert)):
            plus_row = X_plus.loc[idx]
            x = np.array(row.bert)
            q = np.array(plus_row.plus)
            p = X_pretrained.loc[idx].pretrained
            c = np.array(row.cls).reshape(-1)
#             c = np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1]))))
            pronoun_vec = x[row.pronoun_offset_token:row.pronoun_offset_token+1]
            a_vec = x[row.a_span[0]:row.a_span[1]+1]
            b_vec = x[row.b_span[0]:row.b_span[1]+1]
#             print(idx, a_vec.shape, b_vec.shape)
#             x = np.hstack((np.mean(pronoun_vec, axis=0), np.mean(a_vec, axis=0), np.mean(b_vec, axis=0))).reshape(-1)
            
            x_p = np.vstack((pronoun_vec, np.zeros((max_len_tok-pronoun_vec.shape[0], pronoun_vec.shape[1]))))
            x_a = np.vstack((a_vec, np.zeros((max_len_tok-a_vec.shape[0], a_vec.shape[1]))))
            x_b = np.vstack((b_vec, np.zeros((max_len_tok-b_vec.shape[0], b_vec.shape[1]))))
        
            pronoun_vec = q[plus_row.pronoun_offset_token:plus_row.pronoun_offset_token+1]
            a_vec = q[plus_row.a_span[0]:plus_row.a_span[0]+1]
            b_vec = q[plus_row.b_span[0]:plus_row.b_span[0]+1]
            q = np.hstack((np.max(pronoun_vec, axis=0), np.max(a_vec, axis=0), np.max(b_vec, axis=0))).reshape(-1)
            
            y = y_true.loc[idx].values
            
            seq_len = len(row.tokens)
            
            batch_x_p.append(x_p)
            batch_x_a.append(x_a)
            batch_x_b.append(x_b)
            batch_q.append(q)
            batch_p.append(p)
            batch_c.append(c)
            batch_y.append(y)
            seq_lens.append(seq_len)
            seq_lens_p.append(pronoun_vec.shape[0])
            seq_lens_a.append(a_vec.shape[0])
            seq_lens_b.append(b_vec.shape[0])

        return pd.DataFrame([batch_x_p, batch_x_a, batch_x_b, batch_q, batch_p, batch_c, batch_y, seq_lens, seq_lens_p, seq_lens_a, seq_lens_b]).T