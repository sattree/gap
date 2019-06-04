import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from attrdict import AttrDict
from sklearn.metrics import classification_report, log_loss
import functools
import gc
import csv
import contextlib
from timeit import default_timer as timer

import pandas as pd
import numpy as np

from .base_dataloader import DataLoader

tf.logging.set_verbosity(tf.logging.WARN)

class MeanPoolModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, ckpt_path='best_model', **kwargs):
        self.ckpt_path = ckpt_path
        self.init_graph(*args, **kwargs)
        
    def init_graph(self, X, batch_size, **model_params):
        self.batch_size = batch_size

        n_dim_x = len(X[0].values[0])
        n_dim_q = len(X[1].values[0])
        n_dim_p = len(X[2].values[0])
        n_dim_c = len(X[3].values[0])
#         n_dim_c = len(X[3].values[0][0])
        
        for key, val in model_params.items():
            if key.startswith('n_hidden'):
                model_params[key] = int(model_params[key])
        
        self.model = self.create_graph( 
                                       None, 
                                       n_dim_x=n_dim_x, 
                                       n_dim_q=n_dim_q, 
                                       n_dim_p=n_dim_p,
                                       n_dim_c=n_dim_c,
                                        **model_params)
    
        self.init = tf.initializers.global_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
    
    def fit(self, 
            X, 
            y=None, 
            X_val=None, 
            y_val=None, 
            batch_size=32, 
            n_epochs=50, 
            patience=5, 
            dropout_rate_l0=0.5, 
            dropout_rate_l1=0.5, 
            dropout_rate_l2=0.5, 
            dropout_rate_l3=0.5, 
            verbose=1):
        
        start = timer()

        sess = self.sess
        sess.run(init)
        
        print('model creation time: ', timer()-start)
        start = timer()
        
        train_dl = DataLoader(X, batch_size, shuffle=True)
        
        print('data loader time: ', timer()-start)
        start = timer()
        
        model = self.model
        
        best_score = 2
        best_probs = None
        since_best = 0
        for epoch in range(n_epochs):
            pbar = tqdm(desc='Trn', total=len(X)+len(X_val)) if verbose else contextlib.suppress()
            with pbar:
                loss = []
                for idx, (batch_x, batch_q, batch_p, batch_y, seq_lens) in enumerate(train_dl):
                    loss_, _ = sess.run([model.loss,
                                        model.train_op],
                                        feed_dict={model.d_X: batch_x,
                                                    model.d_Q: batch_q,
                                                    model.d_P: batch_p,
                                                    model.d_y: batch_y,
                                                    model.d_seq_lens: seq_lens,
                                                    model.d_dropout_l0: dropout_rate_l0,
                                                    model.d_dropout_l1: dropout_rate_l1,
                                                    model.d_dropout_l2: dropout_rate_l2,
                                                    model.d_dropout_l3: dropout_rate_l3})

                    loss.append(loss_)
                    if verbose:
                        pbar.set_description('Trn {:2d}, Loss={:.3f}, Val-Loss={:.3f}'.format(epoch, np.mean(loss), np.inf))
                        pbar.update(len(batch_x))
                
                trn_loss = np.mean(loss)
                loss, y_true, probs = self.predict(X_val, epoch=epoch, trn_loss=trn_loss, pbar=pbar)

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
        test_dl = DataLoader(X, self.batch_size)
        loss, y_true, probs = [], [], []
        for idx, (batch_x, batch_q, batch_p, batch_y, seq_lens) in enumerate(test_dl):
            loss_, y_true_, probs_ = self.sess.run([self.model.loss,
                                                    self.model.d_y,
                                                    self.model.probs],
                                                    feed_dict={self.model.d_X:batch_x,
                                                                self.model.d_Q: batch_q,
                                                                self.model.d_P: batch_p,
                                                                self.model.d_y: batch_y,
                                                                self.model.d_seq_lens: seq_lens,
                                                                self.model.d_dropout_l0: 0.0, 
                                                                self.model.d_dropout_l1: 0.0,
                                                                self.model.d_dropout_l2: 0.0,
                                                                self.model.d_dropout_l3: 0.0})

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
    
    def train_evaluate(self, 
                       X, 
                       X_val=None, 
                       X_tst=None, 
                       batch_size=32, 
                       verbose=1,
                       return_probs=False,
                       n_trials=None,
                       **parameters):
        
        self.init_graph(X, batch_size, device='gpu', **parameters)
        
        self.fit(X, 
                 X_val=X_val, 
                 verbose=verbose, 
                 batch_size=batch_size, 
                 **parameters)
        
        _, y_true_val, probs_val = self.predict(X_val, batch_size, verbose=verbose, **parameters)
        
        probs_tst = None
        if X_tst is not None:
            _, y_true_tst, probs_tst = self.predict(X_tst, batch_size, verbose=verbose, **parameters)
           
        if verbose:
            print('Validation score: ', self.best_score)
            if X_tst is not None:
                print('Test score: ', log_loss(y_true_tst, probs_tst))
        
        if return_probs:
            return AttrDict(locals())
        
        return -self.best_score
            
    def train_evaluate_cv(self, 
                          X, 
                          X_val=None, 
                          X_tst=[], 
                          n_folds=5, 
                          n_trials=1,
                          batch_size=32, 
                          seed=None,
                          verbose=1, 
                          return_probs=True, 
                          **parameters):
                
        if X_val is not None:
            X = pd.concat([X, X_val], axis = 0).reset_index(drop=True)
        
#         self.init_graph(X, batch_size, **parameters)

        folds = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        probs = [[] for X in X_tst]
        y_true = None
        scores = []
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
            start = timer()
            X_trn, X_val = X.loc[train_index], X.loc[valid_index]
            
            scores_ = []
            for i in range(n_trials):
                self.init_graph(X, batch_size, **parameters)
                self.fit(X_trn, X_val=X_val, verbose=verbose, batch_size=batch_size, **parameters)
                scores_.append(self.best_score)
                for j, X_ in enumerate(X_tst):
                    _, y_true_, probs_ = self.predict(X_, batch_size, verbose=verbose, **parameters)
                    if j==0:
                        y_true = y_true_
                    probs[j].append(probs_)
                    
            scores.append(np.mean(scores_))
            if verbose:
                print('Fold {} done in {}'.format(fold_n, timer()-start))
            start = timer()
        
        probs_raw = probs.copy()
        probs = np.mean(probs[0], axis=0)
        score = log_loss(y_true, probs)
        
        if verbose > 0:
            print('Validation scores: ', scores)
            print('Mean val score: {} +/- {}'.format(np.mean(scores), np.std(scores)))
            if X_tst is not None:
                print('Test score: ', score)
        
        if return_probs:
            return AttrDict(locals())
        
        return -score
    
    def repeated_cv(self, 
                    X, 
                    X_val=None, 
                    X_tst=None, 
                    n_trials=5,
                    seed=None,
                    return_probs=True, 
                    **kwargs):
        
        if seed is None:
            seed = [seed]*n_trials
        probs = []
        probs_raw = []
        scores = []
        for i in range(n_trials):
            start = timer()
            res = self.train_evaluate_cv(X, 
                                         X_val, 
                                         X_tst,
                                         seed=seed[i],
                                         return_probs=True, 
                                         **kwargs)
            probs.append(res.probs)
            probs_raw += res.probs_raw
            y_true = res.y_true
            scores.append(res.score)
            if verbose:
                print('Trial {} done in {}'.format(i, timer()-start))
            start = timer()
        probs = np.mean(probs, axis=0)
        
        if return_probs:
            print('Repeated bag scores: ', scores)
            print('Repeated bag mean: {} +/- {}'.format(np.mean(scores), np.std(scores)))
            print('CV Bagged score: ', log_loss(y_true, probs))
            return AttrDict(locals())
        
        return -log_loss(y_true, probs)
    
    def hyperopt(self,
                 fn,
                 X, 
                 X_val=None,
                 X_tst=None,
                 n_trials=5,
                 init_points=5,
                 n_iter=20,
                 batch_size=64,
                 params={'n_hidden_l1': (32, 256),
                        'n_hidden_l2': (32, 256),
                        'dropout_rate': (.1, .8)}, 
                 verbose=0,
                 seed=None,
                 **kwargs):
        self.bo = bo = BayesianOptimization(
                        f=functools.partial(fn, 
                                            X, 
                                            X_val, 
                                            X_tst, 
                                            n_trials=n_trials,
                                            batch_size=batch_size, 
                                            verbose=verbose-1,
                                            seed=seed,
                                            return_probs=False),
                        pbounds=params,
                        # random_state=1,
                        **kwargs
                    )
        
        bo.maximize(
                        init_points=init_points,
                        n_iter=n_iter,
                    )
        
        print(bo.max)
    
    def create_graph(self,
                    batch_size=None, 
                    seq_len=None, 
                    n_hidden_x=1024, 
                    n_hidden_q=1024, 
                    n_hidden_p=1024,
                     n_hidden_l1=59,
                     n_hidden_l2=59,
                     n_hidden_l3=59,
                     l2_l1=0.05,
                     l2_l2=0.05,
                     l2_l3=0.05,
                     label_smoothing=0.1,
                    data_type=tf.float32, 
                    activation='tanh'):
        
        # np.random.seed(0)
        # tf.set_random_seed(0)
        tf.reset_default_graph()

        d_X = tf.placeholder(data_type, [batch_size, n_hidden_x])
        d_Q = tf.placeholder(data_type, [batch_size, n_hidden_q])
        d_P = tf.placeholder(data_type, [batch_size, n_hidden_p])

        d_y = tf.placeholder(tf.int32, [batch_size, 3])
        d_seq_lens = tf.placeholder(tf.int32, [batch_size])

        d_dropout_l0 = tf.placeholder(tf.float32, None)
        d_dropout_l1 = tf.placeholder(tf.float32, None)
        d_dropout_l2 = tf.placeholder(tf.float32, None)
        d_dropout_l3 = tf.placeholder(tf.float32, None)

        with tf.name_scope('dense_concat_layers'):
            X = tf.concat([d_X, d_P, d_Q], axis=-1)
            X = tf.keras.layers.Dropout(d_dropout_l0)(X)
            
            X = tf.layers.dense(X, n_hidden_l1, activation=None)
            X = tf.layers.batch_normalization(X)
            X = tf.nn.relu(X)
            X = tf.keras.layers.Dropout(d_dropout_l1)(X)

            # X = tf.layers.dense(X, n_hidden_l2, activation=None)
            # X = tf.layers.batch_normalization(X)
            # X = tf.nn.relu(X)
            # X = tf.keras.layers.Dropout(d_dropout_l2)(X)

            y_hat = tf.layers.dense(X, 3, name = 'output', kernel_regularizer = tf.contrib.layers.l2_regularizer(l2_l3))

        with tf.name_scope('loss'):
            probs = tf.nn.softmax(y_hat, axis=-1)
            # label smoothing works
            loss = tf.losses.softmax_cross_entropy(d_y, logits=y_hat, label_smoothing=label_smoothing)
            loss = tf.reduce_mean(loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss)

        return AttrDict(locals())
    
    @staticmethod
    def mean_featurizer(X, X_plus):
        batch_x = []
        batch_q = []
        batch_p = []
        seq_lens = []
        batch_y = []
        for idx, row in X.iterrows():
            plus_row = X_plus.loc[idx]
            x = np.array(row.bert)
            q = np.array(plus_row.plus)
            p = row.pretrained

            pronoun_vec = x[row.pronoun_offset_token]
            a_vec = x[row.a_span[0]:row.a_span[1]+1]
            b_vec = x[row.b_span[0]:row.b_span[1]+1]
            x = np.hstack((pronoun_vec, np.mean(a_vec, axis=0), np.mean(b_vec, axis=0))).reshape(-1)

            pronoun_vec = q[plus_row.pronoun_offset_token]
            a_vec = q[plus_row.a_span[0]:plus_row.a_span[1]+1]
            b_vec = q[plus_row.b_span[0]:plus_row.b_span[1]+1]
            q = np.hstack((pronoun_vec, np.mean(a_vec, axis=0), np.mean(b_vec, axis=0))).reshape(-1)

            batch_q.append(q)
            batch_x.append(x)
            batch_p.append(p)
            seq_lens.append(len(row.tokens))

            batch_y.append(np.array([row.a_coref, row.b_coref, (row.a_coref == 0 and row.b_coref == 0)]))

        return pd.DataFrame([batch_x, batch_q, batch_p, batch_y, seq_lens]).T