from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from attrdict import AttrDict
from sklearn.metrics import classification_report, log_loss
from bayes_opt import BayesianOptimization
import functools

class Ensembler(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model):
        self.base_model = base_model
        self.lr = LogisticRegression(random_state=0, C=1.0, solver='lbfgs',
                          multi_class='multinomial')
    
    def fit(self, X, X_val, X_tst, verbose, **params):
        self.X_val = X_val
        C = params['C']
        del params['C']
        res = self.base_model.train_evaluate_cv(X,
                                X_val=None, 
                                X_tst=[X_val, X_tst], 
                                batch_size=32, 
                                verbose=verbose,
                                seed=21,
                                return_probs=True,
                                **params
                                )
        
        self.X_ens, self.X_tst = res.probs_raw
        X_ens = self.X_ens
        
        self.X_tst = np.transpose(self.X_tst, (1, 0, 2)).reshape(-1, 15)
        self.X_tst = np.hstack((self.X_tst, np.array(X_tst[2].values.tolist())))
                          
        X_ens = np.transpose(X_ens, (1, 0, 2)).reshape(-1, 15)
        X_ens = np.hstack((X_ens, np.array(X_val[2].values.tolist())))
                          
        y_ens = np.argmax(np.array(X_val[4].values.tolist()), axis=1)
        
        self.lr = LogisticRegression(random_state=0, C=C, solver='lbfgs',
                          multi_class='multinomial')
        self.lr.fit(X_ens, y_ens)
        
        return self
    
    def predict(self, X):
        return np.array(X[4].values.tolist()), self.lr.predict_proba(self.X_tst)
    
    def train_evaluate(self, 
                       X, 
                       X_val=None, 
                       X_tst=None, 
                       batch_size=32, 
                       verbose=1,
                       return_probs=True,
                       n_trials=None,
                       **parameters):
        
        ens_size = parameters['ens_size']
        del parameters['ens_size']
        
        X = pd.concat([X, X_val], axis=0).reset_index(drop=True)
        X, X_val = train_test_split(X, test_size=ens_size, shuffle=True, random_state=21)
        X = X.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        
        self.fit(X, 
                 X_val=X_val, 
                 X_tst=X_tst,
                 verbose=verbose,
                 **parameters)
        
        y_true, probs_tst = self.predict(X_tst)
        
        if verbose:
            print('Ensemble Test score: ', log_loss(y_true, probs_tst))
        
        if return_probs:
            return AttrDict(locals())
        
        return -log_loss(y_true, probs_tst)
    
    def hyperopt(self,
                 fn,
                 X, 
                 X_val=None,
                 X_tst=None,
                 init_points=5,
                 n_iter=20,
                 batch_size=32,
                 params={'n_hidden_l1': (32, 256),
                        'n_hidden_l2': (32, 256),
                        'dropout_rate': (.1, .8)}, 
                 verbose=0,
                 **kwargs):
        self.bo = bo = BayesianOptimization(
                        f=functools.partial(fn, 
                                            X, 
                                            X_val, 
                                            X_tst, 
                                            batch_size=batch_size, 
                                            verbose=0,
                                            return_probs=False),
                        pbounds=params,
                        # random_state=1,
                        **kwargs
                    )
        
#         logger = JSONLogger(path="tmp/hyperopt/logs.json")
#         bo.subscribe(Events.OPTMIZATION_STEP, logger)
        
        bo.maximize(
                        init_points=init_points,
                        n_iter=n_iter,
                    )
        
        print(bo.max)