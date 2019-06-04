import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from tqdm import tqdm
from attrdict import AttrDict
from sklearn.metrics import classification_report, log_loss
from externals.modified_gap.gap_scorer_ext import multiclass_to_gap_score
import functools
import time
import logging
from timeit import default_timer as timer
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from .base.bayes_opt import BayesianOptimization

logger= logging.getLogger("Model")
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(syslog)
logger.propagate = False

tf.logging.set_verbosity(tf.logging.ERROR)

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, name=''):
        self.model = model
        self.name = name

    def save_probabilities(self, sub_dir, probs, score, prefix='tst', cols=None):
        probs_df = pd.DataFrame(np.hstack(probs), columns=cols)
        sub_dir = Path(sub_dir)
        probs_df.to_csv(sub_dir / "{}_probs_raw_{}_{}.csv".format(prefix, str(round(score, 5)).replace('.', '_'),
                                                          time.strftime("%Y%m%d-%H%M%S")), 
                          index=False)

    def save_submission(self, sample_path, save_path, probabilties, score, sub_df=None):
        if sample_path:
            sub_df = pd.read_csv(sample_path)
            if len(sub_df) != len(probabilties):
                return sub_df

        sub_df.loc[:, 'A'] = probabilties[:, 0]
        sub_df.loc[:, 'B'] = probabilties[:, 1]
        sub_df.loc[:, 'NEITHER'] = probabilties[:, 2]

        save_path = Path(save_path)
        sub_df.to_csv(save_path / "sub_{}_{}.csv".format(str(round(score, 5)).replace('.', '_'),
                                                          time.strftime("%Y%m%d-%H%M%S")),
                                                        index=False)

        return sub_df
      
    def train_evaluate(self, 
                       fit_fold,
                       X_trn, 
                       X_val=None, 
                       X_tst=None, 
                       lm=None,
                       test_path=None,
                       sub_sample_path=None,
                       verbose=0,
                       seed=7,
                       exp_dir=None,
                       return_hyperopt=False,
                       parameters={}):
        
        exp_dir = Path(exp_dir) / 'train_evaluate' / lm
        parameters['seed'] = seed
        do_lower_case = True if 'uncased' in lm else False
        parameters['do_lower_case'] = do_lower_case
        parameters['bert_model'] = lm
        
        res = fit_fold(0, 
                      exp_dir,
                      X_trn, 
                      X_val, 
                      X_tst, 
                      verbose,
                      parameters)
        
        best_epochs, val_probs, val_scores, tst_probs, tst_scores, attn_wts = res
        
        print('Best validation (early stopping) epochs: ', best_epochs)
        print('Validation scores: ', val_scores)
        
        if X_tst is not None:
            probs = tst_probs
            y_true = X_tst['label']
            tst_score = log_loss(y_true, probs, labels=[0, 1, 2])
            print('Test scores: ', tst_scores)
            print('Ensembled Test score: ', tst_score)

        if test_path:
            sub_df = pd.DataFrame(np.zeros((len(X_tst), 3)), columns=['A', 'B', 'NEITHER'])
            self.save_submission(sub_sample_path, 
                                'submissions/', 
                                tst_probs, 
                                tst_scores, 
                                sub_df=sub_df)
        
        if return_hyperopt:
          return -tst_score

        return AttrDict(locals())
            
    def train_evaluate_cv(self,
                          fit_fold, 
                          X, 
                          X_val=None, 
                          X_tst=None, 
                          n_folds=4, 
                          seed=None,
                          verbose=0, 
                          return_probs=True,
                          sub_sample_path='data/sample_submission_stage_1.csv',
                          parallel=False,
                          exp_dir=None,
                          **parameters):
        exp_dir = Path(exp_dir) / 'train_evaluate_cv'
        
        # Validation data is always concatenated for k-folds cv  
        if X_val is not None:
            X = pd.concat([X, X_val], axis = 0).reset_index(drop=True)

        if n_folds == 1:
          # This makes the test data into validation daa
          # We need to give a better meaning to single fold evaluation
          # this could mean lack of cv i.e. simple train evaluate
          # wrong indexes would be an overlap
          folds = [(X.index, [])]
        else:
          if X is None:
            # if X is none, then it must be the eval case, simply make a copy
            X = X_tst
          folds = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
          folds = folds.split(X)

        workers = []
        val_ids = []
        for fold_n, (trn_idx, val_idx) in enumerate(folds):
          parameters['seed'] = seed
          exp_dir_fold = exp_dir / str(fold_n)

          if parallel:
            worker = delayed(fit_fold_parallel)(*(fold_n, ckpt, self.model.__module__, self.model.__name__, exp_dir, n_gpus, trn_idx.tolist(), val_idx.tolist(), batch_size, seed, parameters, verbose))
            workers.append(worker)
          else:  
            val_ids.append(val_idx)
            X_trn, X_val = X.loc[trn_idx], X.loc[val_idx]
            worker = fit_fold(fold_n, 
                              exp_dir_fold,
                              X_trn, 
                              X_val, 
                              X_tst, 
                              verbose, 
                              parameters)
            workers.append(worker)

        if parallel:
          with Parallel(n_jobs=n_folds, verbose=verbose, backend='threading') as parallel:
            res = parallel(workers)
        else:
            res = workers

        best_epochs, val_probs, val_scores, tst_probs, tst_scores = zip(*res)
        
        SUBMISSION_DIR = exp_dir / 'submission'
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

        # single fold doesn't have any validation
        if n_folds > 1 and parameters['do_train']:
            # consolidate all out of fold predictions
            val_probs_ = np.zeros((len(X), 3))
            for probs, ids in zip(val_probs, val_ids):
                val_probs_[ids, :] = probs
            val_probs = [val_probs_.tolist()]
            
            self.save_probabilities(SUBMISSION_DIR, val_probs, np.mean(val_scores), prefix='val')
            
            probs = np.mean(val_probs, axis=0)
            y_true = X['label']
            val_score = log_loss(y_true, probs, labels=[0, 1, 2])
        else:
            val_score = np.inf

        print('Best validation (early stopping) epochs: ', best_epochs)
        print('Validation scores: ', val_scores)
        print('CV performance: {} +/- {}'.format(np.mean(val_scores), np.std(val_scores)))
        
        if X_tst is not None:
            probs = np.mean(tst_probs, axis=0)
            y_true = X_tst['label']
            tst_score = log_loss(y_true, probs, labels=[0, 1, 2])
            print('Test scores: ', tst_scores)
            print('Mean test score: {} +/- {}'.format(np.mean(tst_scores), np.std(tst_scores)))
            print('Ensembled Test score: ', tst_score)

            self.save_probabilities(SUBMISSION_DIR, tst_probs, tst_score)
            self.save_submission(sub_sample_path, SUBMISSION_DIR, probs, tst_score)
        else:
            # we can make cv preds as test preds
            # wrapper is not aware of the cv splitting and can consider the whole data to be test data
            tst_probs = [val_probs]
            y_true = X['label']
            probs = np.mean(tst_probs, axis=0)
            tst_score = log_loss(y_true, probs, labels=[0, 1, 2])

        if return_probs:
            return AttrDict(locals())
        
        return -score
    
    def ensembled_seeds(self,
                    fit_fn,
                    X, 
                    X_val, 
                    X_tst,
                    seeds=[7, 21, 42, 56, 87], 
                    verbose=0,
                    n_folds=4,
                    return_probs=True,
                    exp_dir=None,
                    sub_sample_path='data/sample_submission_stage_1.csv',
                    parameters={}):

        exp_dir = Path(exp_dir) / 'ensembled_seeds'

        SUBMISSION_DIR = exp_dir / 'submission'
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        
        tst_probs = []
        val_probs = []
        val_scores = []
        scores = []
        for i, seed in enumerate(seeds):
            start = timer()
            exp_dir_seed = exp_dir / str(seed)

            res = self.train_evaluate_cv(fit_fn,
                                        X, 
                                        X_val, 
                                        X_tst,
                                        verbose=verbose,
                                        seed=seed,
                                        n_folds=n_folds,
                                        return_probs=True,
                                        exp_dir=exp_dir_seed,
                                        sub_sample_path=sub_sample_path,
                                        **parameters)

            tst_probs += res.tst_probs
            val_probs += res.val_probs
            val_scores.append(res.val_score)
            scores.append(res.tst_score)
            print('Trial {} done in {}\n'.format(i, timedelta(seconds=int(timer()-start))))

        if X is None:
          y_true = X_tst['label']
        else:
          y_true = pd.concat([X, X_val]).reset_index(drop=True)['label']
        
        if parameters['do_train']:
          probs = np.mean(val_probs, axis=0)
          val_score = log_loss(y_true, probs, labels=[0, 1, 2])

          print('Repeated bag scores: ', val_scores)
          print('Repeated bag validation scores: {} +/- {}'.format(np.mean(val_scores), np.std(val_scores)))
          print('Bagged ensemble validation score: ', val_score)
        else:
          val_score = np.inf

        probs = np.mean(tst_probs, axis=0)
        y_true = X_tst['label']
        tst_score = log_loss(y_true, probs, labels=[0, 1, 2])

        sub_df = None
        if X_tst is None:
            # just a hack, the test results are as if there was only one fold
            n_folds = 1
            sub_path = None
            sub_df = pd.DataFrame(np.zeros((len(X), 3)), columns=['A', 'B', 'NEITHER'])
            
        cols = ['{}_{}_{}'.format(cls, trial, fold) 
                          for trial in range(len(seeds)) 
                            for fold in range(n_folds) 
                              for cls in ['A', 'B', 'NEITHER']]
        self.save_probabilities(SUBMISSION_DIR, tst_probs, tst_score, cols=cols)
        self.save_submission(sub_sample_path, SUBMISSION_DIR, probs, tst_score, sub_df=sub_df)
        
        if return_probs:
            print('Repeated bag scores: ', scores)
            print('Repeated bag mean: {} +/- {}'.format(np.mean(scores), np.std(scores)))
            print('Bagged ensemble score: ', tst_score)
            return AttrDict(locals())
        
        return -log_loss(y_true, probs, labels=[0, 1, 2])

    def ensembled_lms(self,
                    fit_fn,
                    X, 
                    X_val, 
                    X_tst,
                    seeds=[7, 21, 42, 56, 87], 
                    n_folds=4,
                    lms=[],
                    verbose=0,
                    return_probs=True,
                    exp_dir=None,
                    sub_sample_path=None,
                    parameters={}):

        exp_dir = Path(exp_dir) / 'ensembled_lms'

        SUBMISSION_DIR = exp_dir / 'submission'
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        
        tst_probs = []
        val_probs = []
        val_scores = []
        scores = []
        for i, model in enumerate(lms):
            start = timer()
            do_lower_case = True if 'uncased' in model else False
            parameters['do_lower_case'] = do_lower_case
            parameters['bert_model'] = model

            exp_dir_model = exp_dir / model.replace('-', '_')

            res = self.ensembled_seeds(fit_fn,
                                        X, 
                                        X_val, 
                                        X_tst,
                                        verbose=verbose,
                                        seeds=seeds,
                                        n_folds=n_folds,
                                        return_probs=True,
                                        exp_dir=exp_dir_model,
                                        parameters=parameters,
                                        sub_sample_path=sub_sample_path)

            tst_probs += res.tst_probs
            val_probs += res.val_probs
            val_scores.append(res.val_scores)
            scores.append(res.tst_score)
            print('Language model {} done in {}\n'.format(model, timedelta(seconds=int(timer()-start))))

        if X is None:
          y_true = X_tst['label']
        else:
          y_true = pd.concat([X, X_val]).reset_index(drop=True)['label']

        if parameters['do_train']:
          probs = np.mean(val_probs, axis=0)
          val_score = log_loss(y_true, probs, labels=[0, 1, 2])

          print('Language model validation scores: ', val_scores)
          print('Language model validation performance: {} +/- {}'.format(np.mean(val_scores), np.std(val_scores)))
          print('Language model ensemble validation score: ', val_score)

        probs = np.mean(tst_probs, axis=0)
        y_true = X_tst['label']
        tst_score = log_loss(y_true, probs, labels=[0, 1, 2])

        sub_df = None
        if X_tst is None:
            # just a hack, the test results are as if there was only one fold
            n_folds = 1
            sub_path = None
            sub_df = pd.DataFrame(np.zeros((len(X), 3)), columns=['A', 'B', 'NEITHER'])
            
        cols = ['{}_{}_{}_{}'.format(cls, model_idx, trial, fold) 
                        for model_idx in range(len(lms))
                          for trial in range(len(seeds)) 
                            for fold in range(n_folds) 
                              for cls in ['A', 'B', 'NEITHER']]
        self.save_probabilities(SUBMISSION_DIR, tst_probs, tst_score, cols=cols)
        self.save_submission(sub_sample_path, SUBMISSION_DIR, probs, tst_score, sub_df=sub_df)
        
        if return_probs:
            print('Language model scores: ', scores)
            print('Language model performance : {} +/- {}'.format(np.mean(scores), np.std(scores)))
            print('Language model ensemble score: ', tst_score)
            return AttrDict(locals())
        
        return -log_loss(y_true, probs, labels=[0, 1, 2])
    
    # Defunct
    def hyperopt(self,
                 fn,
                 X, 
                 X_val=None,
                 X_tst=None,
                 n_trials=5,
                 n_folds=5,
                 init_points=5,
                 n_iter=20,
                 batch_size=64,
                 use_pretrained=False,
                 use_swa=False,
                 parallel=False,
                 search_params={'n_hidden_l1': (32, 256),
                        'n_hidden_l2': (32, 256),
                        'dropout_rate': (.1, .8)}, 
                 verbose=0,
                 seed=None,
                 n_gpus=0,
                 exp_dir=None,
                 probe_points=None,
                 **kwargs):

        self.bo = bo = BayesianOptimization(
                        f=functools.partial(fn, 
                                            X, 
                                            X_val, 
                                            X_tst, 
                                            batch_size=batch_size, 
                                            verbose=verbose,
                                            seed=seed,
                                            n_folds=n_folds,
                                            return_probs=False,
                                            use_pretrained=use_pretrained,
                                            use_swa=use_swa,
                                            parallel=parallel,
                                            exp_dir=exp_dir,
                                            n_gpus=n_gpus),
                        pbounds=search_params,
                        **kwargs
                    )

        bo._prime_subscriptions()

        log_path = "results/bo_{}_logs.json".format(self.name)
        if os.path.exists(log_path):
          load_logs(bo, logs=[log_path]);
        logger = JSONLogger(path=log_path)
        bo.subscribe(Events.OPTMIZATION_STEP, logger)

        if probe_points is not None:
          bo.probe(
              params=probe_points,
              # lazy=False
          )

        bo.maximize(
                        init_points=init_points,
                        n_iter=n_iter,
                    )
        
        print(bo.max)
