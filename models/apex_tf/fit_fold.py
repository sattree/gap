import sys
import os

sys.path.insert(0, '.')

from models.apex.architectures.coarse_grain_model import CoarseGrainModel
import tensorflow as tf
import joblib
import argparse
import pandas as pd
import logging
import os
import json
import uuid
import joblib
import importlib
import logging
import subprocess
from timeit import default_timer as timer
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, log_loss

import time
import torch

import logging
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

logger = logging.getLogger('Fit fold')
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(syslog)
logger.propagate = False

def fit_fold(fold_n, ckpt, model, X_trn, X_val, X_tst, batch_size, verbose, seed, parameters):
     start = timer()

     model = model(X_trn, ckpt, device='GPU:0', use_pretrained=True, use_swa=True, seed=seed, **parameters)
     model.fit(X_trn, X_val=X_val, verbose=verbose, batch_size=batch_size, use_swa=True, seed=seed)

     _, y_true_tst, probs_tst = model.predict(X_tst, batch_size, verbose=verbose, seed=seed, **parameters)

     tst_score = log_loss(y_true_tst, probs_tst)
     if verbose:
          print('Fold {} done in {}s. Test score - {}'.format(fold_n, int(timer()-start), tst_score))

     return model.best_score, model.best_score_epoch, y_true_tst, probs_tst, tst_score

def fit_fold_parallel(*args, **kwargs):
     verbose = args[-1]
     data_path = '{}/{}'.format(args[4], str(uuid.uuid4()))
     cmd = "python3 models/apex/fit_fold.py \
                  --data_path='{}' \
                  --args='{}' \
                  --kwargs='{}'".format(data_path, json.dumps(args), json.dumps(kwargs))
            
     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
     for line in iter(p.stdout.readline, ''):
          if verbose:
            logger.info(line.strip())
          else:
            sys.stdout.write('\r{0: <140}'.format(line.strip())[:140])
            sys.stdout.flush()

     sys.stdout.write('\r{0: <140}'.format(''))
     sys.stdout.flush()
     retval = p.wait()

     time.sleep(10)

     return joblib.load(data_path)

if __name__ == '__main__':
     parser = argparse.ArgumentParser()

     ## Required parameters
     parser.add_argument("--data_path",
                         default=None,
                         type=str,
                         required=True,
                         help="")
     parser.add_argument("--args",
                         default=None,
                         type=str,
                         required=True,
                         help="")
     parser.add_argument("--kwargs",
                         default=None,
                         type=str,
                         required=True,
                         help="")

     args = parser.parse_args()

     data_path = args.data_path
     fold, ckpt, model_module, model_class, exp_dir, n_gpu, trn_idx, val_idx, batch_size, seed, parameters, verbose = json.loads(args.args)

     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
     os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(fold%n_gpu)

     logger.info('Loading data for fold {}.'.format(fold))

     filepath = os.path.join(exp_dir, 'output', 'train', 'gather_step')
     X_trn = joblib.load(filepath)
     X_trn = pd.DataFrame(X_trn).T
     filepath = os.path.join(exp_dir, 'output', 'validation', 'gather_step')
     X_val = joblib.load(filepath)
     X_val = pd.DataFrame(X_val).T
     filepath = os.path.join(exp_dir, 'output', 'test', 'gather_step')
     X_tst = joblib.load(filepath)
     X_tst = pd.DataFrame(X_tst).T

     X = pd.concat([X_trn, X_val], axis = 0).reset_index(drop=True)

     X_trn, X_val = X.loc[trn_idx], X.loc[val_idx]

     logger.info('Fit model {}.{} for fold {} now.'.format(model_module, model_class, fold))

     module = importlib.import_module(model_module)
     model = getattr(module, model_class)

     res = fit_fold(fold, ckpt, model, X_trn, X_val, X_tst, batch_size, verbose, seed, parameters)

     joblib.dump(res, data_path)