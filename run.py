import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

import torch

from tqdm import tqdm
from attrdict import AttrDict

from models.model_pipelines import Model
from models.gap.exec import fit_fold
from models.utils import init_coref_models, init_data

tqdm.pandas(desc="Applying..")

logger= logging.getLogger("GAP")
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(syslog)
logger.propagate = False

def run(verbose=0,
        model_version=None,
        coref_models=[],
        data_dir=None,
        exp_dir=None,
        do_preprocess_train=False,
        do_preprocess_eval=False,
        force=False,
        **kwargs):

    args = AttrDict(kwargs)
    exp_dir = Path(exp_dir)

    logging.getLogger('steppy').setLevel(logging.INFO)
    if verbose == 0:
        logging.getLogger('steppy').setLevel(logging.WARNING)

    if do_preprocess_train or do_preprocess_eval:
        if do_preprocess_train and force:
            shutil.rmtree(exp_dir / 'data_pipeline', ignore_errors=True)
        if do_preprocess_eval and force:
            # remove eval data
            shutil.rmtree(exp_dir / 'data_pipeline' / 'test', ignore_errors=True)
      
        if model_version == 'grep':
            coref_models_ = init_coref_models(coref_models)
        else:
            coref_models_ = []
    else:
        coref_models_ = {name: None for name in coref_models}

    annotate_coref_mentions = pretrained_proref = model_version == 'grep'
    X_trn, X_val, X_tst, X_neither, X_inference = init_data(data_dir,
                                                            exp_dir,
                                                            persist=True,
                                                            sanitize_labels=args.sanitize_labels,
                                                            annotate_coref_mentions=annotate_coref_mentions,
                                                            pretrained_proref=pretrained_proref,
                                                            coref_models=coref_models_,
                                                            test_path=args.test_path,
                                                            verbose=verbose)


    if args.do_train or args.do_eval:
        n_gpu = torch.cuda.device_count()
        n_samples = 0
        if n_gpu == 4:
            n_samples = 3
        if n_gpu == 8:
            n_samples = 8

        if args.do_kaggle:
            res = Model().ensembled_lms(fit_fold,
                                        pd.concat([X_trn, 
                                                    X_val, 
                                                    X_tst, 
                                                    X_neither, 
                                                    X_neither.head(n_samples)]).reset_index(drop=True),
                                        None,
                                        X_tst=X_inference,
                                        seeds=args.seeds,
                                        n_folds=args.n_folds,
                                        lms=args.lms,
                                        exp_dir=exp_dir,
                                        sub_sample_path=args.sub_sample_path,
                                        verbose=verbose,
                                        parameters = {
                                                'do_train': args.do_train,
                                                'do_eval': args.do_eval,
                                                'max_seq_length': args.max_seq_length,
                                                'train_batch_size': args.train_batch_size,
                                                'eval_batch_size': args.eval_batch_size,
                                                'learning_rate': args.learning_rate,
                                                'num_train_epochs': args.num_train_epochs,
                                                'patience': args.patience,
                                                'model_version': model_version,
                                                'n_coref_models': len(coref_models)
                                            }
                                        )
        else:
            if args.test_path:
                X_tst = X_inference

            res = Model().train_evaluate(fit_fold,
                             X_trn,
                             X_val,
                             X_tst=X_tst,
                             seed=args.seeds[0],
                             lm=args.lms[0],
                             exp_dir=exp_dir,
                             sub_sample_path=args.sub_sample_path,
                             test_path=args.test_path,
                             verbose=verbose,
                             parameters = {
                                                'do_train': args.do_train,
                                                'do_eval': args.do_eval,
                                                'max_seq_length': args.max_seq_length,
                                                'train_batch_size': args.train_batch_size,
                                                'eval_batch_size': args.eval_batch_size,
                                                'learning_rate': args.learning_rate,
                                                'num_train_epochs': args.num_train_epochs,
                                                'patience': args.patience,
                                                'model_version': model_version,
                                                'n_coref_models': len(coref_models)
                                            }
                             )

    return res

def main():
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", 
                        default='grep', 
                        type=str, 
                        choices=['probert', 'grep'],
                        help="probert or grep")

    parser.add_argument("--language_model", 
                        default='bert-base-uncased', 
                        type=str, 
                        choices=['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'],
                        help="Lanugage model to be used. In Kaggle mode, the predictions will be averaged over all runs.")

    parser.add_argument("--coref_models", 
                        default='url,allen,hug,lee', 
                        type=str, 
                        help="Coref models to be used by GREP. Syntactic distance, Parallelism, Parallelism+URL, \
                        AllenNLP, Huggingface NeuralCoref, e2e coref by Lee Et Al. Choices are 'syn', 'par', 'url', 'allen', 'hug', 'lee'")

    parser.add_argument("--preprocess_train",
                        default=False,
                        action='store_true')

    parser.add_argument("--preprocess_eval",
                        default=False,
                        action='store_true')

    parser.add_argument("--train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--predict",
                        default=False,
                        action='store_true',
                        help="Whether to predict on the test set.")

    parser.add_argument("--kaggle",
                        default=False,
                        action='store_true',
                        help="If true all of the data will be used for training. Otherwise, only gap-development will be used.")

    parser.add_argument("--data_dir",
                        default='data/',
                        type=str)
    
    parser.add_argument("--exp_dir",
                        required=True,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--test_path",
                        default=None,
                        type=str)

    parser.add_argument("--sub_sample_path",
                        default=None,
                        type=str)

    parser.add_argument("--verbose",
                        default=0,
                        type=int)

    parser.add_argument("--force",
                        default=False,
                        action='store_true',
                        help='Force clears all cached data.')

    args = parser.parse_args()
    lms = args.language_model.split(',')
    coref_models = args.coref_models.split(',')
    
    res = run(model_version=args.model,
                lms=lms,
                coref_models=coref_models,
                sanitize_labels=True,
                seeds=[42, 59, 75, 46, 91],
                n_folds=5,
                do_preprocess_train=args.preprocess_train,
                do_preprocess_eval=args.preprocess_eval,
                do_train=args.train,
                do_eval=args.predict,
                do_kaggle=args.kaggle,
                data_dir=args.data_dir,
                exp_dir=args.exp_dir,
                test_path=args.test_path,
                sub_sample_path=args.sub_sample_path,
                max_seq_length=512,
                train_batch_size=6,
                eval_batch_size=32,
                learning_rate=4e-6,
                num_train_epochs=20,
                patience=3,
                verbose=args.verbose,
                force=args.force
            )

if __name__ == "__main__":
    main()