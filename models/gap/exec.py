import argparse
import csv
import logging
import os
import random
import sys
import shutil
import contextlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from attrdict import AttrDict
from timeit import default_timer as timer
from datetime import datetime, timedelta
from pprint import pformat

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, log_loss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from .features import convert_examples_to_features
from .probert import ProBERT
from .grep import GREP

from scipy.special import softmax

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, y_true, label_list):
    label_list = [0, 1, 2]
    acc = simple_accuracy(np.argmax(preds, axis=-1), y_true)
    f1 = f1_score(y_true=y_true, y_pred=np.argmax(preds, axis=-1), average='micro', labels=label_list)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "log_loss": log_loss(y_true=y_true, y_pred=preds, labels=label_list),
    }

def compute_metrics(preds, labels, label_list):
    assert len(preds) == len(labels)    
    return acc_and_f1(preds, labels, label_list)

def evaluate(model,
            eval_features,
            device, 
            args, 
            label_list,
            num_labels,
            eval_mode=False):

    model.eval()

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.uint8)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_gpr_tags_mask = torch.tensor([f.gpr_tags_mask for f in eval_features], dtype=torch.uint8)

    all_mention_p_ids = torch.tensor([f.mention_p_ids for f in eval_features], dtype=torch.long)
    all_mention_a_ids = torch.tensor([f.mention_a_ids for f in eval_features], dtype=torch.long)
    all_mention_b_ids = torch.tensor([f.mention_b_ids for f in eval_features], dtype=torch.long)
    all_mention_p_mask = torch.tensor([f.mention_p_mask for f in eval_features], dtype=torch.uint8)
    all_mention_a_mask = torch.tensor([f.mention_a_mask for f in eval_features], dtype=torch.uint8)
    all_mention_b_mask = torch.tensor([f.mention_b_mask for f in eval_features], dtype=torch.uint8)

    all_cluster_ids_a = torch.tensor([f.cluster_ids_a for f in eval_features], dtype=torch.long)
    all_cluster_mask_a = torch.tensor([f.cluster_mask_a for f in eval_features], dtype=torch.uint8)
    all_cluster_ids_b = torch.tensor([f.cluster_ids_b for f in eval_features], dtype=torch.long)
    all_cluster_mask_b = torch.tensor([f.cluster_mask_b for f in eval_features], dtype=torch.uint8)
    all_cluster_ids_p = torch.tensor([f.cluster_ids_p for f in eval_features], dtype=torch.long)
    all_cluster_mask_p = torch.tensor([f.cluster_mask_p for f in eval_features], dtype=torch.uint8)

    all_pretrained = torch.tensor([f.pretrained for f in eval_features], dtype=torch.float)

    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, 
                                all_input_mask, 
                                all_segment_ids, 
                                all_gpr_tags_mask,
                                all_mention_p_ids,
                                all_mention_a_ids,
                                all_mention_b_ids,
                                all_mention_p_mask,
                                all_mention_a_mask,
                                all_mention_b_mask,
                                all_cluster_ids_a,
                                all_cluster_mask_a,
                                all_cluster_ids_b,
                                all_cluster_mask_b,
                                all_cluster_ids_p,
                                all_cluster_mask_p,
                                all_pretrained,
                                all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size)

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    attn_wts = []
    pbar = tqdm(desc="Evaluating", total=len(eval_dataloader)) if eval_mode else contextlib.suppress()
    with pbar:
        for step, batch in enumerate(eval_dataloader):
            # with torch.cuda.device(0):
            batch = tuple(t.to(device) for t in batch)
            (input_ids, input_mask, segment_ids, 
                gpr_tags_mask,
                mention_p_ids, mention_a_ids, mention_b_ids,
                mention_p_mask, mention_a_mask, mention_b_mask,
                cluster_ids_a, cluster_mask_a, cluster_ids_b, cluster_mask_b,
                cluster_ids_p, cluster_mask_p, pretrained, label_ids) = batch

            with torch.no_grad():
                res = model(input_ids, 
                                segment_ids, 
                                input_mask, 
                                gpr_tags_mask=gpr_tags_mask,
                                mention_p_ids=mention_p_ids,
                                mention_a_ids=mention_a_ids,
                                mention_b_ids=mention_b_ids,
                                mention_p_mask=mention_p_mask,
                                mention_a_mask=mention_a_mask,
                                mention_b_mask=mention_b_mask, 
                                cluster_ids_a=cluster_ids_a,
                                cluster_mask_a=cluster_mask_a,
                                cluster_ids_b=cluster_ids_b,
                                cluster_mask_b=cluster_mask_b,
                                cluster_ids_p=cluster_ids_p,
                                cluster_mask_p=cluster_mask_p,
                                pretrained=pretrained,
                                labels=None,
                                training=False,
                                eval_mode=eval_mode)

                if eval_mode:
                    logits, probabilties, attn_wts_m, attn_wts_c, attn_wts_co = res
                else:
                    logits, probabilties = res
            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(probabilties.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], probabilties.detach().cpu().numpy(), axis=0)

            if eval_mode:
                pbar.set_description('Evaluating, Loss={:.3f}'.format(eval_loss / nb_eval_steps))
                pbar.update()

                if len(attn_wts) == 0:
                    attn_wts = [attn_wts_m, attn_wts_c, attn_wts_co]
                else:
                    attn_wts[0] = np.append(attn_wts[0], attn_wts_m, axis=0)
                    attn_wts[1] = np.append(attn_wts[1], attn_wts_c, axis=0)
                    attn_wts[2] = np.append(attn_wts[2], attn_wts_co, axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]

    result = compute_metrics(preds, all_label_ids.numpy(), label_list)

    result['eval_loss'] = eval_loss

    # pbar.set_description('Evaluating, Loss={:.3f}, F1={:.3f}, Log loss={:.3f}'.format(eval_loss, 
    #                                                                                 result['f1'], 
    #                                                                                 result['log_loss']))

    return preds, result['log_loss'], result, attn_wts



def fit(model, 
        train_features,
        eval_features,
        test_features,
        label_list, 
        num_labels,
        tokenizer,
        device, 
        n_gpu,
        args,
        swa_model=None,
        verbose=0):

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.uint8)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_gpr_tags_mask = torch.tensor([f.gpr_tags_mask for f in train_features], dtype=torch.uint8)

    all_mention_p_ids = torch.tensor([f.mention_p_ids for f in train_features], dtype=torch.long)
    all_mention_a_ids = torch.tensor([f.mention_a_ids for f in train_features], dtype=torch.long)
    all_mention_b_ids = torch.tensor([f.mention_b_ids for f in train_features], dtype=torch.long)
    all_mention_p_mask = torch.tensor([f.mention_p_mask for f in train_features], dtype=torch.uint8)
    all_mention_a_mask = torch.tensor([f.mention_a_mask for f in train_features], dtype=torch.uint8)
    all_mention_b_mask = torch.tensor([f.mention_b_mask for f in train_features], dtype=torch.uint8)

    all_cluster_ids_a = torch.tensor([f.cluster_ids_a for f in train_features], dtype=torch.long)
    all_cluster_mask_a = torch.tensor([f.cluster_mask_a for f in train_features], dtype=torch.uint8)
    all_cluster_ids_b = torch.tensor([f.cluster_ids_b for f in train_features], dtype=torch.long)
    all_cluster_mask_b = torch.tensor([f.cluster_mask_b for f in train_features], dtype=torch.uint8)
    all_cluster_ids_p = torch.tensor([f.cluster_ids_p for f in train_features], dtype=torch.long)
    all_cluster_mask_p = torch.tensor([f.cluster_mask_p for f in train_features], dtype=torch.uint8)

    all_pretrained = torch.tensor([f.pretrained for f in train_features], dtype=torch.float)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, 
                                all_input_mask, 
                                all_segment_ids, 
                                all_gpr_tags_mask,
                                all_mention_p_ids,
                                all_mention_a_ids,
                                all_mention_b_ids,
                                all_mention_p_mask,
                                all_mention_a_mask,
                                all_mention_b_mask,
                                all_cluster_ids_a,
                                all_cluster_mask_a,
                                all_cluster_ids_b,
                                all_cluster_mask_b,
                                all_cluster_ids_p,
                                all_cluster_mask_p,
                                all_pretrained,
                                all_label_ids)

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, 
                                    sampler=train_sampler, 
                                    batch_size=args.train_batch_size)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(
                            optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion
                        )

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    best_score = np.inf
    best_epoch = 0
    since_best = 0
    preds = None
    tst_preds = None
    tst_score = np.inf
    best_swa_score = np.inf
    swa_n = 0
    for epoch in range(int(args.num_train_epochs)):
        model.train()

        # BertAdam has a default schedule
        # lr = lr_schedule(0.05, args.learning_rate, epoch, args.num_train_epochs)
        # adjust_learning_rate(optimizer, lr)
        lr = args.learning_rate

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        total = len(train_features) + len(eval_features) + len(test_features)
        with tqdm(desc="Trn, Epoch {}".format(epoch), total=total) as pbar:
            for step, batch in enumerate(train_dataloader):
                # with torch.cuda.device(0):
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask, segment_ids,
                    gpr_tags_mask, 
                    mention_p_ids, mention_a_ids, mention_b_ids,
                    mention_p_mask, mention_a_mask, mention_b_mask,
                    cluster_ids_a, cluster_mask_a, cluster_ids_b, 
                    cluster_mask_b, cluster_ids_p, cluster_mask_p, 
                    pretrained, label_ids) = batch

                # define a new function to compute loss values for both output_modes
                logits, _ = model(input_ids, 
                                    segment_ids, 
                                    input_mask, 
                                    gpr_tags_mask=gpr_tags_mask,
                                    mention_p_ids=mention_p_ids,
                                    mention_a_ids=mention_a_ids,
                                    mention_b_ids=mention_b_ids,
                                    mention_p_mask=mention_p_mask,
                                    mention_a_mask=mention_a_mask,
                                    mention_b_mask=mention_b_mask,
                                    cluster_ids_a=cluster_ids_a,
                                    cluster_mask_a=cluster_mask_a,
                                    cluster_ids_b=cluster_ids_b,
                                    cluster_mask_b=cluster_mask_b,
                                    cluster_ids_p=cluster_ids_p,
                                    cluster_mask_p=cluster_mask_p,
                                    pretrained=pretrained,
                                    labels=None,
                                    training=True)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_description('Trn {:2d}, Loss={:.3f}, Val score={:.3f}, Val SWA={:.3f}, Test score={:.3f}'.format(epoch, 
                                                                                tr_loss/nb_tr_steps, 
                                                                                np.inf,
                                                                                best_swa_score,
                                                                                np.inf))
                pbar.update(len(batch[0]))

                # if global_step % (500//args.train_batch_size) == 0 and global_step > 0:
                #     break

                # if nb_tr_steps % (500//args.train_batch_size) == 0 and nb_tr_steps > 0:
                #     break
        
            preds_, score, res, _ = evaluate(model,
                                        eval_features,
                                        device, 
                                        args, 
                                        label_list,
                                        num_labels)

            pbar.set_description('Trn {:2d}, Loss={:.3f}, Val score={:.3f} F1={:.3f}, Test score={:.3f}'.format(epoch, 
                                                                                tr_loss/nb_tr_steps, 
                                                                                score,
                                                                                res['f1'],
                                                                                np.inf))
            pbar.update(len(eval_features))

            if swa_model is not None and global_step > 150:
                preds_swa, swa_score, _, _ = evaluate(swa_model,
                                            eval_features,
                                            device, 
                                            args, 
                                            label_list,
                                            num_labels)
                if swa_score < best_swa_score:
                    best_swa_score = swa_score

            if score <= best_score:
                best_score = score
                best_epoch = epoch
                since_best = 0
                preds = preds_

                if len(test_features):
                    tst_preds_, tst_score, tst_res, _ = evaluate(model,
                                                test_features,
                                                device, 
                                                args, 
                                                label_list,
                                                num_labels)

                    pbar.set_description('Trn {:2d}, Loss={:.3f}, Val score={:.3f} F1={:.3f}, Test score={:.3f} F1={:.3f}'.format( 
                                                                                    epoch,
                                                                                    tr_loss/nb_tr_steps, 
                                                                                    score,
                                                                                    res['f1'],
                                                                                    tst_res['log_loss'],
                                                                                    tst_res['f1']
                                                                                    ))
                    pbar.update(len(test_features))

                    # score = tst_score

                else:
                    tst_preds_, tst_score = [], np.inf

                tst_preds = tst_preds_

                
                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

            else:
                since_best += 1

            if since_best == args.patience:
                break

    return best_epoch, preds, best_score, tst_preds, tst_score, None

def init_model(X_trn,
        X_val=None,
        X_tst=None,
        bert_model='bert-large-uncased', 
        model_version='grep',
        do_lower_case=True, 
        do_train=True, 
        do_eval=True, 
        eval_batch_size=8, 
        learning_rate=2e-05, 
        max_seq_length=512, 
        no_cuda=False, 
        num_train_epochs=10.0, 
        output_dir=None, 
        seed=42, 
        train_batch_size=32, 
        warmup_proportion=0.1, 
        n_coref_models=0,
        patience=1,
        verbose=0):
    args = AttrDict(locals())

    # logger.info('Executing with parameters {}.'.format(pformat(args)))

    # Environment config
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    logger.info("device: {} n_gpu: {}, ".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer
    # Download default vocab, no customizations
    # vocab_file = 'externals/bert/{}'.format(args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, 
                                            # vocab_file, 
                                            do_lower_case=args.do_lower_case,
                                            never_split=["[UNK]", "[SEP]", "[PAD]", 
                                                            "[CLS]", "[MASK]",
                                                            # "<C_0>", "<C_1>", "<D_0>", "<D_1>"
                                                        ])

    # Load data
    label_list = sorted(set(X_trn['label']))
    num_labels = len(label_list)

    train_features = convert_examples_to_features(X_trn, 
                                                    tokenizer,
                                                    args.max_seq_length, 
                                                    n_coref_models=args.n_coref_models, 
                                                    verbose=verbose
                                                    )

    eval_features = convert_examples_to_features(X_val, 
                                                tokenizer,
                                                args.max_seq_length, 
                                                n_coref_models=args.n_coref_models, 
                                                verbose=verbose
                                                )

    test_features = convert_examples_to_features(X_tst, 
                                                tokenizer,
                                                args.max_seq_length, 
                                                n_coref_models=args.n_coref_models,
                                                verbose=verbose
                                                )

    logger.info("***** Training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    # logger.info("  Num steps = %d", num_train_optimization_steps)

    logger.info("***** Evaluation *****")
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)

    logger.info("***** Testing *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Prepare model
    print('Preparing Model.')
    cache_dir = str(PYTORCH_PRETRAINED_BERT_CACHE)
    if model_version == 'probert':
        model = ProBERT.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels
                                                        )
    elif model_version == 'grep':
        model = GREP.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels
                                                        )

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.do_train:
        return fit(model, 
                            train_features,
                            eval_features,
                            test_features,
                            label_list,
                            num_labels, 
                            tokenizer,
                            device, 
                            n_gpu,
                            args,
                            swa_model=None,
                            verbose=verbose
                            )
    else:
        # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
        
        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)

        if model_version == 'probert':
            model = ProBERT(config,
                                                  num_labels=num_labels,
                                                  pretrained_dim_size=pretrained_dim_size
                                                )
        elif model_version == 'grep':
            model = GREP(config,
                                                  num_labels=num_labels,
                                                  pretrained_dim_size=pretrained_dim_size
                                                )

        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        model.load_state_dict(torch.load(output_model_file, map_location=map_location))
        model.to(device)

        tst_preds, tst_score, _, attn_wts = evaluate(model,
                        test_features,
                        device, 
                        args, 
                        label_list,
                        num_labels,
                        eval_mode=True)

        return 0, tst_preds, tst_score, tst_preds, tst_score, attn_wts

def fit_fold(fold_n, 
                exp_dir,
                X_trn, 
                X_val=None, 
                X_tst=None,
                verbose=0, 
                args={}):
    start = timer()

    logger_level = logging.INFO
    if verbose == 0:
        logger_level = logging.WARNING
    logger.setLevel(logger_level)
    logging.getLogger('pytorch_pretrained_bert.tokenization').setLevel(logger_level)
    logging.getLogger('pytorch_pretrained_bert.modeling').setLevel(logger_level)

    OUTPUT_DIR = exp_dir / args['model_version']/ str(fold_n) / 'model'
    DATA_DIR = exp_dir / args['model_version']/ str(fold_n) / 'data'

    if args['do_train']:
        logger.info('Running in train mode. Clearing model output directory.')
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    logger.info('Clearing data directory with train, val and test files in bert format.')
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    best_epoch, val_preds, score, tst_preds, tst_score, attn_wts = init_model(X_trn,
                                                                        X_val,
                                                                        X_tst, 
                                                                        output_dir=OUTPUT_DIR,
                                                                        verbose=verbose, 
                                                                        **args)

    print('Fold {} done in {}. \nTest score - {}'.format(fold_n, 
                                                        timedelta(seconds=int(timer()-start)), 
                                                        tst_score))

    if not len(X_tst):
        tst_preds = val_preds
        tst_score = score

    return best_epoch, val_preds, score, tst_preds, tst_score, attn_wts