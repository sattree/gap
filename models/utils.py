from IPython.core.display import display, HTML

import logging
import time

from attrdict import AttrDict

import spacy

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import neuralcoref

from .base.utils import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser

from .pretrained.lee_et_al import LeeEtAl2017
from .pretrained.huggingface import HuggingfaceCorefModel
from .pretrained.allennlp import AllenNLPCorefModel
from .pretrained.stanford import StanfordCorefModel
from .heuristics.syntactic_distance import StanfordSyntacticDistanceModel
from .heuristics.parallelism import AllenNLPParallelismModel as ParallelismModel
from .heuristics.url_title import StanfordURLTitleModel as URLModel

from .dataset import Dataset
from .data_pipeline import data_pipeline

logger= logging.getLogger("GAP")

def init_coref_models(coref_models):
    SPACY_MODEL = spacy.load('en_core_web_lg')

    model_url = 'externals/data/coref-model-2018.02.05.tar.gz'
    archive = load_archive(model_url, cuda_device=0)
    ALLEN_COREF_MODEL = Predictor.from_archive(archive)

    model_url = 'externals/data/biaffine-dependency-parser-ptb-2018.08.23.tar.gz'
    archive = load_archive(model_url, cuda_device=0)
    ALLEN_DEP_MODEL = Predictor.from_archive(archive)

    model_url = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz'
    archive = load_archive(model_url, cuda_device=0)
    ALLEN_PARSE_MODEL = Predictor.from_archive(archive)

    HUGGINGFACE_COREF_MODEL = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(HUGGINGFACE_COREF_MODEL)

    STANFORD_CORENLP_PATH = 'externals/stanford-corenlp-full-2018-10-05/'
    server = CoreNLPServer(classpath=STANFORD_CORENLP_PATH,
                          corenlp_options=AttrDict({'port': 9090, 
                                                    'timeout': '600000',
                                                    'thread': '4',
                                                    'quiet': 'true',
                                                    'preload': 'tokenize,ssplit,pos,lemma,parse,depparse,ner,coref'}))
    server.start()
    STANFORD_SERVER_URL = server.url
    STANFORD_MODEL = CoreNLPParser(url=STANFORD_SERVER_URL)

    syntactic_distance_coref_model = StanfordSyntacticDistanceModel(STANFORD_MODEL)
    parallelism_coref_model = ParallelismModel(ALLEN_DEP_MODEL, SPACY_MODEL)
    url_title_coref_model = URLModel(STANFORD_MODEL)
    stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='statistical')
    allen_coref_model = AllenNLPCorefModel(ALLEN_COREF_MODEL, SPACY_MODEL)
    huggingface_coref_model = HuggingfaceCorefModel(HUGGINGFACE_COREF_MODEL)
    lee_coref_model = LeeEtAl2017(SPACY_MODEL, 
                                    config = {'name': 'final',
                                        'log_root': 'externals/data/',
                                        'model': 'externals/modified_e2e_coref/experiments.conf',
                                        'context_embeddings_root': 'externals/data/',
                                        'head_embeddings_root': 'externals/data/',
                                        'char_vocab_root': 'externals/data/',
                                        'device': 0
                                    })

    logger.info('Waiting a minute to allow all models to load.')

    time.sleep(60)

    model_instances = {
        'syn': syntactic_distance_coref_model,
        'par': parallelism_coref_model,
        'url': url_title_coref_model,
        # 'stan': stanford_coref_model,
        'allen': allen_coref_model,
        'hug': huggingface_coref_model,
        'lee': lee_coref_model
    }

    coref_models = {name: model_instances[name] for name in coref_models}

    return coref_models

def init_data(data_dir,
                exp_dir,
                persist=True,
                sanitize_labels=True,
                annotate_coref_mentions=False,
                pretrained_proref=False,
                coref_models=[],
                test_path=None,
                verbose=0):

    train = {
        'input': Dataset().transform('{}/gap-development.tsv'.format(data_dir), 
                                    label_corrections='{}/gap_corrections/corrections_dev.csv'.format(data_dir),
                                    verbose=verbose)
    }

    val = {
        'input': Dataset().transform('{}/gap-validation.tsv'.format(data_dir), 
                                    label_corrections='{}/gap_corrections/corrections_val.csv'.format(data_dir),
                                    verbose=verbose)
    }

    test = {
        'input': Dataset().transform('{}/gap-test.tsv'.format(data_dir), 
                                    label_corrections='{}/gap_corrections/corrections_tst.csv'.format(data_dir),
                                    verbose=verbose)
    }

    neither = {
        'input': Dataset().transform('{}/neither.tsv'.format(data_dir),
                                    verbose=verbose
                                    )
    }

    if test_path is not None:
        test_stage2 = {
            'input': Dataset().transform(test_path, 
                                        verbose=verbose
                                    )
        }
    
    dpl_trn = data_pipeline(exp_dir, 
                           mode='train', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=annotate_coref_mentions, 
                           pretrained_proref=pretrained_proref,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_models=coref_models
                        )
    dpl_val = data_pipeline(exp_dir, 
                           mode='val', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=annotate_coref_mentions, 
                           pretrained_proref=pretrained_proref,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_models=coref_models
                        )
    dpl_tst = data_pipeline(exp_dir, 
                           mode='test', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=annotate_coref_mentions, 
                           pretrained_proref=pretrained_proref,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_models=coref_models
                        )

    dpl_neither = data_pipeline(exp_dir, 
                           mode='neither', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=annotate_coref_mentions, 
                           pretrained_proref=pretrained_proref,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_models=coref_models
                        )

    display(dpl_trn.gather_step)

    logger.info('Transforming data to features.')

    X_trn = dpl_trn.gather_step.transform(train)['X']
    X_val = dpl_val.gather_step.transform(val)['X']
    X_tst = dpl_tst.gather_step.transform(test)['X']
    X_neither = dpl_neither.gather_step.transform(neither)['X']
    X_tst_stage2 = None

    if test_path is not None:
        dpl_test_stage2 = data_pipeline(exp_dir, 
                           mode='inference', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=annotate_coref_mentions, 
                           pretrained_proref=pretrained_proref,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_models=coref_models
                        )
        X_tst_stage2 = dpl_test_stage2.gather_step.transform(test_stage2)['X']

    logger.info('Transforming data to features done.\n Log a couple of examples for sanity check.\n')

    print(X_trn.loc[0].text)
    print(X_trn.loc[0])

    print(X_tst.loc[0].text)
    print(X_tst.loc[0])

    print(X_neither.loc[0].text)
    print(X_neither.loc[0])

    return X_trn, X_val, X_tst, X_neither, X_tst_stage2

# def do_lm_ensemble(predictions_paths,
#                     lms,
#                     seeds,
#                     n_folds,
#                    predictions_paths2=None,
#                     save_path=None,
#                     sub_sample_path=None):

#     if predictions_paths2 is None:
#         probs_raw = pd.concat([pd.read_csv(file) for file in predictions_paths], axis=1)\
#                         .values.reshape(-1, len(lms), len(seeds), 
#                                           n_folds, 3).transpose(1, 2, 3, 0, 4)
#         # average across folds, seeds, models
#         probabilties = probs_raw.mean(axis=0).mean(axis=0).mean(axis=0)
#     else:
#         probs_all = pd.concat([pd.read_csv(file) for file in predictions_paths], axis=1).values.reshape(-1, len(lms), len(seeds), n_folds, 3).transpose(1, 2, 3, 0, 4).reshape(n_folds*len(lms)*len(seeds), -1, 3)
        
#         probs2 = pd.read_csv(predictions_paths2[0]).values.reshape(-1, 1, 1, n_folds, 3).transpose(1, 2, 3, 0, 4).reshape(n_folds*1*1, -1, 3)
        
#         print(probs_all.shape, probs2.shape)

#         probabilties = np.concatenate((probs_all, probs2)).mean(axis=0)


#     if sub_sample_path:
#       sub_df = pd.read_csv(sub_sample_path)
#       sub_df.loc[:, 'A'] = probabilties[:, 0]
#       sub_df.loc[:, 'B'] = probabilties[:, 1]
#       sub_df.loc[:, 'NEITHER'] = probabilties[:, 2]

#       sub_df.to_csv(save_path, index=False)

#     return sub_df