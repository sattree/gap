import logging

import pandas as pd
import numpy as np
from attrdict import AttrDict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from IPython.core.display import display, HTML
from pathlib import Path

from .preprocessing.pretrained_features import PretrainedFeatures
from .preprocessing.label_sanitizer import LabelSanitizer
from .preprocessing.mentions_annotator import MentionsAnnotator
from .preprocessing.coref_extractor import CorefExtractor
from .preprocessing.coref_annotator import CorefAnnotator
from .preprocessing.pretrained_proref import PretrainedProref

from externals.modified_steppy.base import Step, BaseTransformer, make_transformer
from externals.modified_steppy.adapter import Adapter, E

logging.getLogger('allennlp').setLevel(logging.WARN)

logger = logging.getLogger('steppy')
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(syslog)
logger.propagate = False

def concat(X, X1, y):
    X = X.copy()
    X['label'] = np.argmax(y.values, axis=1)
    X['pretrained'] = X1.values.tolist()

    return X

def data_pipeline(EXPERIMENT_DIR='tmp/exp', 
                    mode='train',
                    sanitize_labels=False,
                    annotate_mentions=True, 
                    annotate_coref_mentions=False,
                    pretrained_proref=False,
                    coref_models=None,
                    persist=False):

    if annotate_coref_mentions:
        coref_extractor = CorefExtractor(**coref_models)
        coref_annotator = CorefAnnotator(coref_models.keys())
    if pretrained_proref:
        proref_extractor  = PretrainedProref()
    
    EXPERIMENT_DIR = Path(EXPERIMENT_DIR) / 'data_pipeline'
    EXPERIMENT_DIR = str(EXPERIMENT_DIR)

    input_reader_step = Step(name='InputReader',
                               transformer=make_transformer(lambda X, y, pretrained: {'X': X, 'y': y, 'pretrained': pretrained}),
                               input_data=['input'],
                               adapter=Adapter({
                                  'X': E('input', 'X'),
                                  'y': E('input', 'y'),
                                  'pretrained': E('input', 'pretrained'),
                                }),
                               experiment_directory=EXPERIMENT_DIR,
                               persist_output=False,
                               load_persisted_output=False,
                               is_fittable=False,
                               force_fitting=False)
    input_reader_step._mode = mode
    input_data_step = input_reader_step

    if annotate_coref_mentions or pretrained_proref:
        coref_extraction_step = Step(name='CorefExtractor',
                                     transformer=coref_extractor,
                                     input_data=['input'],
                                     adapter=Adapter({'X': E('input', 'X')}),
                                     experiment_directory=EXPERIMENT_DIR,
                                     persist_output=persist,
                                     load_persisted_output=persist,
                                     cache_output=True,
                                     is_fittable=False,
                                     force_fitting=False)
        coref_extraction_step._mode = mode

    if pretrained_proref:
        proref_step = Step(name='PretrainedProref',
                           transformer=proref_extractor,
                           input_data=['input'],
                           input_steps=[coref_extraction_step],
                           adapter=Adapter({'X': E('input', 'X'),
                                'syn': E('CorefExtractor', 'syn'),
                                'par': E('CorefExtractor', 'par'),
                                'url': E('CorefExtractor', 'url'),
                                'allen': E('CorefExtractor', 'allen'),
                                'hug': E('CorefExtractor', 'hug'),
                                'lee': E('CorefExtractor', 'lee'),
                            }),
                           experiment_directory=EXPERIMENT_DIR,
                           persist_output=persist,
                           load_persisted_output=persist,
                           is_fittable=False,
                           force_fitting=False)
        proref_step._mode = mode

    label_sanitization_step = Step(name='LabelSanitizer',
                                 transformer=LabelSanitizer(sanitize_labels),
                                 input_data=['input'],
                                 adapter=Adapter({
                                  'X': E('input', 'X'),
                                  'corrections': E('input', 'label_corrections')
                                 }),
                                 experiment_directory=EXPERIMENT_DIR,
                                 persist_output=False,
                                 load_persisted_output=False,
                                 is_fittable=False,
                                 force_fitting=True)
    label_sanitization_step._mode = mode


    if annotate_coref_mentions:
        coref_annotator_step = Step(name='CorefAnnotator',
                                 transformer=coref_annotator,
                                 input_steps=[coref_extraction_step],
                                 input_data=['input'],
                                  adapter=Adapter({'X': E('input', 'X'),
                                    'syn': E('CorefExtractor', 'syn'),
                                    'par': E('CorefExtractor', 'par'),
                                    'url': E('CorefExtractor', 'url'),
                                    'allen': E('CorefExtractor', 'allen'),
                                    'hug': E('CorefExtractor', 'hug'),
                                    'lee': E('CorefExtractor', 'lee'),
                                }),
                                 experiment_directory=EXPERIMENT_DIR,
                                 persist_output=False,
                                 load_persisted_output=False,
                                 is_fittable=False,
                                 force_fitting=False)
        coref_annotator_step._mode = mode
        input_data_step = coref_annotator_step

    if annotate_mentions:
        mentions_annotator_step = Step(name='MentionsAnnotator',
                                     transformer=MentionsAnnotator(),
                                     input_steps=[input_data_step],
                                     experiment_directory=EXPERIMENT_DIR,
                                     persist_output=False,
                                     load_persisted_output=False,
                                     is_fittable=False,
                                     force_fitting=False)
        mentions_annotator_step._mode = mode

    if pretrained_proref:
        pretrained_features_step = Step(name='PretrainedFeatures',
                                     transformer=PretrainedFeatures(coref_models.keys()),
                                     input_steps=[proref_step],
                                     experiment_directory=EXPERIMENT_DIR,
                                     is_fittable=False)
        pretrained_features_step._mode = mode
    
    input_steps = [input_reader_step, label_sanitization_step]

    if annotate_mentions:
        mentions_adapter = E('MentionsAnnotator', 'X')
        input_steps.append(mentions_annotator_step)
    else:
        mentions_adapter = E('InputReader', 'X')

    if pretrained_proref:
        pretrained_adapter = E('PretrainedFeatures', 'X')
        input_steps.append(pretrained_features_step)
    else:
        pretrained_adapter = E('InputReader', 'pretrained')

    gather_step = Step(name='gather_step',
                        transformer=make_transformer(lambda X, X_pretrained, y: {'X': concat(X, X_pretrained, y)}),
                        input_steps=input_steps,
                        adapter=Adapter({
                            'X': mentions_adapter,
                            'X_pretrained': pretrained_adapter,
                            'y': E('LabelSanitizer', 'y'),
                        }),
                        persist_output=False,
                        load_persisted_output=False,
                        experiment_directory=EXPERIMENT_DIR,
                        is_fittable=False,
                      force_fitting=True)
    gather_step._mode = mode

    return AttrDict(locals())