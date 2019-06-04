from sklearn.base import BaseEstimator, TransformerMixin
from attrdict import AttrDict
import pandas as pd

class PretrainedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, models):
        self.models = {
                        'syn': 'syntactic_distance', 
                        'par':  'parallelism', 
                        'url': 'parallelism_url',
                        'allen': 'allen_ml', 
                        'hug': 'huggingface_ml', 
                        # 'stanford_ml_deterministic', 
                        # 'stanford_ml_statistical',
                        # 'stanford_ml_neural',
                        # 'bcs',
                        'lee': 'lee'
                      }
                      
        self.models = [self.models[model] for model in models]
        self.feats = ['{}_a_coref'.format(model) for model in self.models] + \
                        ['{}_b_coref'.format(model) for model in self.models] + \
                        ['{}_neither_coref'.format(model) for model in self.models]
    
    def transform(self, X):
        #X = pd.read_csv(X)
        pretrained = X[self.feats]
        # REALLY??
        # oh, bcoz only lee et al is being used
        # pretrained = pretrained.reshape(len(X), -1, 2)
        # pretrained.loc[:, 'NEITHER'] = 0.33
        # pretrained.loc[:, 'NEITHER'] = pretrained.sum(axis=1) == 0
        return AttrDict({'X': pretrained})