import inspect

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import pandas as pd
from attrdict import AttrDict
from sklearn.externals import joblib


class PlusFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.plus_feats = None
    
    def fit(self, X):
        X = X['text']
        vocab_feats = []
        for idx, text in tqdm(enumerate(X), total=X.shape[0]):
            for token in self.model(text):
                vocab_feats.append(PlusFeatures.get_feats(idx, token))
        
        vocab_feats = pd.DataFrame(vocab_feats)
        vocab_feats['is_sent_start'].fillna(False, inplace=True)
        
        n_vals = vocab_feats.describe(include='all')
        plus_feats = n_vals.loc['unique'][n_vals.loc['unique'] < 500].index.tolist()
        
        self.plus_feats = plus_feats
        self.ohe.fit(vocab_feats[plus_feats])
        return self
    
    def transform(self, X):
        X_ = X
        X = X['text']
        vocab_feats = []
        for idx, text in tqdm(enumerate(X), total=X.shape[0]):
            for token in self.model(text):
                vocab_feats.append(self.get_feats(idx, token))
        
        vocab_feats = pd.DataFrame(vocab_feats)
        vocab_feats['is_sent_start'].fillna(False, inplace=True)
        
        plus_feats = self.ohe.transform(vocab_feats[self.plus_feats])
        plus_feats = pd.DataFrame(plus_feats)
        plus_feats['sample_idx'] = vocab_feats['sample_idx']
        plus_feats = plus_feats.set_index('sample_idx')
        
        data = []
        for idx in range(len(X)):
            feats = plus_feats.loc[idx].values
            data.append(feats)
        
        X_['plus'] = data
        return {'X': X_}
    
    @staticmethod
    def get_props(obj):
        pr = []
        for name in dir(obj):
            value = getattr(obj, name)
            if not name.startswith('_') and not inspect.ismethod(value) and \
                    (name.startswith('like') or name.startswith('is_') or \
                     name.startswith('text') or name.endswith('_')) and not name.endswith('_ancestor'):
                pr.append(name)
        return pr

    @staticmethod
    def get_feats(idx, tok):
        props = PlusFeatures.get_props(tok)

        features = {prop: getattr(tok, prop) for prop in props}
        features['word'] = tok.text
        features['sample_idx'] = idx
        return features
    
    def persist(self, filepath):
        joblib.dump([self.ohe, self.plus_feats], filepath)
        
    def load(self, filepath):
        self.ohe, self.plus_feats = joblib.load(filepath)
        return self