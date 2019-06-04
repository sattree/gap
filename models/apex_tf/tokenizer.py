from ..heuristics.spacy_base import SpacyModel
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pandas as pd
from attrdict import AttrDict


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer):
        self.tokenizer = SpacyModel(tokenizer)
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        try:
            res = []
            for idx, row in tqdm(X.iterrows(), total=len(X)):
                res.append(self.tokenizer.tokenize(**row)[1:])

            res = pd.DataFrame(res, columns=['tokens', 'pronoun_offset_token',
                                                    'a_offset_token', 'b_offset_token', 'a_span',
                                                    'b_span', 'pronoun_token', 'a_tokens', 'b_tokens'])

            cols = set(X.columns).difference(res.columns)
            X = pd.concat([X[cols], res], axis=1)
            return AttrDict({'X': X})
        except Exception as e:
            print(row.text)
            raise e