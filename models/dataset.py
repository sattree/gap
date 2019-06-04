import pandas as pd
import numpy as np
from attrdict import AttrDict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from IPython.core.display import display, HTML

class Dataset(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples
        
    def transform(self, 
                    X, 
                    pretrained=None, 
                    label_corrections=None,
                    shift_by_one=True,
                    verbose=0):

        if label_corrections is not None:
            label_corrections = pd.read_csv(label_corrections, 
                                            sep='-', 
                                            header=None, 
                                            comment='#', 
                                            names=['id', 'label'])

            if shift_by_one:
                label_corrections['id'] = label_corrections['id'] -1
        else:
            label_corrections = pd.DataFrame(columns=['id', 'label'])
        
        X = pd.read_csv(X, sep='\t')
        
        if pretrained is not None:
            pretrained = pd.read_csv(pretrained)
        else:
            pretrained = pd.DataFrame(np.ones((len(X), 3))*0.33)

        if self.n_samples:
            X = X.head(self.n_samples)
            pretrained = pretrained.head(self.n_samples)
            
        # normalizing column names
        X.columns = map(lambda x: x.lower().replace('-', '_'), X.columns)
        if verbose:
            with pd.option_context('display.max_rows', 10, 'display.max_colwidth', 15):
                display(X)
        
        if 'a_coref' in X.columns and 'b_coref' in X.columns:
            y = pd.DataFrame(X[['a_coref', 'b_coref']].values, columns=['A', 'B'])
            y['NEITHER'] = ~y['A'] & ~y['B']
        else:
            y = pd.DataFrame([[False, False]]*len(X), columns=['A', 'B'])
            y['NEITHER'] = ~y['A'] & ~y['B']

        return AttrDict(locals())