import re
import pandas as pd
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class Sanitizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X):
        rows = []
        for idx, row in tqdm(X.iterrows(), total=len(X)):
            ex = self.example_to_debug({'X': X}, idx)
            ex = self.cleanser(ex)
            ex = self.debug_to_example(ex)
            rows.append(ex)
            
        return {'X': pd.DataFrame(rows)}

    def cleanser(self, row):
        row.text = row.text.replace("`", "'").replace("*", "'")
        row.a = row.a.replace("`", "'").replace("*", "'")
        row.b = row.b.replace("`", "'").replace("*", "'")
        
        matches = re.findall('\([^)]*\)', row.text)
        for match in matches:
            if row.a in match or match[:-1] in row.a or row.b in match or match[:-1] in row.b or row.pronoun in match:
                continue
                
            row.text = row.text.replace(match, '', 1)
        
        return row

    def example_to_debug(self, X, idx):
        ex = AttrDict(X['X'].to_dict(orient='records')[idx])
        
        text = ex.text
        text = '{}<A>{}'.format(text[:ex.a_offset], text[ex.a_offset:])
        text = '{}<B>{}'.format(text[:ex.b_offset+3], text[ex.b_offset+3:])
        
        offset = ex.pronoun_offset
        if ex.pronoun_offset > ex.a_offset:
            offset += 3
        if ex.pronoun_offset > ex.b_offset:
            offset += 3
            
        text = '{}<P>{}'.format(text[:offset], text[offset:])

        ex.a_offset = text.index('<A>')
        ex.b_offset = text.index('<B>')
        ex.pronoun_offset = text.index('<P>')

        ex.text = text
        
        return ex

    def debug_to_example(self, ex):
        text = ex.text
        ex.pronoun_offset = text.replace('<A>', '').replace('<B>', '').index('<P>')
        ex.a_offset = text.replace('<P>', '').replace('<B>', '').index('<A>')
        ex.b_offset = text.replace('<P>', '').replace('<A>', '').index('<B>')
        
        ex.text = text.replace('<P>', '').replace('<A>', '').replace('<B>', '')
        
        return ex