import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from attrdict import AttrDict
from .tokenizer import Tokenizer

class ContextReducer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def transform(self, X):
        tokenizer = Tokenizer(self.tokenizer)
        X = tokenizer.transform(X).X

        X['sentences'] = X['text'].progress_apply(self.get_sent_feats)
        X[['sent_idx_min', 'sent_idx_max']] = X.progress_apply(self.coverage, axis=1, result_type='expand')
    
        X[['text', 'pronoun_offset', 'a_offset', 'b_offset']] = X.progress_apply(self.reduction_logic, axis=1, result_type='expand')
        
        return {'X': X}
        
    def get_sent_feats(self, text):
        doc = self.tokenizer(text)
        sents = []
        for sent in doc.sents:
            sents.append({
                'start_char_offset': sent.start_char,
                'end_char_offset': sent.end_char,
                'start_token_offset': sent.start,
                'end_token_offset': sent.end,
                'text': sent.text
            })
        return sents
    
    def coverage(self, row):
        pronoun_sent_span = [np.where(np.array([sent['start_token_offset'] for sent in row.sentences]) <= row.pronoun_offset_token)[0][-1]]
        a_sent_span = [np.where(np.array([sent['start_token_offset'] for sent in row.sentences]) <= row.a_span[0])[0][-1],
                      np.where(np.array([sent['start_token_offset'] for sent in row.sentences]) <= row.a_span[1])[0][-1]]
        b_sent_span = [np.where(np.array([sent['start_token_offset'] for sent in row.sentences]) <= row.b_span[0])[0][-1],
                      np.where(np.array([sent['start_token_offset'] for sent in row.sentences]) <= row.b_span[1])[0][-1]]
        coverage = pronoun_sent_span+a_sent_span+b_sent_span
        return min(coverage), max(coverage)
    
    def reduction_logic(self, row):
        start_char_offset = row.sentences[row.sent_idx_min]['start_char_offset']
        end_char_offset = row.sentences[row.sent_idx_max]['end_char_offset']
        text = row.text[start_char_offset:end_char_offset]
        pronoun_offset = row.pronoun_offset - start_char_offset
        a_offset = row.a_offset - start_char_offset
        b_offset = row.b_offset - start_char_offset
        return text, pronoun_offset, a_offset, b_offset