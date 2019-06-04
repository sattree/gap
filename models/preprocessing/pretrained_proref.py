import pandas as pd
import numpy as np

from ..pronoun_resolution import PronounResolutionModel, PronounResolutionModelV2

class PretrainedProref():
    def __init__(self):
        self.syn = PronounResolutionModel(None, n_jobs=1)
        self.par = PronounResolutionModel(None, n_jobs=1)
        self.url = PronounResolutionModel(None, n_jobs=1)
        self.stan = PronounResolutionModelV2(None, n_jobs=1, multilabel=True)
        self.allen = PronounResolutionModelV2(None, n_jobs=1, multilabel=True)
        self.hug = PronounResolutionModelV2(None, n_jobs=1, multilabel=True)
        self.lee = PronounResolutionModelV2(None, n_jobs=1, multilabel=True)

    def transform(self, 
                    X,
                    # stanford,
                    syn=None,
                    par=None,
                    url=None,
                    hug=None,
                    allen=None,
                    lee=None):

        X = X.copy()

        cols = ['tokens', 'clusters', 'pronoun_offset', 'a_span', 'b_span', 'token_to_char_mapping']
        x_cols = set(X.columns).difference(cols)

        if syn is not None:
            data = pd.concat([X[x_cols], pd.DataFrame(syn, columns=cols)], axis=1)
            preds = self.syn.predict(data)
            X['syntactic_distance_a_coref'], X['syntactic_distance_b_coref'] = zip(*preds)
            X['syntactic_distance_neither_coref'] = ~X['syntactic_distance_a_coref'] & ~X['syntactic_distance_b_coref']

        if par is not None:
            data = pd.concat([X[x_cols], pd.DataFrame(par, columns=cols)], axis=1)
            preds = self.par.predict(data)
            X['parallelism_a_coref'], X['parallelism_b_coref'] = zip(*preds)
            X['parallelism_neither_coref'] = ~X['parallelism_a_coref'] & ~X['parallelism_b_coref']

        if url is not None:
            data = pd.concat([X[x_cols], pd.DataFrame(url, columns=cols)], axis=1)
            preds = self.url.predict(data)
            X['parallelism_url_a_coref'], X['parallelism_url_b_coref'] = zip(*preds)
            X['parallelism_url_neither_coref'] = ~X['parallelism_url_a_coref'] & ~X['parallelism_url_b_coref']

        if hug is not None:
            data = pd.concat([X[x_cols], pd.DataFrame(hug, columns=cols)], axis=1)
            preds = self.hug.predict(data)
            X['huggingface_ml_a_coref'], X['huggingface_ml_b_coref'] = zip(*preds)
            X['huggingface_ml_neither_coref'] = ~X['huggingface_ml_a_coref'] & ~X['huggingface_ml_b_coref']

        if allen is not None:
            data = pd.concat([X[x_cols], pd.DataFrame(allen, columns=cols)], axis=1)
            preds = self.allen.predict(data)
            X['allen_ml_a_coref'], X['allen_ml_b_coref'] = zip(*preds)
            X['allen_ml_neither_coref'] = ~X['allen_ml_a_coref'] & ~X['allen_ml_b_coref']

        if lee is not None:
            data = pd.concat([X[x_cols], pd.DataFrame(lee, columns=cols)], axis=1)
            preds = self.lee.predict(data)
            X['lee_a_coref'], X['lee_b_coref'] = zip(*preds)
            X['lee_neither_coref'] = ~X['lee_a_coref'] & ~X['lee_b_coref']

        return {'X': X}