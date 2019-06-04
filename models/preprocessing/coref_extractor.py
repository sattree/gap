import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm

tqdm.pandas(desc='Extracting coref clusters...')

class CorefExtractor():
    def __init__(self,
              syn=None,
              par=None,
              url=None,
              stan=None,
              allen=None,
              hug=None,
              lee=None):

        self.syn = syn
        self.par = par
        self.url = url
        self.stan = stan
        self.allen = allen
        self.hug = hug
        self.lee = lee
        self.backend = 'threading'

    def transform(self, X, *args, **kwargs):
        syn, url, par, allen, hug, lee = None, None, None, None, None, None

        with Parallel(n_jobs=8, verbose=10, backend='loky') as parallel:
            if self.syn is not None:
                syn = parallel([delayed(self.syn.predict)(**row) 
                                            for idx, row in X.iterrows()])
            if self.url is not None:
                url = parallel([delayed(self.url.predict)(**row) 
                                            for idx, row in X.iterrows()])

        # if self.url is not None:
            # url = X.progress_apply(lambda x: self.url.predict(**x) , axis=1).values.tolist()

        # with Parallel(n_jobs=4, verbose=10, backend='loky') as parallel:
        #   stanford = parallel([delayed(self.statistical_stanford_proref_model.predict)(**row) 
        #                                     for idx, row in X.iterrows()])

        if self.par is not None:
            par = X.progress_apply(lambda x: self.par.predict(**x) , axis=1).values.tolist()

        if self.allen is not None:
            allen = X.progress_apply(lambda x: self.allen.predict(**x) , axis=1).values.tolist()

        with Parallel(n_jobs=16, verbose=10, backend=self.backend) as parallel:
            if self.hug is not None:
                hug = parallel([delayed(self.hug.predict)(**row) 
                                            for idx, row in X.iterrows()])

            if self.lee is not None:
                lee = parallel([delayed(self.lee.predict)(**row) 
                                            for idx, row in X.iterrows()])

        return {
                'syn': syn,
                'par': par,
                'url': url,
                'allen': allen,
                'hug': hug,
                'lee': lee
                # 'stanford': stanford,
                }