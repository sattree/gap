import pandas as pd

from tqdm import tqdm
from sklearn.externals.joblib import Parallel, delayed

class PronounResolutionModel:
    def __init__(self, coref_model, n_jobs=1, verbose=1, backend='threading'):
        self.model = coref_model
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend

    def batch_predict(fn):
        def _predict(self, df, preprocessor=None, **kwargs):
            # print('Is given instance a df? ', isinstance(df, pd.DataFrame))
            if isinstance(df, pd.DataFrame):
                if preprocessor:
                    preprocessor(df)
                
                rows = []
                if self.n_jobs != 1:
                    with Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend=self.backend) as parallel:
                        rows = parallel([delayed(fn)(*(self, row), **kwargs) for idx, row in df.iterrows()])
                else:
                    with tqdm(total=df.shape[0]) as pbar:
                        for idx, row in df.iterrows():
                            rows.append(fn(self, row, **{**row, **kwargs}))
                            pbar.update()
                return rows
            else:
                return fn(self, df, **kwargs)
        return _predict

    @batch_predict
    def predict(self, 
                x, 
                tokens=None, 
                clusters=None, 
                pronoun_offset=None, 
                a_span=None, 
                b_span=None,
                debug=False,
                **kwargs):
        if clusters is None:
            tokens, clusters, pronoun_offset, a_span, b_span, _ = self.model.predict(**x)

        pred = [False, False]
        for cluster in clusters:
            for mention in cluster:
                # if the cluster contains pronoun
                if mention[0] == pronoun_offset and mention[1] == pronoun_offset:
                    for mention in cluster:
                        # some part of token is covered as mention
                        if a_span[0] <= mention[0] and a_span[1] >= mention[1]:
                            pred = [True, False]
                        elif b_span[0] <= mention[0] and b_span[1] >= mention[1]:
                            pred = [False, True]

        if debug:
            return [pronoun_offset, pronoun_offset], a_span, b_span, tokens, clusters, pred[0], pred[1]
        
        return pred

# override pronoun resolution predict method
class PronounResolutionModelV2(PronounResolutionModel):
    # Right now the model looks to see if some part of coref mention is present in the span of gold-two-mention
    # This was done bcoz the distance models compute the distance of only first token in multi word entities.
    #    and would return only a single token in predicted mention
    # Pronoun resolution is a more general purpose wrapper, so the logic should be more generic.
    # Needs fix?
    # I think full containment is not necessary, if there is some overlap in both 
    #      multi-word entities, then it should be sufficient
    #      - may lead to partial matches, but I think that's sufficient for named entites and two 
    #         different entities will probably be well separated
    # If both gold-mentions are present in the predicted mention?
    #      - I think this should be resolved by a downstream resolver
    #      - An exception should be thrown
    # Certain coref resolvers have a tendency to link longer references
    # Both gold-mentions in the same cluster?
    #    - Reject both
    #    - But useful information for confidence model
    #    - can backoff to heuristics
    #           - There should be a difference between linking both vs none, can be captured by confidence model
    def __init__(self, coref_model, multilabel=False, n_jobs=1, verbose=1, backend='threading'):
        self.model = coref_model
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend
        self.multilabel = multilabel
    
    # Need a neater way to do this
    batch_predict = PronounResolutionModel.batch_predict
    
    @batch_predict
    def predict(self, 
                x, 
                tokens=None, 
                clusters=None, 
                pronoun_offset=None, 
                a_span=None, 
                b_span=None,
                debug=False,
                **kwargs):
        if clusters is None:
            tokens, clusters, pronoun_offset, a_span, b_span, _ = self.model.predict(**x)

        pred = [False, False]
        for cluster in clusters:
            for mention in cluster:
                # if the cluster contains pronoun
                if mention[0] == pronoun_offset and mention[1] == pronoun_offset:
                    for mention in cluster:
                        # some part of token is covered as mention
                        # what if we just look for an overlap
                        if self.has_overlap(mention, a_span):
                            pred[0] = True
                        elif self.has_overlap(mention, b_span):
                            pred[1] = True
#                         if mention[0] <= a_span[0] and mention[1] >= a_span[0]:
#                             pred = [True, False]
#                         elif mention[0] <= b_span[0] and mention[1] >= b_span[0]:
#                             pred = [False, True]

        if pred[0] and pred[1]:
            msg = '{}, The pronoun was resolved to both gold mentions.'.format(x.id)
            if debug:
                raise Exception(msg)
            elif self.verbose:
                print(msg)
            if not self.multilabel:
                pred = [False, False]
            
            
        if debug:
            return [pronoun_offset, pronoun_offset], a_span, b_span, tokens, clusters, pred[0], pred[1]
        
        return pred
    
    def has_overlap(self, mention_pred, mention_gold):
        return mention_pred[0] <= mention_gold[1] and mention_pred[1] >= mention_gold[0]