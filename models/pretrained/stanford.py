import numpy as np

from ..base.coref import Coref
from ..base.stanford_base import StanfordModel
from ..base.utils import api_call

class StanfordCorefModel(Coref, StanfordModel):
    def __init__(self, model, algo='neural', greedyness=0.5):
        self.model = model
        self.algo = algo
        self.greedyness = greedyness
    
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, id=None, debug=False, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)

        clusters = []
        
        props = {'annotators': 'pos, lemma, ner, parse, coref', 
                                           'coref.statisical.pairwiseScoreThresholds': 1,
                                           'coref.maxMentionDistance': 50
                                          }
        props.update({'coref.algorithm': self.algo,
                        'coref.neural.greedyness': self.greedyness})
        data = api_call(self.model, text, props)
        
        sents = []
        for sent in data['sentences']:
            sents.append([])
            for token in sent['tokens']:
                sents[-1].append(token['originalText'])
                
        assert sum(sents, []) == tokens, 'The tokens in coref dont match.'

        clusters = []
        if data['corefs'] is not None:
            for num, mentions in data['corefs'].items():
                clusters.append([])
                for mention in mentions:
                    start = np.cumsum([0]+list(map(len, sents)))[mention['sentNum']-1] + mention['startIndex']-1
                    end = np.cumsum([0]+list(map(len, sents)))[mention['sentNum']-1] + mention['endIndex']-2
                    clusters[-1].append([start, end])

        token_to_char_mapping = [token.idx for token in doc]

        return tokens, clusters, pronoun_offset, a_span, b_span, token_to_char_mapping