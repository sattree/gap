import random

from ..base.coref import Coref
from ..base.spacy_base import SpacyModel

class RandomModel(Coref, SpacyModel):
    def __init__(self, model):
        self.model = model
        super().__init__(model)
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                    a, 
                                                                                    b, 
                                                                                    pronoun_offset, 
                                                                                    a_offset, 
                                                                                    b_offset, 
                                                                                    **kwargs)
            
        clusters = []
            
        if random.random() < 0.5:
            clusters.append([[pronoun_offset, pronoun_offset], [a_offset, a_offset]])
        else:
            clusters.append([[pronoun_offset, pronoun_offset], [b_offset, b_offset]])

        return tokens, clusters, pronoun_offset, a_span, b_span