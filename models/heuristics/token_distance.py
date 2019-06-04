import random

from ..base.coref import Coref
from ..base.spacy_base import SpacyModel

class TokenDistanceModel(Coref, SpacyModel):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
            
        token_to_char_mapping = [token.idx for token in doc]

        # Assuming first token represents the whole mention
        candidates = [token for token in a_tokens[:1]] + [token for token in b_tokens[:1]]
        candidates = sorted(candidates, key=lambda token: abs(token.i - pronoun_offset))
        
        # probably want to resolve equidistant mentions randomly
        # two candidates can only be equidistant from pronoun if they come from different mentions
        clusters = []
        dist_a = abs(candidates[0].i - pronoun_offset)
        dist_b = abs(candidates[1].i - pronoun_offset)

        if dist_a != dist_b:
            candidate = candidates[0]
            clusters = [[[pronoun_offset, pronoun_offset], [candidate.i, candidate.i]]]

        return tokens, token_to_char_mapping, clusters, pronoun_offset, a_span, b_span