from ..base.coref import Coref
from ..base.spacy_base import SpacyModel

class HuggingfaceCorefModel(Coref, SpacyModel):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, id=None, debug=False, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
        
        token_to_char_mapping = [token.idx for token in doc]

        clusters = []
        if doc._.coref_clusters is not None:
            for cluster in doc._.coref_clusters:
                clusters.append([])
                for mention in cluster.mentions:
                    clusters[-1].append([mention.start, mention.end-1])
        
        return tokens, clusters, pronoun_offset, a_span, b_span, token_to_char_mapping