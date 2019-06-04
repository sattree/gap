from ..base.coref import Coref
from ..base.spacy_base import SpacyModel

class AllenNLPCorefModel(Coref, SpacyModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        super().__init__(tokenizer)
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, id=None, debug=False, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)

        clusters = []
        
        res = self.model.predict(text)
        assert res['document'] == tokens, 'Tokens from coref dont match.'
        
        token_to_char_mapping = [token.idx for token in doc]

        return tokens, res['clusters'], pronoun_offset, a_span, b_span, token_to_char_mapping