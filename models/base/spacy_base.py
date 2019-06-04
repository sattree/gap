from .utils import map_chars_to_tokens

class SpacyModel:
    def __init__(self, model):
        self.tokenizer = model
    
    def tokenize(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        doc = self.tokenizer(text)

        tokens = [tok.text for tok in doc]

        a_end_idx = a_offset+len(a)-1
        b_end_idx = b_offset+len(b)-1

        pronoun_offset = map_chars_to_tokens(doc, pronoun_offset)
        pronoun_token = doc[pronoun_offset]
        
        a_offset = map_chars_to_tokens(doc, a_offset)
        token_end = map_chars_to_tokens(doc, a_end_idx)
        a_span = [a_offset, token_end]
        a_tokens = doc[a_offset:token_end+1]
        
        b_offset = map_chars_to_tokens(doc, b_offset)
        token_end = map_chars_to_tokens(doc, b_end_idx)
        b_span = [b_offset, token_end]
        b_tokens = doc[b_offset:token_end+1]

                
        return doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens