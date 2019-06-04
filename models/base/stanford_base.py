from attrdict import AttrDict
import numpy as np

from .utils import map_chars_to_tokens

class StanfordModel:
    def __init__(self, model):
        self.model = model
    
    def tokenize(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        # needs to make options work for certain scenarios, might affect downstream applications
        # Example - fractions are not pslit by white-space
        #           but syntactic parsing splits the fractions
        res = self.model.api_call(text, properties={'annotators': 'tokenize,ssplit'})
                                                    #, 'options': 'tokenizeNLs=true,strictTreebank3=false,normalizeSpace=True'})
        res = AttrDict(res)

        sent_lens = [0]+[len(sent.tokens)for sent in res.sentences]
        sent_lens = np.cumsum(sent_lens)
        
        # Stanford token indexing start at 1
        # rename keys: index -> i, 
        #              characterOffsetBegin -> idx,
        #              originalText -> text
        # rename keys, to induce a uniform api between stanford and spacy
        # remember allennlp and huggingface both use spacy under the hood
        doc = []
        for i, sent in enumerate(res.sentences):
            assert i == sent.index
            for j, token in enumerate(sent.tokens):
                assert j+1 == token.index

                doc.append(AttrDict({
                        'i': token.index + sent_lens[i] - 1,
                        'idx': token.characterOffsetBegin,
                        'text': token.originalText,
                        # word is normalized, list '(' -> '-LRB-'
                        # prase tree contains words
                        'word': token.word
                    }))

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

        
        tokens = [tok.text for tok in doc]
        
        return doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens