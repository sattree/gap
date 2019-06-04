from ..base.coref import Coref
from ..base.spacy_base import SpacyModel
from ..base.stanford_base import StanfordModel
from ..base.utils import get_normalized_tag

class SpacyParallelismModel(Coref, SpacyModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
        
        candidates = [token for token in a_tokens] + [token for token in b_tokens]
        candidates = sorted(candidates, key=lambda token: abs(token.i - pronoun_offset))
        
        clusters = []
        if get_normalized_tag(pronoun_token) in ['subj', 'dobj']:
            for candidate in candidates:
                if get_normalized_tag(candidate) == get_normalized_tag(pronoun_token):
                    clusters.append([[pronoun_offset, pronoun_offset], [candidate.i, candidate.i]])
                    break
                    
        return tokens, clusters, pronoun_offset, a_span, b_span

class AllenNLPParallelismModel(Coref, SpacyModel):
    def __init__(self, model, tokenizer):
        super().__init__(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
        
        token_to_char_mapping = [token.idx for token in self.tokenizer(text)]

        clusters = []

        try:
            deps = []
            words = []
            for sent in self.tokenizer(text).sents:
                preds = self.model.predict(sentence=sent.text)
                deps += preds['predicted_dependencies']
                words += preds['words']

            assert words == tokens, 'Dependency parse and tokenizer tokens dont match.'

            for i, token in enumerate(doc):
                token.dep_ = deps[i]

            candidates = [token for token in a_tokens] + [token for token in b_tokens]
            candidates = sorted(candidates, key=lambda token: abs(token.i - pronoun_offset))

            if get_normalized_tag(pronoun_token) in ['subj', 'dobj']:
                for candidate in candidates:
                    if get_normalized_tag(candidate) == get_normalized_tag(pronoun_token):
                        clusters.append([[pronoun_offset, pronoun_offset], [candidate.i, candidate.i]])
                        break
        except Exception as e:
            print(e)
                    
        token_to_char_mapping = [token.idx for token in doc]

        return tokens, clusters, pronoun_offset, a_span, b_span, token_to_char_mapping

class StanfordParallelismModel(Coref, StanfordModel):
    def __init__(self, model, dependency_parser):
        self.model = model
        self.dep_parser = dependency_parser
    
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        doc = self.get_dependency_tags(text)
        
        a_end_idx = a_offset+len(a)-1
        b_end_idx = b_offset+len(b)-1

        pronoun_offset = next(filter(lambda token: pronoun_offset in range(token.characterOffsetBegin, token.characterOffsetEnd), doc), None).index
        a_offset = next(filter(lambda token: a_offset in range(token.characterOffsetBegin, token.characterOffsetEnd), doc), None).index
        b_offset = next(filter(lambda token: b_offset in range(token.characterOffsetBegin, token.characterOffsetEnd), doc), None).index
        
        token_end = next(filter(lambda token: a_end_idx in range(token.characterOffsetBegin, token.characterOffsetEnd), doc), None).index
        a_span = [a_offset, token_end]
        a_tokens = doc[a_offset:token_end+1]
        
        token_end = next(filter(lambda token: b_end_idx in range(token.characterOffsetBegin, token.characterOffsetEnd), doc), None).index
        b_span = [b_offset, token_end]
        b_tokens = doc[b_offset:token_end+1]
        
        pronoun_token = doc[pronoun_offset]
        
        candidates = [token for token in a_tokens] + [token for token in b_tokens]
        candidates = sorted(candidates, key=lambda token: abs(token.index - pronoun_offset))
        
        clusters = []
        if get_normalized_tag(pronoun_token) in ['subj', 'dobj']:
            for candidate in candidates:
                if get_normalized_tag(candidate) == get_normalized_tag(pronoun_token):
                    clusters.append([[pronoun_offset, pronoun_offset], [candidate.index, candidate.index]])
                    break
                    
        return [token.originalText for token in doc], clusters, pronoun_offset, a_span, b_span
        
    def map_chars_to_tokens(self, text):
        res = self.model.api_call(text, properties={'annotators': 'tokenize,ssplit'})
        res = AttrDict(res)
        sent_lens = [0]+[len(sent.tokens)for sent in res.sentences]
        sent_lens = np.cumsum(sent_lens)
        
        # reset indexes
        tokens = []
        for i, sent in enumerate(res.sentences):
            assert i == sent.index
            for j, token in enumerate(sent.tokens):
                assert j+1 == token.index
                token.index += sent_lens[i] - 1
                tokens.append(token)

        return tokens
    
    def get_dependency_tags(self, text):
        doc = self.map_chars_to_tokens(text)

        sents = self.dep_parser.parse_text(text)
        # 0 index contains ROOT node
        dependencies = chain(*[[sent.nodes[i] for i in range(1, len(sent.nodes))] for sent in sents])
        
        for idx, node in enumerate(dependencies):
            assert node['word'] == doc[idx].word
            doc[idx].dep_ = node['rel']
        return doc
        