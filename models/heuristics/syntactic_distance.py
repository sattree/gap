import nltk

from ..base.coref import Coref
from ..base.spacy_base import SpacyModel
from ..base.stanford_base import StanfordModel
from ..base.utils import parse_tree_to_graph, get_syntactical_distance_from_graph

class StanfordSyntacticDistanceModel(Coref, StanfordModel):
    # NOTES
    # Only applies to the scenarios where pronoun and both mentions are in the same sentence
    # For the cases where only one mention is in the same sentence as pronoun, the model will automatically resolve to it.
    #     Bad heuristic - we should only consider the cases where all three reside in the same sentence
    # Edit: Applies to all scenarios, needs fruther thought
    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug
    
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, id=None, debug=False, **kwargs):
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)

        clusters = []
        
        try:

            trees = self.model.parse_text(text)#, properties={'options': 'strictTreebank3=false,normalizeSpace=True'})
            graph = parse_tree_to_graph(trees, doc)

            for token in a_tokens:
                token.syn_dist = get_syntactical_distance_from_graph(graph, token, pronoun_token)
                
            for token in b_tokens:
                token.syn_dist = get_syntactical_distance_from_graph(graph, token, pronoun_token)
                
            # Assumption - first token in multi word entity is representative of the whole entity
            candidate_a = sorted(filter(lambda token: token.syn_dist, a_tokens[:1]), key=lambda token: token.syn_dist)
            candidate_b = sorted(filter(lambda token: token.syn_dist, b_tokens[:1]), key=lambda token: token.syn_dist)
            
            if not len(candidate_a) or not len(candidate_b):
                if debug: 
                    print('{}, Syntactic distance doesn\'t apply to at least one of the candidates'.format(id))
            
            elif len(candidate_a) and len(candidate_b) and candidate_a[0].syn_dist == candidate_b[0].syn_dist:
                if debug: 
                    print('{}, Both mentions have the same syntactic distance'.format(id))

            else:
                candidates = sorted(candidate_a + candidate_b, key=lambda token: token.syn_dist)
                if len(candidates):
                    candidate = candidates[0]
                    clusters.append([[pronoun_offset, pronoun_offset], [candidate.i, candidate.i]])
                
        except Exception as e:
            print('{}, {}'.format(id, e))

        token_to_char_mapping = [token.idx for token in doc]

        return tokens, clusters, pronoun_offset, a_span, b_span, token_to_char_mapping

class AllenNLPSyntacticDistanceModel(Coref, SpacyModel):
    # NOTES
    # Only applies to the scenarios where pronoun and both mentions are in the same sentence
    # For the cases where only one mention is in the same sentence as pronoun, the model will automatically resolve to it.
    #     Bad heuristic - we should only consider the cases where all three reside in the same sentence
    # Edit: Applies to all scenarios, needs fruther thought
    def __init__(self, model, tokenizer, debug=False):
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
        
        try:

            sents = self.tokenizer(text).sents
            trees = [nltk.Tree.fromstring(self.model.predict(sentence=sent.text)['trees']) for sent in sents]
            graph = parse_tree_to_graph(trees, doc)

            for token in a_tokens:
                token.syn_dist = get_syntactical_distance_from_graph(graph, token, pronoun_token)
                
            for token in b_tokens:
                token.syn_dist = get_syntactical_distance_from_graph(graph, token, pronoun_token)
                
            # Assumption - first token in multi word entity is representative of the whole entity
            candidate_a = sorted(filter(lambda token: token.syn_dist, a_tokens[:1]), key=lambda token: token.syn_dist)
            candidate_b = sorted(filter(lambda token: token.syn_dist, b_tokens[:1]), key=lambda token: token.syn_dist)
            
            if not len(candidate_a) or not len(candidate_b):
                if debug: 
                    print('{}, Syntactic distance doesn\'t apply to at least one of the candidates'.format(id))
            
            elif len(candidate_a) and len(candidate_b) and candidate_a[0].syn_dist == candidate_b[0].syn_dist:
                if debug: 
                    print('{}, Both mentions have the same syntactic distance'.format(id))

            else:
                candidates = sorted(candidate_a + candidate_b, key=lambda token: token.syn_dist)
                if len(candidates):
                    candidate = candidates[0]
                    clusters.append([[pronoun_offset, pronoun_offset], [candidate.i, candidate.i]])
                
        except Exception as e:
            print('{}, {}'.format(id, e))

        token_to_char_mapping = [token.idx for token in doc]

        return tokens, clusters, pronoun_offset, a_span, b_span, token_to_char_mapping