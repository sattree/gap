import os
import pyhocon
import tensorflow as tf

from ..base.coref import Coref
from ..base.spacy_base import SpacyModel

from externals.modified_e2e_coref import coref_model
from externals.modified_e2e_coref import util

class LeeEtAl2017(Coref, SpacyModel):
    def __init__(self, tokenizer, config, verbose=0):
        util.set_gpus(config['device'])

        config_name = config['name']
        model_config = pyhocon.ConfigFactory.parse_file(config['model'])[config_name]
        model_config['log_dir'] = util.mkdirs(config['log_root'])
        model_config['context_embeddings']['path'] = os.path.join(config['context_embeddings_root'], 
                                                                    model_config['context_embeddings'].path)
        model_config['head_embeddings']['path'] = os.path.join(config['head_embeddings_root'], 
                                                                model_config['head_embeddings'].path)
        model_config['char_vocab_path'] = os.path.join(config['char_vocab_root'], 
                                                                model_config['char_vocab_path'])

        if verbose:
            print(pyhocon.HOCONConverter.convert(model_config, "hocon"))

        tf.reset_default_graph()

        model = coref_model.CorefModel(model_config)
        
        config = tf.ConfigProto(device_count = {'GPU': 0})
        session = tf.Session(config=config)
        model.restore(session)

        self.model = model
        self.session = session
        
        self.tokenizer = tokenizer
        super().__init__(tokenizer)
        
    def predict(self, 
                text, 
                a, 
                b, 
                pronoun_offset, 
                a_offset, 
                b_offset, 
                id=None, 
                debug=False, 
                return_features=False,
                **kwargs):
    
        doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
        
        sentences = [[token.text for token in sent] for sent in self.tokenizer(text).sents]
        token_to_char_mapping = [token.idx for token in self.tokenizer(text)]
        speakers = [["" for _ in sentence] for sentence in sentences]
        example = {
            "doc_key": "nw",
            "clusters": [],
            "sentences": sentences,
            "speakers": speakers,
        }

        tensorized_example = self.model.tensorize_example(example, is_training=False)
        feed_dict = {i:t for i,t in zip(self.model.input_tensors, tensorized_example)}
        _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = self.session.run(self.model.predictions + [self.model.head_scores], feed_dict=feed_dict)

        predicted_antecedents = self.model.get_predicted_antecedents(antecedents, antecedent_scores)
        
        example["mention_starts"] = mention_starts
        example["mention_ends"] = mention_ends
        example["antecedents"] = antecedents
        example["antecedent_scores"] = antecedent_scores

        example["predicted_antecedents"] = predicted_antecedents

        example["predicted_clusters"], _ = self.model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
        example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
        example["head_scores"] = head_scores.tolist()

        
        if return_features:
            return (tokens, token_to_char_mapping, pronoun_offset, a_span, b_span, 
                    example["predicted_clusters"],
                    mention_starts,
                    mention_ends,
                    antecedents,
                    antecedent_scores,
                    predicted_antecedents,
                    example["top_spans"],
                    head_scores
                )


        return tokens, example["predicted_clusters"], pronoun_offset, a_span, b_span, token_to_char_mapping