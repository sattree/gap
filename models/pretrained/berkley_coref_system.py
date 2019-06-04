import os
import subprocess
import shutil

from tqdm import tqdm
from collections import defaultdict
from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes

from ..base.coref import Coref
from ..base.stanford_base import StanfordModel

class BCS(Coref, StanfordModel):
    def __init__(self, model):
        self.model = model
        super().__init__(model)

    @staticmethod
    def preprocess(df, root_dir):
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.makedirs('{}/text'.format(root_dir))
        os.makedirs('{}/preprocessed'.format(root_dir))
        os.makedirs('{}/coref'.format(root_dir))
        os.makedirs('{}/logs'.format(root_dir))

        for idx, row in tqdm(df.iterrows()):
            with open('{}/text/{}'.format(root_dir, row.id), 'w', encoding='utf-8') as f:
                f.write(row.text)

        print('Running BCS preprocessor')
        subprocess.run('cd berkeley-entity && java -Xmx1g -cp berkeley-entity-1.0.jar edu.berkeley.nlp.entity.preprocess.PreprocessingDriver ++config/base.conf -execDir ../{0}/logs -inputDir ../{0}/text -outputDir ../{0}/preprocessed'.format(root_dir), shell=True, stdout=None, stderr=None)
        print('Running BCS coref')
        subprocess.run('cd berkeley-entity && java -Xmx1g -cp berkeley-entity-1.0.jar edu.berkeley.nlp.entity.Driver ++config/base.conf -execDir ../{0}/logs/ -mode COREF_PREDICT -modelPath models/coref-onto.ser.gz -testPath ../{0}/preprocessed/ -outputPath ../{0}/coref -corefDocSuffix ""'.format(root_dir), shell=True, stdout=None, stderr=None)
        
        # os.remove('tmp/text/{}'.format(id))
        # os.remove('tmp/preprocessed/{}'.format(id))
        # os.remove('tmp/coref/{}-0.pred_conll'.format(id))
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, id=None, debug=False, **kwargs):
        doc, tokens_, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
        
        # try:
        #     os.makedirs('tmp/text/{0}/'.format(id), exist_ok=True)
        #     os.makedirs('tmp/preprocessed/{0}/'.format(id), exist_ok=True)
        #     os.makedirs('tmp/coref/{0}/'.format(id), exist_ok=True)

        # except OSError:
        #     pass

        # with open('tmp/text/{0}/{0}'.format(id), 'w', encoding='utf-8') as f:
        #     f.write(text)

        # subprocess.run('cd berkeley-entity && java -Xmx1g -cp berkeley-entity-1.0.jar edu.berkeley.nlp.entity.preprocess.PreprocessingDriver ++config/base.conf -execDir ../tmp/logs -inputDir ../tmp/text/{0}/ -outputDir ../tmp/preprocessed/{0}/'.format(id), shell=True, stdout=None, stderr=None)
        # subprocess.run('cd berkeley-entity && java -Xmx1g -cp berkeley-entity-1.0.jar edu.berkeley.nlp.entity.Driver ++config/base.conf -execDir ../tmp/logs/ -mode COREF_PREDICT -modelPath models/coref-onto.ser.gz -testPath ../tmp/preprocessed/{0}/ -outputPath ../tmp/coref/{0}/ -corefDocSuffix ""'.format(id), shell=True, stdout=None, stderr=None)
        

        data = Ontonotes().dataset_document_iterator('{}/coref/{}-0.pred_conll'.format(root_dir, id))
        for i, doc in enumerate(data):
            tokens = []
            clusters = defaultdict(list)
            for fi in doc:
                for c in fi.coref_spans:
                    clusters[c[0]].append([len(tokens)+c[1][0], len(tokens)+c[1][1]])
                tokens += fi.words
                
        tokens = [token.replace('\\*', '*').replace('-LRB-', '(').replace('-RRB-', ')') for token in tokens]
        
        if any([token not in tokens for token in tokens_[a_span[0]:a_span[1]]+tokens_[b_span[0]:b_span[1]]]):
            print('Tokens dont match', tokens, tokens_, a, b)

        return tokens, list(clusters.values()), pronoun_offset, a_span, b_span