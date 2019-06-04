import subprocess
import os
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from attrdict import AttrDict
from tqdm import tqdm

import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from externals.bert.pytorch_extract_features import InputExample, convert_examples_to_features

class BERTFeaturesV2(BaseEstimator, TransformerMixin):
    def __init__(self, model='bert-large-uncased', use_cuda=True):
        self.model = model
        
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
        logger = logging.getLogger(__name__)

        self.args = args = AttrDict({
            'bert_model': self.model,
            'do_lower_case': True,
            'layers': "-1,-2,-3,-4",
            'max_seq_length': 512,
            'batch_size': 2,
            'local_rank': -1,
            'no_cuda': not use_cuda
        })
        
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        
        logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

        print('loading from model')
        model = BertModel.from_pretrained('results/bert_finetuned/lm/', cache_dir='results/bert_finetuned/lm/')
        print('loaded model')
        model.to(device)
        
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        model.eval()
        
        self.device = device
        self.model = model
    
    def transform(self, X):
        tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case, cache_dir='tmp/')

        examples = []

        for idx, row in X.iterrows():
            examples.append(InputExample(unique_id=idx, text_a=row.text, text_b=None))

        features = convert_examples_to_features(
            examples=examples, seq_length=self.args.max_seq_length, tokenizer=tokenizer)
        
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        if self.args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.batch_size)
        
        layer_indexes = [int(x) for x in self.args.layers.split(",")]

        output = []
        for input_ids, input_mask, example_indices in tqdm(eval_dataloader):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            all_encoder_layers, _ = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = int(feature.unique_id)
                tokens = []
                layers = [[] for _ in layer_indexes]
                all_out_features = []
                for (i, token) in enumerate(feature.tokens):
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                        layers[j].append([round(x.item(), 6) for x in layer_output[i]])
                    tokens.append(token)
                output.append([tokens, *layers])
                
        output = pd.DataFrame(output, columns=['tokens', *['layer_{}'.format(idx) for idx in layer_indexes]])
        res = []
        for idx, row in X.iterrows():
            res.append(self.get_sample_props(output.loc[idx], layer_indexes, **row)[1:])
        
        res = pd.DataFrame(res, columns=['tokens', 'pronoun_offset_token',
                                                'a_offset_token', 'b_offset_token', 'a_span',
                                                'b_span', 'pronoun_token', 'a_tokens', 'b_tokens', 'bert', 'cls'])
        
        cols = set(X.columns).difference(res.columns)
        return {'X': pd.concat([X[cols], res], axis=1)}
    
    def get_sample_props(self, features, layer_indexes, text, pronoun, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        cls = [features['layer_{}'.format(idx)][0] for idx in layer_indexes]
        tokens = features['tokens'][1:-1]
        embs = [features['layer_{}'.format(idx)][1:-1] for idx in layer_indexes]
        
        #assuming only whitespaces have been removed from text
        # bert tokens have some hashes for word piece
        assert len(''.join(tokens).replace('##', '')) == len(text.replace(' ', '')), ([token.replace('##', '') for token in tokens], text.split(' '))
        
        idx = [0] + list(map(lambda x: len(x.replace('##', '')), tokens))
        idx = np.cumsum(idx).tolist()
                
        a_end_idx = a_offset+len(a)
        b_end_idx = b_offset+len(b)

        pronoun_offset = idx.index(len(text[:pronoun_offset].replace(' ', '')))
        pronoun_token = tokens[pronoun_offset]
        
        a_offset = idx.index(len(text[:a_offset].replace(' ', '')))
        token_end = idx.index(len(text[:a_end_idx].replace(' ', '')))-1
        a_span = [a_offset, token_end]
        a_tokens = tokens[a_offset:token_end+1]
        
        b_offset = idx.index(len(text[:b_offset].replace(' ', '')))
        token_end = idx.index(len(text[:b_end_idx].replace(' ', '')))-1
        b_span = [b_offset, token_end]
        b_tokens = tokens[b_offset:token_end+1]
        
        return tokens, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens, embs, cls
    
class BERTFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model='uncased_L-12_H-768_A-12'):
        self.model = model
    
    def transform(self, X):
        def cleanser(row):
            pronoun_offset = row.pronoun_offset
            a_offset = row.a_offset
            b_offset = row.b_offset
            text = row.text.replace("`", "'")
            matches = re.findall('\([^)]*\)', text)
            for match in matches:
                if row.a in match or row.b in match or row.pronoun in match:
                    continue
                
                if text.index(match) < pronoun_offset:
                    pronoun_offset -= len(match)
                    
                if text.index(match) < a_offset:
                    a_offset -= len(match)
                    
                if text.index(match) < b_offset:
                    b_offset -= len(match)
                    
                text = text.replace(match, '', 1)
            
            return text, pronoun_offset, a_offset, b_offset
            
        # X[['text', 'pronoun_offset', 'a_offset', 'b_offset']] = X.apply(cleanser, axis=1, result_type='expand')
            
        # X.text.to_csv('tmp/input.txt', index = False, header = False, quoting=csv.QUOTE_NONE)
        with open('tmp/input.txt', 'w') as f:
            f.write('\n'.join(X.text.values.tolist()))
        
        cmd = "cd bert && python3 extract_features.py \
              --input_file=../tmp/input.txt \
              --output_file=../tmp/output.jsonl \
              --vocab_file={0}/vocab.txt \
              --bert_config_file={0}/bert_config.json \
              --init_checkpoint={0}/bert_model.ckpt \
              --layers=-1 \
              --max_seq_length=512 \
              --batch_size=32".format(self.model)
        
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in iter(p.stdout.readline, ''):
            print(line)
        retval = p.wait()
        
        bert_output = pd.read_json("tmp/output.jsonl", lines = True)
                
        res = []
        for idx, row in X.iterrows():
            features = pd.DataFrame(bert_output.loc[idx,"features"])
            res.append(self.get_sample_props(features, **row)[1:])
            
        os.system("rm tmp/output.jsonl")
        os.system("rm tmp/input.txt")
        
        
        res = pd.DataFrame(res, columns=['tokens', 'pronoun_offset_token',
                                                'a_offset_token', 'b_offset_token', 'a_span',
                                                'b_span', 'pronoun_token', 'a_tokens', 'b_tokens', 'bert', 'cls'])
        
        cols = set(X.columns).difference(res.columns)
        return {'X': pd.concat([X[cols], res], axis=1)}
    
    def get_sample_props(self, features, text, pronoun, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        embs = []
        cls = []
        tokens = []
        tokens_ = [features.loc[j,"token"] for j in range(len(features))]
        for j in range(len(features)):
            token = features.loc[j,"token"]
            if token in ['[CLS]', '[SEP]']:# or (token == '"' and j in [1, len(features)-2]):
                if token == '[CLS]':
                    cls.append(features.loc[j,"layers"][0]['values'])
                continue
                
            tokens.append(token)
            embs.append(features.loc[j,"layers"][0]['values'])#+features.loc[j,"layers"][1]['values'])
            
        #assuming only whitespaces have been removed from text
        # bert tokens have some hashes for word piece
        assert len(''.join(tokens).replace('##', '')) == len(text.replace(' ', '')), ([token.replace('##', '') for token in tokens], text.split(' '))
        
        idx = [0] + list(map(lambda x: len(x.replace('##', '')), tokens))
        idx = np.cumsum(idx).tolist()
                
        a_end_idx = a_offset+len(a)
        b_end_idx = b_offset+len(b)

        pronoun_offset = idx.index(len(text[:pronoun_offset].replace(' ', '')))
        pronoun_token = tokens[pronoun_offset]
        
        a_offset = idx.index(len(text[:a_offset].replace(' ', '')))
        token_end = idx.index(len(text[:a_end_idx].replace(' ', '')))-1
        a_span = [a_offset, token_end]
        a_tokens = tokens[a_offset:token_end+1]
        
        b_offset = idx.index(len(text[:b_offset].replace(' ', '')))
        token_end = idx.index(len(text[:b_end_idx].replace(' ', '')))-1
        b_span = [b_offset, token_end]
        b_tokens = tokens[b_offset:token_end+1]
        
        return tokens, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens, embs, cls