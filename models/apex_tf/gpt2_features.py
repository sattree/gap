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

import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

from externals.bert.pytorch_extract_features import InputExample, convert_examples_to_features

class GPT2Features(BaseEstimator, TransformerMixin):
    def __init__(self, model='bert-large-uncased'):
        self.model = model
    
    def transform(self, X):
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Load pre-trained model (weights)
        model = GPT2Model.from_pretrained('gpt2', cache_dir='tmp/gpt2/')
        model.eval()
        
        output = []
        for idx, row in tqdm(X.iterrows(), total=len(X)):
            # Encode some inputs
            indexed_tokens_1 = tokenizer.encode(row.text)

            # If you have a GPU, put everything on cuda
            # Convert inputs to PyTorch tensors
            tokens_tensor_1 = torch.tensor([indexed_tokens_1])
            tokens_tensor_1 = tokens_tensor_1.to('cuda')
            model.to('cuda')

            # Predict hidden states features for each layer
            with torch.no_grad():
                hidden_states_1, past = model(tokens_tensor_1)
                
            tokens = [tokenizer.decoder[token].replace('Ġ', '') for token in indexed_tokens_1]
            output.append([tokens, hidden_states_1.cpu()[0]])
                
        output = pd.DataFrame(output, columns=['tokens', 'layer_-1'])
        res = []
        for idx, row in X.iterrows():
            res.append(self.get_sample_props(output.loc[idx], **row)[1:])
        
        res = pd.DataFrame(res, columns=['tokens', 'pronoun_offset_token',
                                                'a_offset_token', 'b_offset_token', 'a_span',
                                                'b_span', 'pronoun_token', 'a_tokens', 'b_tokens', 'bert', 'cls'])
        
        cols = set(X.columns).difference(res.columns)
        return {'X': pd.concat([X[cols], res], axis=1)}
    
    def get_sample_props(self, features, text, pronoun, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
        tokens = features['tokens']
        embs = features['layer_-1']
        cls = features['layer_-1'][0]
        
        #assuming only whitespaces have been removed from text
        # bert tokens have some hashes for word piece
        assert len(''.join(tokens)) == len(text.replace(' ', '')), ([token for token in tokens], text.split(' '))

        idx = [0] + list(map(lambda x: len(x), tokens))
        idx = np.cumsum(idx).tolist()
                
        a_end_idx = a_offset+len(a)
        b_end_idx = b_offset+len(b)

        pronoun_offset = idx.index(len(text[:pronoun_offset].replace(' ', '')))
        pronoun_token = tokens[pronoun_offset]
        
        a_offset = idx.index(len(text[:a_offset].replace(' ', '')))
        token_end = np.where(np.array(idx) >= len(text[:a_end_idx].replace(' ', '')))[0][0] -1
        a_span = [a_offset, token_end]
        a_tokens = tokens[a_offset:token_end+1]
        
        b_offset = idx.index(len(text[:b_offset].replace(' ', '')))
        token_end = np.where(np.array(idx) >= len(text[:b_end_idx].replace(' ', '')))[0][0] -1
        b_span = [b_offset, token_end]
        b_tokens = tokens[b_offset:token_end+1]
        
        return tokens, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens, embs, cls
    