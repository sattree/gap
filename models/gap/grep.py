import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertConfig, BertModel, BertPooler, BertPreTrainedModel

from .evidence_pooling import EvidencePooler

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class GREP(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        self.pooler = BertPooler(config)

        self.evidence_pooler_p = EvidencePooler(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, num_labels)

        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.apply(self.init_bert_weights)

    def forward(self, 
                input_ids, 
                token_type_ids=None, 
                attention_mask=None, 
                gpr_tags_mask=None,
                mention_p_ids=None,
                mention_a_ids=None,
                mention_b_ids=None,
                mention_p_mask=None,
                mention_a_mask=None,
                mention_b_mask=None,
                cluster_ids_a=None,
                cluster_mask_a=None,
                cluster_ids_b=None,
                cluster_mask_b=None,
                cluster_ids_p=None,
                cluster_mask_p=None,
                pretrained=None,
                labels=None,
                training=False,
                eval_mode=False):
        sequence_output, pooled_output = self.bert(input_ids, 
                                                    token_type_ids, 
                                                    attention_mask, 
                                                    output_all_encoded_layers=False)
        
        batch_size = sequence_output.size()[0]
        sequence_output = sequence_output[~gpr_tags_mask].view(batch_size, -1, self.config.hidden_size)
        mention_p_ids = mention_p_ids.unsqueeze(-1)
        mention_p_ids = mention_p_ids.repeat(1, 1, self.config.hidden_size)

        p_output = torch.gather(sequence_output, 1, mention_p_ids)

        # only extracts first element
        pooled_p = self.pooler(p_output)

        # evidence-aware modeling
        # input_ids = input_ids[~gpr_tags_mask].view(batch_size, -1)
        # token_type_ids = token_type_ids[~gpr_tags_mask].view(batch_size, -1)
        # attention_mask = attention_mask[~gpr_tags_mask].view(batch_size, -1)
        # input_ids = F.pad(input_ids, pad=(0, 18), mode='constant', value=0)
        # token_type_ids = F.pad(token_type_ids, pad=(0, 18), mode='constant', value=0)
        # attention_mask = F.pad(attention_mask, pad=(0, 18), mode='constant', value=0)
        # sequence_output, pooled_output = self.bert(input_ids, 
        #                                             token_type_ids, 
        #                                             attention_mask, 
        #                                             output_all_encoded_layers=False)

        # get states corresponding to A, B, P
        p_output = torch.gather(sequence_output, 1, mention_p_ids)
        pooled_p_ = self.pooler(p_output).unsqueeze(1)

        mention_a_ids = mention_a_ids.unsqueeze(-1)
        mention_a_ids = mention_a_ids.repeat(1, 1, self.config.hidden_size)
        a_output = torch.gather(sequence_output, 1, mention_a_ids)
        a_output = self.pooler(a_output).unsqueeze(1)

        mention_b_ids = mention_b_ids.unsqueeze(-1)
        mention_b_ids = mention_b_ids.repeat(1, 1, self.config.hidden_size)
        b_output = torch.gather(sequence_output, 1, mention_b_ids)
        b_output = self.pooler(b_output).unsqueeze(1)

        y_p, m_attn_p, c_attn_p, co_attn_p = self.evidence_pooler_p(sequence_output, 
                                                                    cluster_ids_p,
                                                                    cluster_mask_p,
                                                                    query_a=a_output, 
                                                                    mask_a=mention_a_mask,
                                                                    query_b=b_output, 
                                                                    mask_b=mention_b_mask,
                                                                    query_p=pooled_p_,
                                                                    training=training)

        output = torch.cat((pooled_p, y_p), dim=1)

        pooled_output = self.dropout(output)
        logits = self.classifier(pooled_output)

        probabilities = F.softmax(logits, dim=1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()#smooth_eps=0.02, reduction=None)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, probabilities
        elif eval_mode:
            return logits, probabilities, m_attn_p, c_attn_p, co_attn_p
        else:
            return logits, probabilities