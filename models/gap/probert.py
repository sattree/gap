import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertConfig, BertModel, BertPooler, BertPreTrainedModel

class ProBERT(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        self.pooler = BertPooler(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(1*config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, 
                input_ids, 
                token_type_ids=None, 
                attention_mask=None, 
                gpr_tags_mask=None,
                mention_p_ids=None, 
                labels=None,
                eval_mode=False,
                **kwargs):
        sequence_output, pooled_output = self.bert(input_ids, 
                                                    token_type_ids, 
                                                    attention_mask, 
                                                    output_all_encoded_layers=False)
        
        batch_size = sequence_output.size()[0]
        sequence_output = sequence_output[~gpr_tags_mask].view(batch_size, -1, self.config.hidden_size)
        mention_p_ids = mention_p_ids.unsqueeze(-1)
        mention_p_ids = mention_p_ids.repeat(1, 1, self.config.hidden_size)

        p_output = torch.gather(sequence_output, 1, mention_p_ids)

        pooled_p = self.pooler(p_output)

        pooled_output = self.dropout(pooled_p)
        logits = self.classifier(pooled_output)

        probabilities = F.softmax(logits, dim=1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, probabilities
        elif eval_mode:
            return logits, probabilities, [], [], []
        else:
            return logits, probabilities