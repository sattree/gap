import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertLayerNorm

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)

def masked_softmax(matrix, mask=None, q_mask=None, dim=-1):
    NEG_INF = -1e-6
    TINY_FLOAT = 1e-6

    if q_mask is not None:
        mask = (~mask.byte()).float().unsqueeze(-1)
        q_mask = (~q_mask.byte()).float().unsqueeze(-1).transpose(1, 2).contiguous()
        mask = ~torch.bmm(mask, q_mask).byte()

    if mask is None:
        result = F.softmax(matrix, dim=dim)
    else:
        matrix = matrix.masked_fill(mask.byte(), -float('inf'))
        result = F.softmax(matrix, dim=dim)
        result = result.masked_fill(mask.byte(), 0.0)

    return result

class TransformerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, keys, values, attention_mask):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(keys)
        mixed_value_layer = self.value(values)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (attention_mask.float()) * -10000.0
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TransformerSelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, query, keys, values, attention_mask):
        self_output = self.self(query, keys, values, attention_mask)
        attention_output = self.output(self_output, keys)
        return attention_output

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.FFN = nn.Sequential(
                # nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )

        self.FFN2 = nn.Sequential(
                # nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(1*config.hidden_size, config.hidden_size),
                nn.Tanh()
            )

        self.FFN.apply(init_weights)
        self.FFN2.apply(init_weights)

        self._self_attentive_pooling_projection = self.FFN # nn.Linear(config.hidden_size, 1)
        self._integrator_dropout = nn.Dropout(0.1)
        self._output_layer = self.FFN2 # nn.Linear(1*config.hidden_size, config.hidden_size)

    def forward(self, X, mask=None, training=False):
        a_s = self.FFN(X).squeeze(-1)
        a_s = masked_softmax(a_s, mask) 
        G_s = torch.sum(a_s.unsqueeze(-1) * X, dim=1)

        return G_s, a_s

    def replace_masked_values(self, tensor, mask, replace_with):
        if tensor.dim() != mask.dim():
            raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
        #TODO: mask should have 1 at valid postions
        # then mask fill is 1 - mask
        return tensor.masked_fill(mask.byte(), replace_with)

    def masked_softmax(self,
                    vector,
                   mask,
                   dim = -1,
                   memory_efficient = False,
                   mask_fill_value = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
        return result

    def weighted_sum(self, matrix, attention):
        if attention.dim() == 2 and matrix.dim() == 3:
            return attention.unsqueeze(1).bmm(matrix).squeeze(1)
        if attention.dim() == 3 and matrix.dim() == 3:
            return attention.bmm(matrix)
        if matrix.dim() - 1 < attention.dim():
            expanded_size = list(matrix.size())
            for i in range(attention.dim() - matrix.dim() + 1):
                matrix = matrix.unsqueeze(1)
                expanded_size.insert(i + 1, attention.size(i + 1))
            matrix = matrix.expand(*expanded_size)
        intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
        return intermediate.sum(dim=-2)

    def forward(self, X, mask=None, training=False):
        # Simple Pooling layers
        max_masked = self.replace_masked_values(X, mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked, 1)[0]
        min_masked = self.replace_masked_values(X, mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked, 1)[0]
        mean_pool = torch.sum(X, 1) / torch.sum((1-mask).float(), 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        # X = X.permute(0, 2, 1)   # convert to [batch, channels, time]
        # X = F.dropout2d(X, 0.5, training=training)
        # X = X.permute(0, 2, 1)   # back to [batch, time, channels]
        self_attentive_logits = self._self_attentive_pooling_projection(X).squeeze(2)
        self_weights = self.masked_softmax(self_attentive_logits, 1-mask)
        self_attentive_pool = self.weighted_sum(X, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(self_attentive_pool)

        outputs = self._output_layer(pooled_representations_dropped)

        return outputs, self_weights

class EvidencePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.selfattn_mention_level = SelfAttention(config)
        self.selfattn_cluster_level = SelfAttention(config)
        self.selfattn_coref_models_level = SelfAttention(config)

        # should pronoun attend differently to and b?
        self.adaptive_attn = TransformerMultiHeadAttention(config)

        self.y_fine = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

        self.y_fine.apply(init_weights)

        self.compatability = nn.Sequential(
                nn.Linear(1*config.hidden_size, config.hidden_size),
                nn.Tanh()
            )

        self.compatability.apply(init_weights)

    def forward(self, 
                sequence_output, 
                cluster_ids, 
                cluster_mask, 
                query_a, 
                mask_a=None,
                query_b=None, 
                mask_b=None,
                query_p=None, 
                training=False):

        batch_size, n_coref_models, max_coref_mentions, max_mentions_len = cluster_ids.size()

        cluster_mentions, mention_mask, mention_attn_wts = self.reduce_mention_tokens(sequence_output, 
                                                                query_p,
                                                                cluster_ids, 
                                                                cluster_mask,
                                                                batch_size,
                                                                n_coref_models,
                                                                max_coref_mentions,
                                                                max_mentions_len,
                                                                training=training)

        output_ = query_a.repeat(n_coref_models, 1, 1)
        cluster_mentions = self.adaptive_attn(output_,
                                                cluster_mentions,
                                                cluster_mentions,
                                                mention_mask)

        if query_b is not None:
            output_ = query_b.repeat(n_coref_models, 1, 1)
            cluster_mentions = self.adaptive_attn(output_,
                                                cluster_mentions,
                                                cluster_mentions,
                                                mention_mask)

        
        U_m, coref_mentions_attn_wts = self.selfattn_cluster_level(cluster_mentions, mention_mask, training=training)

        # reshape and reduce the coref models
        cluster_mask = cluster_mask.view(batch_size, n_coref_models, -1)
        cluster_mask = cluster_mask.sum(-1) == max_coref_mentions*max_mentions_len
        U_m = U_m.view(batch_size, n_coref_models, -1)

        G_m, coref_models_attn_wts = self.selfattn_coref_models_level(U_m, cluster_mask, training=training)


        y_fine = self.y_fine(G_m)

        return (y_fine, mention_attn_wts.detach().cpu().numpy().reshape(batch_size, -1), 
                        coref_mentions_attn_wts.detach().cpu().numpy().reshape(batch_size, -1), 
                        coref_models_attn_wts.detach().cpu().numpy().reshape(batch_size, -1))

    def reduce_mention_tokens(self,
                                sequence_output, 
                                query_p,
                                token_ids, 
                                token_mask,
                                batch_size,
                                n_coref_models,
                                max_coref_mentions,
                                max_mentions_len,
                                training=False):
        # cluster predictions for the given mention from first coref model
        # flatten batch, coref models and coreferent mentions dimension
        token_mask = token_mask.contiguous().view(-1, max_mentions_len)

        # gather all coreferent mentions in the batch
        token_ids = token_ids.unsqueeze(dim=-1)
        token_ids = token_ids.repeat(1, 1, 1, 1, self.config.hidden_size)
        # for a sample, flatten coref models, coref mentions and mention tokens
        token_ids = token_ids.view(batch_size, -1, self.config.hidden_size)
        cluster_mentions = torch.gather(sequence_output, 1, token_ids)
        # now get mention tokens as sequence and flatten coref models and coref mentions
        cluster_mentions = cluster_mentions.view(-1, max_mentions_len, self.config.hidden_size)

        # reduce mention sequences by attention pooling
        cluster_mentions = self.compatability(cluster_mentions)
        query_p_ = query_p.repeat(n_coref_models*max_coref_mentions, 1, 1)
        cluster_mentions = cluster_mentions * query_p_

        cluster_mentions, mention_attn_wts = self.selfattn_mention_level(cluster_mentions, token_mask, training=training)
        # coref models at batch dim, coref mentions, at time dim
        # reshape the mask for mentions as sequence and reduce the mention sequence dimension
        mention_mask = token_mask.view(batch_size*n_coref_models, -1, max_mentions_len)
        mention_mask = mention_mask.sum(-1) == max_mentions_len
        cluster_mentions = cluster_mentions.view(batch_size*n_coref_models, -1, self.config.hidden_size)

        return cluster_mentions, mention_mask, mention_attn_wts