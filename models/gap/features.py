import csv
import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def convert_examples_to_features(examples, 
                                    tokenizer,
                                    max_seq_length,
                                    n_coref_models,
                                    max_gpr_mention_len=20,
                                    pad_value=0,
                                    verbose=0):

    logger.setLevel(logging.DEBUG)
    if verbose == 0:
        logger.setLevel(logging.WARNING)

    features = []
    for ex_index, example in tqdm(examples.iterrows(), 
                                    desc='Convert Examples to features', 
                                    disable=False):

        tokens = tokenizer.tokenize(example.text)

        if get_sanitized_seq_len(tokens)[0] > max_seq_length - 2:
            logger.info('Need to adjust the sequence: {}, {}'.format(ex_index, ''.join(tokens)))

            tokens = _truncate_seq(tokens, max_seq_length - 2)

        tokens_ = ["[CLS]"] + tokens + ["[SEP]"]

        # first set with gpr tags
        tokens, _, _, _ = extract_cluster_ids(ex_index, 
                                                tokens_.copy(), 
                                                n_coref_models, 
                                                max_mention_len=8,
                                                remove_gpr_tags=False)

        # second without gpr tags only to be used for coref clusters embeddings
        _, cluster_ids_a, cluster_ids_b, cluster_ids_p = extract_cluster_ids(ex_index, 
                                                                            tokens_.copy(), 
                                                                            n_coref_models, 
                                                                            max_mention_len=8,
                                                                            remove_gpr_tags=True)
        
        # mention_ids = A, B and P entity token indices
        # gpr_tag_ids = <A>, <B>, <P> tag token indices
        # The mask has 1 for real tokens and 0 for padding tokens. 
        mention_p_ids, mention_a_ids, mention_b_ids, gpr_tag_ids = get_gpr_mention_ids(tokens, 
                                                                                        max_gpr_mention_len,
                                                                                        ignore_gpr_tags=True)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)
        gpr_tags_mask = np.zeros(len(tokens))
        gpr_tags_mask[gpr_tag_ids] = 1
        gpr_tags_mask = gpr_tags_mask.tolist()
        mention_p_mask = [1] * len(mention_p_ids)
        mention_a_mask = [1] * len(mention_a_ids)
        mention_b_mask = [1] * len(mention_b_ids)

        # Zero-pad up to the max sequence length.
        padding = [pad_value] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        gpr_tags_mask += padding
        mention_p_ids += [pad_value] * (max_gpr_mention_len - len(mention_p_ids))
        mention_a_ids += [pad_value] * (max_gpr_mention_len - len(mention_a_ids))
        mention_b_ids += [pad_value] * (max_gpr_mention_len - len(mention_b_ids))
        mention_p_mask += [pad_value] * (max_gpr_mention_len - len(mention_p_mask))
        mention_a_mask += [pad_value] * (max_gpr_mention_len - len(mention_a_mask))
        mention_b_mask += [pad_value] * (max_gpr_mention_len - len(mention_b_mask))

        # Zero pad coref clusters
        cluster_ids_a, cluster_mask_a = pad_cluster_ids(cluster_ids_a, n_coref_models, 
                                                        max_seq_length,
                                                        max_mention_len=8,
                                                        max_coref_mentions=20,
                                                        pad_value=pad_value)
        cluster_ids_b, cluster_mask_b = pad_cluster_ids(cluster_ids_b, n_coref_models,
                                                        max_seq_length,
                                                        max_mention_len=8,
                                                        max_coref_mentions=20,
                                                        pad_value=pad_value)
        cluster_ids_p, cluster_mask_p = pad_cluster_ids(cluster_ids_p, n_coref_models, 
                                                        max_seq_length,
                                                        max_mention_len=8,
                                                        max_coref_mentions=20,
                                                        pad_value=pad_value)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 1:
            logger.debug("*** Example ***")
            logger.debug("id: {}".format(example.id))
            logger.debug("tokens: {}".format(" ".join([str(x) for x in tokens])))
            # logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.debug("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.debug("label: {}".format(example.label))
            logger.debug("pretrained: {}".format(example.pretrained))

            tokens_ = np.array(tokens)
            logger.debug("GPR tags mask: {}".format(tokens_[np.array(gpr_tags_mask).astype(bool)[:len(tokens_)]]))

            tokens_ = np.array(tokens)[~np.array(gpr_tags_mask).astype(bool)[:len(tokens)]]
            logger.debug("Pronoun tokens: {}".format(tokens_[mention_p_ids][np.array(mention_p_mask).astype(bool)]))
            logger.debug("A tokens: {}".format(tokens_[mention_a_ids][np.array(mention_a_mask).astype(bool)]))
            logger.debug("B tokens: {}".format(tokens_[mention_b_ids][np.array(mention_b_mask).astype(bool)]))

            for model_idx in range(n_coref_models):
                logger.debug("clusters P model {}: {}".format(model_idx, 
                                                                [tokens_[mention][np.array(cluster_mask_p[model_idx][i]).astype(bool)].tolist()
                                                                    for i, mention in enumerate(cluster_ids_p[model_idx])]))

                logger.debug("clusters A model {}: {}".format(model_idx, 
                                                                [tokens_[mention][np.array(cluster_mask_a).astype(bool)[model_idx][i]].tolist()
                                                                    for i, mention in enumerate(cluster_ids_a[model_idx])]))

                logger.debug("clusters B model {}: {}".format(model_idx, 
                                                                [tokens_[mention][np.array(cluster_mask_b).astype(bool)[model_idx][i]].tolist()
                                                                    for i, mention in enumerate(cluster_ids_b[model_idx])]))

        assert len(tokens) <= max_seq_length, '{}\n{}\n{}'.format(ex_index, len(tokens), tokens)
        assert ''.join(tokens).upper().count('<P>') == 2 and \
                    ''.join(tokens).upper().count('<A>') == 2 and \
                    ''.join(tokens).upper().count('<B>') == 2, (ex_index,
                                                                "".join(tokens).upper().count('<P>'), 
                                                                "".join(tokens).upper().count('<A>'),
                                                                "".join(tokens).upper().count('<B>'), 
                                                                "".join(tokens))

        features.append(
                AttrDict({'input_ids': input_ids,
                              'input_mask': input_mask,
                              'segment_ids': segment_ids,
                              'gpr_tags_mask': gpr_tags_mask,
                              'mention_p_ids': mention_p_ids,
                              'mention_a_ids': mention_a_ids,
                              'mention_b_ids': mention_b_ids,
                              'mention_p_mask': mention_p_mask,
                              'mention_a_mask': mention_a_mask,
                              'mention_b_mask': mention_b_mask,
                              'cluster_ids_a': cluster_ids_a,
                              'cluster_mask_a': cluster_mask_a,
                              'cluster_ids_b': cluster_ids_b,
                              'cluster_mask_b': cluster_mask_b,
                              'cluster_ids_p': cluster_ids_p,
                              'cluster_mask_p': cluster_mask_p,
                              'label_id': example.label,
                              'pretrained': example.pretrained}))
    return features

def get_gpr_mention_ids(tokens, max_gpr_mention_len, ignore_gpr_tags=False):
    gpr_ids = {'<P>': [], '<A>': [], '<B>': []}
    gpr_tag_ids = []
    entity = None
    for i, token_ in enumerate(tokens):
        token = ''.join(tokens[i:i+3]).upper()

        if token in ['<P>', '<A>', '<B>']:
            gpr_tag_ids += [i, i+1, i+2]

        if entity is not None and token not in ['<P>', '<A>', '<B>']:
            if ignore_gpr_tags:
                gpr_ids[entity].append(i+2-len(gpr_tag_ids))
            else:
                gpr_ids[entity].append(i+2)

        if token in ['<P>', '<A>', '<B>']:
            if entity == token:
                entity = None
            else:
                entity = token

    return (gpr_ids['<P>'][:-2][:max_gpr_mention_len], 
            gpr_ids['<A>'][:-2][:max_gpr_mention_len], 
            gpr_ids['<B>'][:-2][:max_gpr_mention_len], 
            gpr_tag_ids)

def pad_cluster_ids(cluster_ids, n_coref_models, max_seq_length, 
                    max_mention_len=4, 
                    max_coref_mentions=5,
                    pad_value=0):
    # pad cluster ids
    cluster_mask = [[] for i in range(n_coref_models)]

    for model_idx in range(n_coref_models):
        # limit to 10 mentions max for now
        # pad mentions length
        model_cluster_ids = cluster_ids[model_idx][:max_coref_mentions]
        for i, mention in enumerate(model_cluster_ids):
            cluster_mask[model_idx].append([1] * len(model_cluster_ids[i]) + [0] * (max_mention_len-len(model_cluster_ids[i])))
            model_cluster_ids[i] += [pad_value] * (max_mention_len-len(model_cluster_ids[i]))
        cluster_ids[model_idx] = model_cluster_ids

        # pad cluster lengths
        if len(cluster_ids[model_idx]) < max_coref_mentions:
            cluster_ids[model_idx] += [[pad_value] * max_mention_len] * (max_coref_mentions-len(cluster_ids[model_idx]))
            cluster_mask[model_idx] += [[0] * max_mention_len] * (max_coref_mentions-len(cluster_mask[model_idx]))

    return cluster_ids, cluster_mask


def populate_cluster(cluster_ids, tokens_to_remove, token_ids):
    if len(cluster_ids[-1]) == 0:
        tokens_to_remove += token_ids
        cluster_ids[-1].append(token_ids[-1] + 1 - len(tokens_to_remove))
    else:
        mention_tokens = range(cluster_ids[-1][0], token_ids[0] - len(tokens_to_remove))
        mention_tokens = list(mention_tokens)
        cluster_ids.pop()
        cluster_ids.append(mention_tokens)
        tokens_to_remove += token_ids
        cluster_ids.append([])

    return cluster_ids, tokens_to_remove

def filter_coref_mentions(tokens, cluster_ids, max_mention_len=4):
    mentions = []
    for mention in cluster_ids:
        if len(mention) == 0:
            print(ex_index, cluster_ids, tokens, mention)
        token_ids = []
        start = mention[0]
        while start < mention[-1]+1:
            token = ''.join(tokens[start:start+3]).upper()
            if token in ['<P>', '<A>', '<B>']:
                start += 2
            else:
                token_ids.append(start)

            start += 1

        # Lee at al may tag a span like this
        # <C>the murdered actress Dorothy Stratten, who was dating director Peter Bogdanovich at the time of her<C>
        if len(token_ids) <= max_mention_len:
            mentions.append(token_ids)

    return mentions

def extract_cluster_ids(ex_index, tokens, n_coref_models, max_mention_len=4, remove_gpr_tags=False):
    gpr_tags = ['<P>', '<A>', '<B>']
    if remove_gpr_tags:
        tokens_ = []
        start = 0
        while start < len(tokens):
            if ''.join(tokens[start:start+3]).upper() in gpr_tags:
                start += 3
            else:
                tokens_.append(tokens[start])
                start += 1
        tokens = tokens_

    cluster_tags = ['<C_{}>'.format(i) for i in range(n_coref_models)] + \
                        ['<D_{}>'.format(i) for i in range(n_coref_models)] + \
                        ['<E_{}>'.format(i) for i in range(n_coref_models)]

    # map cluster ids to tokens so that we can make pairs and keep track of token ids for removal
    map_idx_to_token = []
    start = 0
    while start < len(tokens):
        token = ''.join(tokens[start:start+5]).upper()

        if token in cluster_tags:
            mapping = (list(range(start, start+5)), token)
            map_idx_to_token.append(mapping)
        else:
            map_idx_to_token.append(([start], tokens[start]))
        start += 1

    cluster_ids_a = [[[]] for i in range(n_coref_models)]
    cluster_ids_b = [[[]] for i in range(n_coref_models)]
    cluster_ids_p = [[[]] for i in range(n_coref_models)]
    tokens_to_remove = []
    for (token_ids, token) in map_idx_to_token:
        if token in cluster_tags:
            coref_model_idx = int(token[3])
            # if ex_index < 1:
                # logger.info((ex_index, token, coref_model_idx))
            if 'C' in token:
                cluster_ids_a[coref_model_idx], tokens_to_remove = populate_cluster(cluster_ids_a[coref_model_idx], 
                                                                                    tokens_to_remove, 
                                                                                    token_ids)
            if 'D' in token:
                cluster_ids_b[coref_model_idx], tokens_to_remove = populate_cluster(cluster_ids_b[coref_model_idx], 
                                                                                    tokens_to_remove, 
                                                                                    token_ids)
            if 'E' in token:
                cluster_ids_p[coref_model_idx], tokens_to_remove = populate_cluster(cluster_ids_p[coref_model_idx], 
                                                                                    tokens_to_remove, 
                                                                                    token_ids)

    for i in range(n_coref_models):
        # if ex_index < 1:
        #     logger.info((ex_index, i, cluster_ids_a[i]))
        #     logger.info((ex_index, i, cluster_ids_b[i]))
        #     logger.info((ex_index, i, cluster_ids_p[i]))
        cluster_ids_a[i].pop()
        cluster_ids_b[i].pop()
        cluster_ids_p[i].pop()

    # remove coref tags from tokens
    for i, idx in enumerate(tokens_to_remove):
        del tokens[idx-i]

    # gather tokens between cluster tags
    # filter out coref mention that are either a gpr tag or has tokens more than 6
    for i in range(n_coref_models):
        cluster_ids_a[i] = filter_coref_mentions(tokens, cluster_ids_a[i], max_mention_len=max_mention_len)
        cluster_ids_b[i] = filter_coref_mentions(tokens, cluster_ids_b[i], max_mention_len=max_mention_len)
        cluster_ids_p[i] = filter_coref_mentions(tokens, cluster_ids_p[i], max_mention_len=max_mention_len)

    return tokens, cluster_ids_a, cluster_ids_b, cluster_ids_p

def remove_first_matching_tag(tokens, tag):
    start = 1
    while start < len(tokens):
        if ''.join(tokens[start:start+5]) == tag:
            del tokens[start:start+5]
            break
        start += 1

    return tokens

def get_sanitized_seq_len(tokens):
    seq_len = 0
    start = 0
    tokens_ = []
    while start < len(tokens):
        if (''.join(tokens[start:start+3] + tokens[start+4:start+5])).upper() in ['<C_>', '<D_>', '<E_>']:
            start += 5
        else:
            tokens_.append(tokens[start])
            seq_len += 1
            start += 1

    return seq_len, tokens_

def _truncate_seq(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # 1. First truncate the begining
    # 2. truncate the end
    # 3. truncate the middle

    # map gpr tokens - cannot be removed
    # if a token matches c or d, then don't consider it in sequence length
    
    gpr_tags = ['<P>', '<A>', '<B>']
    cluster_tags = ['<C_>', '<D_>', '<E_>']
    # 1 Start truncating from begining
    #   if first token is not in gpr tags then remove it.
    #       if it was a cluster tag, then remove the corresponding matching end tag as well
    while get_sanitized_seq_len(tokens)[0] > max_length:
        _, sanitized_tokens = get_sanitized_seq_len(tokens)

        token = ''.join(sanitized_tokens[0:3]).upper()
        if token not in gpr_tags:
            # while first token is a cluster tag keep removing it and its matching end tag
            while (''.join(tokens[:3] + tokens[4:5])).upper() in ['<C_>', '<D_>', '<E_>']:
                tokens = remove_first_matching_tag(tokens, ''.join(tokens[:5]))
                del tokens[:5]
            del tokens[0]
            continue

        token = ''.join(sanitized_tokens[-3:]).upper()
        if token not in gpr_tags:
            # while last token is a cluster tag keep removing it and its matching start tag
            while (''.join(tokens[-5:-2] + tokens[-1:])).upper() in ['<C_>', '<D_>', '<E_>']:
                tokens_ = tokens[::-1]
                tokens = remove_first_matching_tag(tokens_, ''.join(tokens_[:5]))
                tokens = tokens[::-1]
                del tokens[-5:]
            del tokens[-1]
            continue

        raise Exception('Couldnt find a good way to truncate the sequence.')

        # need a third case to remove from middle
        # if none of the above cases apply then pick a random token from middle
        # ensure that it's not beetween a gpr tag and drop it, or skip word between gpr tags
        tags = 1
        start = 3
        while start < len(tokens):
            token = ''.join(tokens[start:start+3]).upper() 
            token2 = (''.join(tokens[start:start+3] + tokens[start+4:start+5])).upper()
            if token in gpr_tags:
                start += 3
                tag += 1
            elif token2 in cluster_tags:
                start += 5
            elif togs % 2 == 0:
                del tokens[start]
                break
            else:
                start += 1

    return tokens

# class InputExample(object):
#     def __init__(self, guid, text_a, pretrained=None, label=None):
#         self.guid = guid
#         self.text_a = text_a
#         self.pretrained = pretrained
#         self.label = label


# class InputFeatures(object):
#     def __init__(self, 
#                 input_ids, 
#                 input_mask, 
#                 segment_ids, 
#                 label_id,
#                 gpr_tags_mask, 
#                 gpr_ids=None, 
#                 gpr_mask=None, 
#                 gpr_ids2=None, 
#                 gpr_mask2=None, 
#                 cluster_ids_a=None,
#                 cluster_mask_a=None,
#                 cluster_ids_b=None,
#                 cluster_mask_b=None,
#                 cluster_ids_p=None,
#                 cluster_mask_p=None,
#                 pretrained=None):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.gpr_tags_mask = gpr_tags_mask
#         self.label_id = label_id
#         self.gpr_ids = gpr_ids
#         self.gpr_mask = gpr_mask
#         self.gpr_ids2 = gpr_ids2
#         self.gpr_mask2 = gpr_mask2
#         self.pretrained = pretrained
#         self.cluster_ids_a = cluster_ids_a
#         self.cluster_mask_a = cluster_mask_a
#         self.cluster_ids_b = cluster_ids_b
#         self.cluster_mask_b = cluster_mask_b
#         self.cluster_ids_p = cluster_ids_p
#         self.cluster_mask_p = cluster_mask_p

# class DataProcessor(object):
#     """Base class for data converters for sequence classification data sets."""

#     def get_train_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the train set."""
#         raise NotImplementedError()

#     def get_dev_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the dev set."""
#         raise NotImplementedError()

#     def get_labels(self):
#         """Gets the list of labels for this data set."""
#         raise NotImplementedError()

#     @classmethod
#     def _read_tsv(cls, input_file, quotechar=None):
#         """Reads a tab separated value file."""
#         with open(input_file, "r") as f:
#             reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#             lines = []
#             for line in reader:
#                 if sys.version_info[0] == 2:
#                     line = list(unicode(cell, 'utf-8') for cell in line)
#                 lines.append(line)
#             return lines

# class GPRProcessor(DataProcessor):
#   """Processor for the MRPC data set (GLUE version)."""

#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

#   def get_labels(self):
#     """See base class."""
#     return ["0", "1", "2"]

#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     verbose = logger.level == logging.info
#     examples = []
#     for (i, line) in tqdm(enumerate(lines), desc='Creating examples', disable=~verbose):
#         guid = "%s-%s" % (set_type, i)
#         label, text_a, pretrained = line[:3]
#         pretrained = eval(pretrained)
#         examples.append(InputExample(guid=guid, text_a=text_a, pretrained=pretrained, label=label))
#     return examples

# def convert_examples_to_features(examples, 
#                                     label_list, 
#                                     max_seq_length,
#                                     tokenizer, 
#                                     n_coref_models=6,
#                                     verbose=0):
#     logger.setLevel(logging.DEBUG)
#     if verbose == 0:
#         logger.setLevel(logging.WARNING)

#     label_map = {label : i for i, label in enumerate(label_list)}

#     features = []
#     for (ex_index, example) in tqdm(enumerate(examples), desc='Convert Examples to features', disable=False):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))

#         tokens_a = tokenizer.tokenize(example.text_a)

#         is_truncated = False
#         if get_sanitized_seq_len(tokens_a)[0] > max_seq_length - 2:
#             logger.info('Need to adjust the sequence: {}, {}'.format(ex_index, ''.join(tokens_a)))

#             tokens_a = _truncate_seq(tokens_a, max_seq_length - 2)

#             is_truncated = True

#         tokens_ = ["[CLS]"] + tokens_a + ["[SEP]"]

#         tokens, cluster_ids_a, cluster_ids_b, cluster_ids_p = extract_cluster_ids(ex_index, 
#                                                                                     tokens_.copy(), 
#                                                                                     n_coref_models, 
#                                                                                     max_mention_len=8)

#         tokens2, cluster_ids_a, cluster_ids_b, cluster_ids_p = extract_cluster_ids(ex_index, 
#                                                                                     tokens_.copy(), 
#                                                                                     n_coref_models, 
#                                                                                     max_mention_len=8,
#                                                                                     remove_gpr_tags=False)
        
#         gpr_ids = {'<P>': [], '<A>': [], '<B>': []}
#         token_idx = 0
#         tokens_to_remove = []
#         entity = None
#         for i, token_ in enumerate(tokens):
#             token = ''.join(tokens[i:i+3]).upper()

#             if token in ['<P>', '<A>', '<B>']:
#                 tokens_to_remove += [i, i+1, i+2]

#             if entity is not None and token not in ['<P>', '<A>', '<B>']:
#                 # +1 bcoz of [CLS] that will be appended
#                 gpr_ids[entity].append([i+2])

#             if token in ['<P>', '<A>', '<B>']:
#                 if entity == token:
#                     entity = None
#                 else:
#                     entity = token

#         gpr_ids = [gpr_ids['<P>'][:-2], gpr_ids['<A>'][:-2], gpr_ids['<B>'][:-2]]
#         # for i, idx in enumerate(tokens_to_remove):
#         #     del tokens_a[idx-i]

#         gpr_ids2 = {'<P>': [], '<A>': [], '<B>': []}
#         token_idx = 0
#         tokens_to_remove_ = []
#         entity = None
#         for i, token_ in enumerate(tokens):
#             token = ''.join(tokens[i:i+3]).upper()

#             if token in ['<P>', '<A>', '<B>']:
#                 tokens_to_remove_ += [i, i+1, i+2]

#             if entity is not None and token not in ['<P>', '<A>', '<B>']:
#                 gpr_ids2[entity].append([i+2-len(tokens_to_remove_)])

#             if token in ['<P>', '<A>', '<B>']:
#                 if entity == token:
#                     entity = None
#                 else:
#                     entity = token

#         gpr_ids2 = [gpr_ids2['<P>'][:-2], gpr_ids2['<A>'][:-2], gpr_ids2['<B>'][:-2]]
#         gpr_ids2 = gpr_ids

#         segment_ids = [0] * len(tokens)

#         input_ids = tokenizer.convert_tokens_to_ids(tokens)

#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)

#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding

#         gpr_tags_mask = np.ones(len(input_ids))
#         gpr_tags_mask[tokens_to_remove] = 0

#         segment_ids2 = [0] * len(tokens2)
#         input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
#         input_mask2 = [1] * len(input_ids2)
#         padding = [0] * (max_seq_length - len(input_ids2))
#         input_ids2 += padding
#         input_mask2 += padding
#         segment_ids2 += padding

#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length

#         label_id = label_map[example.label]

#         pretrained = example.pretrained

#         if ex_index < 1 or is_truncated:
#             logger.debug("*** Example ***")
#             logger.debug("guid: %s" % (example.guid))
#             logger.debug("tokens: %s" % " ".join(
#                     [str(x) for x in tokens]))
#             logger.debug("tokens2: %s" % " ".join(
#                     [str(x) for x in np.array(tokens)[np.array(gpr_tags_mask[:len(tokens)], dtype=bool)]]))
#             logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.debug(
#                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

#             logger.debug("Pronoun ids: %s (tokens = %s)" % (np.array(gpr_ids[0])[:, 0], 
#                                                         np.array(tokens)[np.array(gpr_ids[0])[:, 0]]))
#             logger.debug("A ids: %s (tokens = %s)" % (np.array(gpr_ids[1])[:, 0], 
#                                                         np.array(tokens)[np.array(gpr_ids[1])[:, 0]]))
#             logger.debug("B ids: %s (tokens = %s)" % (np.array(gpr_ids[2])[:, 0], 
#                                                         np.array(tokens)[np.array(gpr_ids[2])[:, 0]]))

#             logger.debug("Pronoun ids 2: %s (tokens = %s)" % (np.array(gpr_ids2[0])[:, 0], 
#                                                         np.array(tokens2)[np.array(gpr_ids2[0])[:, 0]]))
#             logger.debug("A ids 2: %s (tokens = %s)" % (np.array(gpr_ids2[1])[:, 0], 
#                                                         np.array(tokens2)[np.array(gpr_ids2[1])[:, 0]]))
#             logger.debug("B ids 2: %s (tokens = %s)" % (np.array(gpr_ids2[2])[:, 0], 
#                                                         np.array(tokens2)[np.array(gpr_ids2[2])[:, 0]]))

#             logger.debug("label: %s (id = %d)" % (example.label, label_id))

#             logger.debug("pretrained: %s" % (example.pretrained))

#             logger.debug("tokens_a: %s" % "".join([str(x) for x in tokens_a]))

#             for model_idx in range(n_coref_models):
#                 logger.debug("clusters A model {}: {}".format(model_idx, 
#                                                                 [np.array(tokens)[mention].tolist()
#                                                                     for mention in cluster_ids_a[model_idx]]))
#                 logger.debug("clusters B model {}: {}".format(model_idx,
#                                                                 [np.array(tokens)[mention].tolist()
#                                                                     for mention in cluster_ids_b[model_idx]]))

#                 logger.debug("clusters P model {}: {}".format(model_idx,
#                                                                 [np.array(tokens)[mention].tolist()
#                                                                     for mention in cluster_ids_p[model_idx]]))

#         assert len(tokens) <= max_seq_length, '{}\n{}\n{}'.format(ex_index, len(tokens), tokens)
#         assert len(gpr_ids[0]) > 0 and len(gpr_ids[1]) > 0 and len(gpr_ids[2]) > 0
#         assert ''.join(tokens).upper().count('<P>') == 2 and \
#                     ''.join(tokens).upper().count('<A>') == 2 and \
#                     ''.join(tokens).upper().count('<B>') == 2, (ex_index,
#                                                                 "".join(tokens).upper().count('<P>'), 
#                                                                 "".join(tokens).upper().count('<A>'),
#                                                                 "".join(tokens).upper().count('<B>'), 
#                                                                 "".join(tokens))


#         # pad gpr ids
#         gpr_mask = [[], [], []]
#         pad_value = 0
#         for idx, entity in enumerate(['<P>', '<A>', '<B>']):
#             gpr_mask[idx] = [0] * len(gpr_ids[idx]) + [1] * (20-len(gpr_ids[idx]))
#             gpr_ids[idx] += [[pad_value]]*(20-len(gpr_ids[idx]))

#         gpr_mask2 = [[], [], []]
#         for idx, entity in enumerate(['<P>', '<A>', '<B>']):
#             gpr_mask2[idx] = [0] * len(gpr_ids2[idx]) + [1] * (20-len(gpr_ids2[idx]))
#             gpr_ids2[idx] += [[pad_value]]*(20-len(gpr_ids2[idx]))

#         cluster_ids_a, cluster_mask_a = pad_cluster_ids(cluster_ids_a, n_coref_models, 
#                                                         max_seq_length,
#                                                         max_mention_len=8,
#                                                         max_coref_mentions=20,
#                                                         pad_value=pad_value)
#         cluster_ids_b, cluster_mask_b = pad_cluster_ids(cluster_ids_b, n_coref_models,
#                                                         max_seq_length,
#                                                         max_mention_len=8,
#                                                         max_coref_mentions=20,
#                                                         pad_value=pad_value)
#         cluster_ids_p, cluster_mask_p = pad_cluster_ids(cluster_ids_p, n_coref_models, 
#                                                         max_seq_length,
#                                                         max_mention_len=8,
#                                                         max_coref_mentions=20,
#                                                         pad_value=pad_value)

#         if ex_index < 1 or is_truncated:
#             for model_idx in range(n_coref_models):
#                 logger.debug("clusters A ids, model {}: {}".format(model_idx, cluster_ids_a[model_idx]))
#                 logger.debug("clusters A mask, model {}: {}".format(model_idx, cluster_mask_a[model_idx]))

#                 logger.debug("clusters B ids, model {}: {}".format(model_idx, cluster_ids_b[model_idx]))
#                 logger.debug("clusters B mask, model {}: {}".format(model_idx, cluster_mask_b[model_idx]))

#                 logger.debug("clusters P ids, model {}: {}".format(model_idx, cluster_ids_p[model_idx]))
#                 logger.debug("clusters P mask, model {}: {}".format(model_idx, cluster_mask_p[model_idx]))

#         features.append(
#                 InputFeatures(input_ids=input_ids,
#                               input_mask=input_mask,
#                               segment_ids=segment_ids,
#                               label_id=label_id,
#                               gpr_tags_mask=gpr_tags_mask,
#                               gpr_ids=gpr_ids,
#                               gpr_mask=gpr_mask,
#                               gpr_ids2=gpr_ids2,
#                               gpr_mask2=gpr_mask2,
#                               cluster_ids_a=cluster_ids_a,
#                               cluster_mask_a=cluster_mask_a,
#                               cluster_ids_b=cluster_ids_b,
#                               cluster_mask_b=cluster_mask_b,
#                               cluster_ids_p=cluster_ids_p,
#                               cluster_mask_p=cluster_mask_p,
#                               pretrained=pretrained))
#     return features

# def pad_cluster_ids(cluster_ids, n_coref_models, max_seq_length, 
#                     max_mention_len=4, 
#                     max_coref_mentions=5,
#                     pad_value=0):
#     # pad cluster ids
#     cluster_mask = [[] for i in range(n_coref_models)]

#     for model_idx in range(n_coref_models):
#         # limit to 10 mentions max for now
#         # pad mentions length
#         model_cluster_ids = cluster_ids[model_idx][:max_coref_mentions]
#         for i, mention in enumerate(model_cluster_ids):
#             cluster_mask[model_idx].append([0] * len(model_cluster_ids[i]) + [1] * (max_mention_len-len(model_cluster_ids[i])))
#             model_cluster_ids[i] += [pad_value] * (max_mention_len-len(model_cluster_ids[i]))
#         cluster_ids[model_idx] = model_cluster_ids

#         # pad cluster lengths
#         if len(cluster_ids[model_idx]) < max_coref_mentions:
#             cluster_ids[model_idx] += [[pad_value] * max_mention_len] * (max_coref_mentions-len(cluster_ids[model_idx]))
#             cluster_mask[model_idx] += [[1] * max_mention_len] * (max_coref_mentions-len(cluster_mask[model_idx]))

#     return cluster_ids, cluster_mask


# def populate_cluster(cluster_ids, tokens_to_remove, token_ids):
#     if len(cluster_ids[-1]) == 0:
#         tokens_to_remove += token_ids
#         cluster_ids[-1].append(token_ids[-1] + 1 - len(tokens_to_remove))
#     else:
#         mention_tokens = range(cluster_ids[-1][0], token_ids[0] - len(tokens_to_remove))
#         mention_tokens = list(mention_tokens)
#         cluster_ids.pop()
#         cluster_ids.append(mention_tokens)
#         tokens_to_remove += token_ids
#         cluster_ids.append([])

#     return cluster_ids, tokens_to_remove

# def filter_coref_mentions(tokens, cluster_ids, max_mention_len=4):
#     mentions = []
#     for mention in cluster_ids:
#         if len(mention) == 0:
#             print(ex_index, cluster_ids, tokens, mention)
#         token_ids = []
#         start = mention[0]
#         while start < mention[-1]+1:
#             token = ''.join(tokens[start:start+3]).upper()
#             if token in ['<P>', '<A>', '<B>']:
#                 start += 2
#             else:
#                 token_ids.append(start)

#             start += 1

#         # Lee at al may tag a span like this
#         # <C>the murdered actress Dorothy Stratten, who was dating director Peter Bogdanovich at the time of her<C>
#         if len(token_ids) <= max_mention_len:
#             mentions.append(token_ids)

#     return mentions

# def extract_cluster_ids(ex_index, tokens, n_coref_models, max_mention_len=4, remove_gpr_tags=False):
#     gpr_tags = ['<P>', '<A>', '<B>']
#     if remove_gpr_tags:
#         tokens_ = []
#         start = 0
#         while start < len(tokens):
#             if ''.join(tokens[start:start+3]).upper() in gpr_tags:
#                 start += 3
#             else:
#                 tokens_.append(tokens[start])
#                 start += 1
#         tokens = tokens_

#     cluster_tags = ['<C_{}>'.format(i) for i in range(n_coref_models)] + \
#                         ['<D_{}>'.format(i) for i in range(n_coref_models)] + \
#                         ['<E_{}>'.format(i) for i in range(n_coref_models)]

#     # map cluster ids to tokens so that we can make pairs and keep track of token ids for removal
#     map_idx_to_token = []
#     start = 0
#     while start < len(tokens):
#         token = ''.join(tokens[start:start+5]).upper()

#         if token in cluster_tags:
#             mapping = (list(range(start, start+5)), token)
#             map_idx_to_token.append(mapping)
#         else:
#             map_idx_to_token.append(([start], tokens[start]))
#         start += 1

#     cluster_ids_a = [[[]] for i in range(n_coref_models)]
#     cluster_ids_b = [[[]] for i in range(n_coref_models)]
#     cluster_ids_p = [[[]] for i in range(n_coref_models)]
#     tokens_to_remove = []
#     for (token_ids, token) in map_idx_to_token:
#         if token in cluster_tags:
#             coref_model_idx = int(token[3])
#             # if ex_index < 1:
#                 # logger.info((ex_index, token, coref_model_idx))
#             if 'C' in token:
#                 cluster_ids_a[coref_model_idx], tokens_to_remove = populate_cluster(cluster_ids_a[coref_model_idx], 
#                                                                                     tokens_to_remove, 
#                                                                                     token_ids)
#             if 'D' in token:
#                 cluster_ids_b[coref_model_idx], tokens_to_remove = populate_cluster(cluster_ids_b[coref_model_idx], 
#                                                                                     tokens_to_remove, 
#                                                                                     token_ids)
#             if 'E' in token:
#                 cluster_ids_p[coref_model_idx], tokens_to_remove = populate_cluster(cluster_ids_p[coref_model_idx], 
#                                                                                     tokens_to_remove, 
#                                                                                     token_ids)

#     for i in range(n_coref_models):
#         # if ex_index < 1:
#         #     logger.info((ex_index, i, cluster_ids_a[i]))
#         #     logger.info((ex_index, i, cluster_ids_b[i]))
#         #     logger.info((ex_index, i, cluster_ids_p[i]))
#         cluster_ids_a[i].pop()
#         cluster_ids_b[i].pop()
#         cluster_ids_p[i].pop()

#     # remove coref tags from tokens
#     for i, idx in enumerate(tokens_to_remove):
#         del tokens[idx-i]

#     # gather tokens between cluster tags
#     # filter out coref mention that are either a gpr tag or has tokens more than 6
#     for i in range(n_coref_models):
#         cluster_ids_a[i] = filter_coref_mentions(tokens, cluster_ids_a[i], max_mention_len=max_mention_len)
#         cluster_ids_b[i] = filter_coref_mentions(tokens, cluster_ids_b[i], max_mention_len=max_mention_len)
#         cluster_ids_p[i] = filter_coref_mentions(tokens, cluster_ids_p[i], max_mention_len=max_mention_len)

#     return tokens, cluster_ids_a, cluster_ids_b, cluster_ids_p


# def get_sanitized_seq_len(tokens):
#     seq_len = 0
#     start = 0
#     tokens_ = []
#     while start < len(tokens):
#         if (''.join(tokens[start:start+3] + tokens[start+4:start+5])).upper() in ['<C_>', '<D_>', '<E_>']:
#             start += 5
#         else:
#             tokens_.append(tokens[start])
#             seq_len += 1
#             start += 1

#     return seq_len, tokens_

# def remove_first_matching_tag(tokens, tag):
#     start = 1
#     while start < len(tokens):
#         if ''.join(tokens[start:start+5]) == tag:
#             del tokens[start:start+5]
#             break
#         start += 1

#     return tokens

# def _truncate_seq(tokens, max_length):
#     """Truncates a sequence pair in place to the maximum length."""

#     # 1. First truncate the begining
#     # 2. truncate the end
#     # 3. truncate the middle

#     # map gpr tokens - cannot be removed
#     # if a token matches c or d, then don't consider it in sequence length
    
#     gpr_tags = ['<P>', '<A>', '<B>']
#     cluster_tags = ['<C_>', '<D_>', '<E_>']
#     # 1 Start truncating from begining
#     #   if first token is not in gpr tags then remove it.
#     #       if it was a cluster tag, then remove the corresponding matching end tag as well
#     while get_sanitized_seq_len(tokens)[0] > max_length:
#         _, sanitized_tokens = get_sanitized_seq_len(tokens)

#         token = ''.join(sanitized_tokens[0:3]).upper()
#         if token not in gpr_tags:
#             # while first token is a cluster tag keep removing it and its matching end tag
#             while (''.join(tokens[:3] + tokens[4:5])).upper() in ['<C_>', '<D_>', '<E_>']:
#                 tokens = remove_first_matching_tag(tokens, ''.join(tokens[:5]))
#                 del tokens[:5]
#             del tokens[0]
#             continue

#         token = ''.join(sanitized_tokens[-3:]).upper()
#         if token not in gpr_tags:
#             # while last token is a cluster tag keep removing it and its matching start tag
#             while (''.join(tokens[-5:-2] + tokens[-1:])).upper() in ['<C_>', '<D_>', '<E_>']:
#                 tokens_ = tokens[::-1]
#                 tokens = remove_first_matching_tag(tokens_, ''.join(tokens_[:5]))
#                 tokens = tokens[::-1]
#                 del tokens[-5:]
#             del tokens[-1]
#             continue

#         raise Exception('Couldnt find a good way to truncate the sequence.')

#         # need a third case to remove from middle
#         # if none of the above cases apply then pick a random token from middle
#         # ensure that it's not beetween a gpr tag and drop it, or skip word between gpr tags
#         tags = 1
#         start = 3
#         while start < len(tokens):
#             token = ''.join(tokens[start:start+3]).upper() 
#             token2 = (''.join(tokens[start:start+3] + tokens[start+4:start+5])).upper()
#             if token in gpr_tags:
#                 start += 3
#                 tag += 1
#             elif token2 in cluster_tags:
#                 start += 5
#             elif togs % 2 == 0:
#                 del tokens[start]
#                 break
#             else:
#                 start += 1

#     return tokens
