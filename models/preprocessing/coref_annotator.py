import pandas as pd
import numpy as np

from tqdm import tqdm

class CorefAnnotator():
    def __init__(self, models):
        self.models = models

    def transform(self, X, **kwargs):
        X = X.copy()
        # only run through the coref models that have been populated
        kwargs = {k:v for k,v in kwargs.items() if v is not None}

        for idx, x in tqdm(X.iterrows(), total=len(X)):
            tag_at_char_index = []
            for model_idx, model_name in enumerate(self.models):
                data = kwargs[model_name]
                # mention starts and ends are mention detection
                tokens, predicted_clusters, pronoun_offset, a_span, b_span, token_to_char_mapping = data[idx]

                # so far we assumed tokens in a cluster follow in a sequential order
                ### The overlaps can be handled by keeping track of offset with respect to token index
                # they could overlap and 
                #tokens between clusters will definitely not be sequential
                entity_cluster = []
                for cluster in predicted_clusters:
                    for mention in cluster:
                        # If there is an overlap between entity mention and cluster mention
                        if a_span[0] <= mention[1] and a_span[1] >= mention[0]:
                            entity_cluster = list(cluster)
                            #### MULTIPLE CLUSTERS COREFERENT WITH A?????
                            break
                    if len(entity_cluster):
                        break

                ### IT SEEMS ENTITY CLUSTER CAN ITSELF HAVE SUB MENTIONS AS MENTION
                ### TODO: Explicitly place start and end tags and mention index
                ## this implicitly assumes the clusters are always sorted by index
                entity_cluster = sorted(entity_cluster, key=lambda x: x[0])
                cluster = entity_cluster[:1]
                for i, mention in enumerate(entity_cluster[1:]):
                    if cluster[-1][1] >= mention[0]:
                        continue
                    else:
                        cluster.append(mention)

                tag_at_char_index += [(token_to_char_mapping[token]+i*len(tokens[token]), 
                                          '<C_{}>'.format(model_idx)) for mention in cluster
                                                                        for i, token in enumerate(mention)]

                entity_cluster = []
                for cluster in predicted_clusters:
                    for mention in cluster:
                        # If there is an overlap between entity mention and cluster mention
                        if b_span[0] <= mention[1] and b_span[1] >= mention[0]:
                            entity_cluster = cluster
                            #### MULTIPLE CLUSTERS COREFERENT WITH A?????
                            break
                    if len(entity_cluster):
                        break

                ### IT SEEMS ENTITY CLUSTER CAN ITSELF HAVE SUB MENTIONS AS MENTION
                ### THE current logic will take the longest span containing - should we instead focus on mentions??
                entity_cluster = sorted(entity_cluster, key=lambda x: x[0])
                cluster = list(entity_cluster[:1])
                for i, mention in enumerate(entity_cluster[1:]):
                    if cluster[-1][1] >= mention[0]:
                        continue
                    else:
                        cluster.append(mention)

                tag_at_char_index += [(token_to_char_mapping[token]+i*len(tokens[token]), 
                                          '<D_{}>'.format(model_idx)) for mention in cluster
                                                                          for i, token in enumerate(mention)]

                entity_cluster = []
                for cluster in predicted_clusters:
                    for mention in cluster:
                        # If there is an overlap between entity mention and cluster mention
                        if pronoun_offset <= mention[1] and pronoun_offset >= mention[0]:
                            entity_cluster = cluster
                            #### MULTIPLE CLUSTERS COREFERENT WITH A?????
                            break
                    if len(entity_cluster):
                        break

                ### IT SEEMS ENTITY CLUSTER CAN ITSELF HAVE SUB MENTIONS AS MENTION
                ### THE current logic will take the longest span containing - should we instead focus on mentions??
                entity_cluster = sorted(entity_cluster, key=lambda x: x[0])
                cluster = list(entity_cluster[:1])
                for i, mention in enumerate(entity_cluster[1:]):
                    if cluster[-1][1] >= mention[0]:
                        continue
                    else:
                        cluster.append(mention)

                tag_at_char_index += [(token_to_char_mapping[token]+i*len(tokens[token]), 
                                          '<E_{}>'.format(model_idx)) for mention in cluster
                                                                          for i, token in enumerate(mention)]
                            

                # sort the mentions according to start token index
                tag_at_char_index = sorted(tag_at_char_index, key=lambda x: x[0])

            offset, text, pronoun_offset, a_offset, b_offset = self.embed(x,
                                                                  tag_at_char_index, 
                                                                )

            X.loc[idx, ['text', 'pronoun_offset', 'a_offset', 'b_offset']] = [text, pronoun_offset, a_offset, b_offset]

        return {'X': X}

    def embed(self, 
              x,
              tag_at_char_index,
              ):

        a_offset = x.a_offset
        b_offset = x.b_offset
        pronoun_offset = x.pronoun_offset
        text = x.text

        offset = 0

        for mention in tag_at_char_index:
            tag = mention[1]
            tag_len = len(tag)

            mention_start_char_idx = mention[0]
            text = text[:mention_start_char_idx+offset] + tag + text[mention_start_char_idx+offset:]

            if a_offset >= mention_start_char_idx+offset:
                a_offset += tag_len
            if b_offset >= mention_start_char_idx+offset:
                b_offset += tag_len
            if pronoun_offset >= mention_start_char_idx+offset:
                pronoun_offset += tag_len

            offset += tag_len

        return offset, text, pronoun_offset, a_offset, b_offset