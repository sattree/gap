import re
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas(desc='Annotating gpr mentions in text...')

class MentionsAnnotator():
    def transform(self, X, *args, **kwargs):
          X = X.copy()
          X['text'] = X.progress_apply(self.annotate_mentions, axis=1)

          return {'X': X}

    def annotate_mentions(self, ex):
        ex.a_offset = int(ex.a_offset)
        ex.b_offset = int(ex.b_offset)
        ex.pronoun_offset = int(ex.pronoun_offset)

        assert ex.a_offset < ex.b_offset, ex

        text = ex.text
        text = '{}<A> {}'.format(text[:ex.a_offset], text[ex.a_offset:])
        text = '{}<B> {}'.format(text[:ex.b_offset+4], text[ex.b_offset+4:])
        
        offset = ex.pronoun_offset
        if ex.pronoun_offset > ex.a_offset:
            offset += 4
        if ex.pronoun_offset > ex.b_offset:
            offset += 4
            
        text = '{}<P> {}'.format(text[:offset], text[offset:])

        ex.a_offset = text.index('<A> ') + 4
        ex.b_offset = text.index('<B> ') + 4
        ex.pronoun_offset = text.index('<P> ') + 4

        offset = 5*len(re.findall('<(C|D|E)_.>', re.search(''.join([re.escape(c)+'(<(C|D|E)_.>)*?' for c in ex.a]), text[ex.a_offset:])[0]))
        text = '{} <A>{}'.format(text[:ex.a_offset+len(ex.a)+offset], text[ex.a_offset+len(ex.a)+offset:])
        
        offset = 5*len(re.findall('<(C|D|E)_.>', re.search(''.join([re.escape(c)+'(<(C|D|E)_.>)*?' for c in ex.b]), text[ex.b_offset:])[0])) + 4
        text = '{} <B>{}'.format(text[:ex.b_offset+len(ex.b)+offset], text[ex.b_offset+len(ex.b)+offset:])
        
        offset = 0
        if ex.pronoun_offset > ex.a_offset:
            offset += 4
        if ex.pronoun_offset > ex.b_offset:
            offset += 4

        offset += 5*len(re.findall('<(C|D|E)_.>', text[ex.pronoun_offset:ex.pronoun_offset+len(ex.pronoun)]))
        text = '{} <P>{}'.format(text[:ex.pronoun_offset+len(ex.pronoun)+offset], 
                                    text[ex.pronoun_offset+len(ex.pronoun)+offset:])

        ex.text = text
        
        return text