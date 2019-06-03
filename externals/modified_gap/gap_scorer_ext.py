from .gap_scorer import Scores, make_scorecard, Annotation
from collections import defaultdict

from .constants import Gender
from .constants import GOLD_FIELDNAMES
from .constants import PRONOUNS
from .constants import SYSTEM_FIELDNAMES

import pandas as pd

def multiclass_to_gap_score(preds, gold_df, score_df, index=None, verbose=1):
    if score_df is None:
        score_df = pd.DataFrame(columns=['M', 'F', 'B', 'O'])
    
    preds_df = pd.DataFrame(columns=['id', 'a_coref', 'b_coref'])
    preds_df['a_coref'] = preds == 0
    preds_df['b_coref'] = preds == 1
    preds_df['id'] = gold_df['id']

    gold_annotations = read_annotations(gold_df, is_gold=True)
    system_annotations = read_annotations(preds_df, is_gold=False)
    scores = calculate_scores(gold_annotations, system_annotations)
    
    display_names = [(None, 'Overall'), (Gender.MASCULINE, 'Masculine'),
                   (Gender.FEMININE, 'Feminine')]

    row = {}
    for gender, display_name in display_names:
        gender_scores = scores.get(gender, Scores())
        f1 = gender_scores.f1()
        row[display_name] = [f1]
    score = pd.DataFrame.from_dict(row)
    score.columns = [col[0] for col in score.columns]
    score['B'] = score['F']/score['M']
    score = score[['M', 'F', 'B', 'O']]
    if index is not None:
        score.index = [index]
    
    score_df = pd.concat([score_df, score])
    if verbose:
      display(score_df.round(2))
      print(make_scorecard(scores))

    return score_df

def add_to_score_view(preds, gold, score_df, index=None, verbose=1):
    if score_df is None:
        score_df = pd.DataFrame(columns=['M', 'F', 'B', 'O'])
    
    preds = pd.DataFrame(preds, columns=['a_coref', 'b_coref'])
    preds['id'] = gold['id']
    gold_annotations = read_annotations(gold, is_gold=True)
    system_annotations = read_annotations(preds, is_gold=False)
    scores = calculate_scores(gold_annotations, system_annotations)
    
    display_names = [(None, 'Overall'), (Gender.MASCULINE, 'Masculine'),
                   (Gender.FEMININE, 'Feminine')]

    row = {}
    for gender, display_name in display_names:
        gender_scores = scores.get(gender, Scores())
        f1 = gender_scores.f1()
        row[display_name] = [f1]
    score = pd.DataFrame.from_dict(row)
    score.columns = [col[0] for col in score.columns]
    score['B'] = score['F']/score['M']
    score = score[['M', 'F', 'B', 'O']]
    if index is not None:
        score.index = [index]
    
    score_df = pd.concat([score_df, score])
    if verbose:
      display(score_df.round(2))
    return score_df

def calculate_scores(gold_annotations, system_annotations):
  """Score the system annotations against gold.

  Args:
    gold_annotations: dict from example ID to its gold Annotation.
    system_annotations: dict from example ID to its system Annotation.

  Returns:
    A dict from gender to a Scores object for that gender. None is used to
      denote no specific gender, i.e. overall scores.
  """
  scores = {}
  for example_id, gold_annotation in gold_annotations.items():
    system_annotation = system_annotations[example_id]

    name_a_annotations = [
        gold_annotation.name_a_coref, system_annotation.name_a_coref
    ]
    name_b_annotations = [
        gold_annotation.name_b_coref, system_annotation.name_b_coref
    ]
    for gender in [None, gold_annotation.gender]:
      if gender not in scores:
        scores[gender] = Scores()

      for (gold, system) in [name_a_annotations, name_b_annotations]:
        if system is None:
          print('Missing output for', example_id)
          scores[gender].false_negatives += 1
        elif gold and system:
          scores[gender].true_positives += 1
        elif not gold and system:
          scores[gender].false_positives += 1
        elif not gold and not system:
          scores[gender].true_negatives += 1
        elif gold and not system:
          scores[gender].false_negatives += 1
  return scores

def read_annotations(df, is_gold):
    fieldnames = GOLD_FIELDNAMES if is_gold else SYSTEM_FIELDNAMES

    annotations = defaultdict(Annotation)

    for idx, row in df.iterrows():
      example_id = row['id']
      if example_id in annotations:
        print('Multiple annotations for', example_id)
        continue

      annotations[example_id].name_a_coref = row['a_coref']
      annotations[example_id].name_b_coref = row['b_coref']
      if is_gold:
        gender = PRONOUNS.get(row['pronoun'].lower(), Gender.UNKNOWN)
        assert gender != Gender.UNKNOWN, row
        annotations[example_id].gender = gender
    return annotations