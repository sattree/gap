import pandas as pd
import numpy as np

from sklearn.metrics import log_loss

from models.data_pipeline import data_pipeline_v2, Dataset, CorefAnnotator

def load_data_version(data_dir=None,
                     exp_dir=None,
                     tst_data_version=None,
                     sanitize_labels=None,
                     persist=True,
                     coref_extractor=None,
                     proref_extractor=None
                 ):
    
    train = {
        'input': Dataset().transform('{}/gap-test.tsv'.format(data_dir), 
                                    label_corrections='{}/gap_corrections/my_corrections_tst.csv'.format(data_dir))
    }

    val = {
        'input': Dataset().transform('{}/gap-validation.tsv'.format(data_dir), 
                                    label_corrections='{}/gap_corrections/my_corrections_val.csv'.format(data_dir))
    }

    test = {
        'input': Dataset().transform('{}/gap-development.tsv'.format(data_dir), 
                                    label_corrections='{}/gap_corrections/my_corrections_dev.csv'.format(data_dir))
    }

    neither = {
        'input': Dataset().transform('{}/gpr-neither.tsv'.format(data_dir),
                                    )
    }

    test_stage2 = {
        'input': Dataset().transform('{}/{}.tsv'.format(data_dir, tst_data_version),
                                    label_corrections='{}/gap_corrections/my_corrections_tst_stage2.csv'.format(data_dir),
                                    shift_by_one=False)
    }

    dpl_trn = data_pipeline_v2(exp_dir, 
                           mode='train', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=True, 
                           pretrained_proref=True,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_extractor=coref_extractor,
                           proref_extractor=proref_extractor
                        )
    dpl_val = data_pipeline_v2(exp_dir, 
                           mode='val', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=True, 
                           pretrained_proref=True,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_extractor=coref_extractor,
                           proref_extractor=proref_extractor
                        )
    dpl_tst = data_pipeline_v2(exp_dir, 
                           mode='test', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=True, 
                           pretrained_proref=True,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_extractor=coref_extractor,
                           proref_extractor=proref_extractor
                        )

    dpl_neither = data_pipeline_v2(exp_dir, 
                           mode='neither', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=True, 
                           pretrained_proref=True,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_extractor=coref_extractor,
                           proref_extractor=proref_extractor
                        )

    dpl_tst_2 = data_pipeline_v2(exp_dir, 
                           mode='test_stage2', 
                           annotate_mentions=True, 
                           annotate_coref_mentions=True, 
                           pretrained_proref=True,
                           sanitize_labels=sanitize_labels,
                           persist=persist,
                           coref_extractor=coref_extractor,
                           proref_extractor=proref_extractor
                        )

    X_trn = dpl_trn.gather_step.transform(train)['X']
    X_val = dpl_val.gather_step.transform(val)['X']
    X_tst = dpl_tst.gather_step.transform(test)['X']
    X_neither = dpl_neither.gather_step.transform(neither)['X']
    X_tst_2 = dpl_tst_2.gather_step.transform(test_stage2)['X']

    train = pd.concat([X_trn, X_val, X_tst, X_neither, X_neither.head(3)]).reset_index(drop=True)
    print(train.shape)
    
    return X_trn, X_val, X_tst, train, X_tst_2

def get_score(probs, data):
    y_true = data['label']
    return round(log_loss(y_true, probs[:len(y_true), :])*100, 3)

def get_val_scores(predictions, 
                   lms, 
                   seeds, 
                   train_san, 
                   train_unsan, 
                   tst_san, 
                   tst_unsan,
                  X_trn,
                  X_val):
    # CV ensemble score
    probs_all = pd.concat([pd.read_csv(file) for file in predictions], axis=1).values.reshape(-1, len(lms), len(seeds), 3).transpose(1, 2, 0, 3)

    index = []
    oof_all_rows = []
    oof_tst_rows = []
    for j, lm in enumerate(lms):
        for i, seed in enumerate(seeds):
            index.append('{} {}'.format(lm, seed))

            probs = probs_all[j][i]
            oof_all_rows.append((get_score(probs, train_san), get_score(probs, train_unsan)))

            probs = probs_all[j][i, len(X_trn)+len(X_val):len(X_trn)+len(X_val)+len(tst_san), :]
            oof_tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))

        index.append('{} {}'.format(lm, 'mean cv'))
        
        oof_all_rows.append(np.mean(oof_all_rows[-5:], axis=0))
        oof_tst_rows.append(np.mean(oof_tst_rows[-5:], axis=0))
        
        index.append('{} {}'.format(lm, 'seed-based ensemble'))

        probs = probs_all[j].mean(axis=0)
        oof_all_rows.append((get_score(probs, train_san), get_score(probs, train_unsan)))

        probs = probs_all[j].mean(axis=0)[len(X_trn)+len(X_val):len(X_trn)+len(X_val)+len(tst_san), :]
        oof_tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))

    index.append('lm-based ensemble')
    cv_probs = probs = probs_all.mean(axis=0).mean(axis=0)
    oof_all_rows.append((get_score(probs, train_san), get_score(probs, train_unsan)))

    probs = probs_all.mean(axis=0).mean(axis=0)[len(X_trn)+len(X_val):len(X_trn)+len(X_val)+len(tst_san), :]
    oof_tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))

    cols = pd.MultiIndex.from_product([['oof_all', 'oof_tst'], ['sanitized', 'unsanitized']])
    return pd.DataFrame(np.hstack((oof_all_rows, oof_tst_rows)), index=index, columns=cols), cv_probs, probs

def get_tst_scores(predictions, 
                   lms, 
                   seeds, 
                   n_folds,
                   tst_san, 
                   tst_unsan):
    
    probs_raw = pd.concat([pd.read_csv(file) for file in predictions], axis=1).values.reshape(-1, len(lms), len(seeds), n_folds, 3).transpose(1, 2, 3, 0, 4)

    index = []
    tst_rows = []
    for i, lm in enumerate(lms):
        for j, seed in enumerate(seeds):
            for k in range(n_folds):
                index.append('{} {} {}'.format(lm, seed, k))

                probs = probs_raw[i][j][k]
                tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))


            index.append('{} {} {}'.format(lm, seed, 'fold-based ensemble'))

            probs = probs_raw[i][j].mean(axis=0)
            tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))

        index.append('{} {}'.format(lm, 'seed-based ensemble'))

        probs = probs_raw[i].mean(axis=0).mean(axis=0)
        tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))

    index.append('lm-based ensemble')
    probs = probs_raw.mean(axis=0).mean(axis=0).mean(axis=0)
    tst_rows.append((get_score(probs, tst_san), get_score(probs, tst_unsan)))

    cols = pd.MultiIndex.from_product([['tst'], ['sanitized', 'unsanitized']])
    return pd.DataFrame(tst_rows, index=index, columns=cols)