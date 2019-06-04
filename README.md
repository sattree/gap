# Gendered Ambiguous Pronouns (GAP) - ProBERT and GREP

This repo contains code for the paper [Gendered Ambiguous Pronouns Shared Task: Boosting Model Confidence
by Evidence Pooling](https://arxiv.org/pdf/1906.00839.pdf)

and the winning model in the Kaggle competition [Gendered Pronoun Resolution](https://www.kaggle.com/c/gendered-pronoun-resolution/leaderboard)

*If you use this code for your research, please [cite the paper](#bibtex)*


## Setup

### Hardware/Platform Specs

The models were trained using 4 V100 gpus. 
It is possible to train the models on a single gpu to get a comparable performance by adjusting the batch size accordingly.

All models were developed and tested in

```
python3.6
```

### Dependencies

Most of the dependencies are listed in requirement.txt and can be installed by

```
pip install requirements.txt
```

Note that this file was generated automatically and you may need to resolve certain dependencies manually.

### Download

The coref models need the following external data to be downloaded and placed in externals/data/

```
curl -O https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

curl -O https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz
curl -O https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz
```

Lee et all e2e-coref model checkpoints:

```
Download pretrained models at https://drive.google.com/file/d/1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi
Move the downloaded file to externals/data and extract: tar -xzvf e2e-coref.tgz
```

Refer https://github.com/kentonl/e2e-coref for additional configuration if needed.

Stanford CoreNLP package needs to be downloaded and placed in externals/

```
curl -O http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
rm stanford-corenlp-full-2018-10-05.zip
```

## Data

The GAP dataset along with the corrections is included in this repo in the 'data' folder.

## Preprocessing

```
python run.py --preprocess_train 
              --model=grep 
              --language_model=bert-large-uncased 
              --coref_models=url,allen,hug,lee 
              --exp_dir=results/grep
```

## Training

Trained models are not included as part of the archive due to their large size.

Models can be trained by executing the following command from project root

```
python run.py --train 
              --model=grep 
              --language_model=bert-large-uncased 
              --coref_models=url,allen,hug,lee 
              --exp_dir=results/grep
```

## Prediction

The trained models can used for prediction by running the code in predict mode

```
python run.py --predict 
              --preprocess_eval 
              --model=grep 
              --language_model=bert-large-uncased 
              --coref_models=url,allen,hug,lee 
              --exp_dir=results/grep
```

## Kaggle submission

To reproduce kaggle submission results

```
!python run.py --train 
               --kaggle 
               --preprocess_train 
               --preprocess_eval 
               --model=grep 
               --language_model=bert-large-uncased 
               --coref_models=url,allen,hug,lee 
               --verbose=1 
               --exp_dir=results/kaggle 
               --test_path=data/test_stage_2.tsv 
               --sub_sample_path=sample_submission_stage_2.csv
```

NOTE: It is possible to run the pipeline end to end by using appropriate flags.

usage.ipynb contains example usage of the pipeline.



## BibTex



