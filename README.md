# An MRC Framework for Semantic Role Labeling
This repo contains code for paper [An MRC Framework for Semantic Role Labeling](https://arxiv.org/abs/2109.06660).

## Introduction
Semantic  Role  Labeling  (SRL)  aims  at  recognizing  the  predicate-argument  structure  of a  sentence  and  can  be  decomposed  into  two subtasks:  predicate disambiguation and argument labeling. Prior work deals with these twotasks independently, which ignores the semantic connection between the two tasks.  In this paper, we propose to use the machine readingcomprehension  (MRC)  framework  to  bridge this gap.  We formalize predicate disambiguation as multiple-choice machine reading comprehension,  where  the  descriptions  of  candidate  senses  of  a  given  predicate  are  used  as options to select the correct sense.  The chosen predicate sense is then used to determine the semantic roles for that predicate, and these semantic roles are used to construct the query for another MRC model for argument labeling. In this way, we are able to leverage both the predicate semantics and the semantic role semantics  for  argument  labeling.  We  also  propose to select a subset of all the possible semantic roles for computational efficiency. Experiments show that the proposed framework achieves  state-of-the-art  results  on  both  span and dependency benchmarks.

## Results
Table 1: Results for predicate disambiguation.
| Model                  |  Dev |  WSJ | Brown |
|------------------------|:----:|:----:|:-----:|
| Shi and Zhang (2017)   |   -  | 93.4 |  82.4 |
| Roth and Lapata (2016) | 94.8 | 95.5 |   -   |
| He et al. (2018b)      | 95.0 | 95.6 |   -   |
| Shi and Lin (2019)     | 96.3 | 96.9 |  90.6 |
| Ours                   | 96.8 | 97.3 |  91.3 |

Table 2: Results for argument labeling.
| Model                              | Encoder | CoNLL05 WSJ | CoNLL05 Brown | CoNLL09 WSJ | CoNLL09 Brown | CoNLL12 Test |
|------------------------------------|---------|:-----------:|:-------------:|:-----------:|:-------------:|:------------:|
| Zhou, Li, and Zhao (2020)          | BERT    |     88.9    |      81.4     |     91.2    |      85.9     |       -      |
| Mohammadshahi and Henderson (2021) | BERT    |     88.9    |      83.2     |     91.2    |      86.4     |       -      |
| Xia et al. (2020)                  | RoBERTa |     88.6    |      83.2     |      -      |       -       |       -      |
| Marcheggiani and Titov (2020)      | RoBERTa |     88.0    |      80.6     |      -      |       -       |     86.8     |
| Conia and Navigli (2020)           | BERT    |      -      |       -       |     92.6    |      85.9     |     87.3     |
| Jindal et al. (2020)               | BERT    |     87.9    |      80.2     |     90.8    |      85.0     |     86.6     |
| Paolini et al. (2021)              | T5      |     89.3    |      82.0     |      -      |       -       |     87.7     |
| Blloshmi et al. (2021)             | BART    |      -      |       -       |     92.4    |      85.2     |     87.3     |
| Shi and Lin (2019)                 | BERT    |     88.8    |      82.0     |     92.4    |      85.7     |     86.5     |
| Ours                               | BERT    |     89.3    |      84.7     |     93.0    |      87.0     |     87.8     |
| Ours                               | RoBERTa |     90.0    |      85.1     |     93.3    |      87.2     |     88.3     |


## Usage
### Requirements
- python>=3.6
- pip install -r requirements.txt

### Dataset Preparation
#### CoNLL2005
The data is provided by: [CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html), but the original words are from the Penn Treebank dataset, which is not publicly available. If you have the PTB corpus, you can run:

`./scripts/make_conll05_data.sh /path/to/ptb/`

#### CoNLL2009
The data is provided by: [CoNLL-2009 Shared Task](http://ufal.mff.cuni.cz/conll2009-st/index.html). Run: 

`./scripts/make_conll2009_data.sh /path/to/conll-2009`

#### CoNLL2012
You can follow this [instruction](https://cemantix.org/data/ontonotes.html) to get the data for CoNLL2012 which will result in a directory called /path/to/conll-formatted-ontonotes-5.0, and we also provided the missing `skeleton2conll` scripts for this instruction in the `scripts` directory. Run:

`./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0`

#### Frame Files
You can follow the example below to get the parsed frame files:

`./scripts/get_frames.sh`

### Pretrained Models Preparation
We use [Roberta-Large](https://huggingface.co/roberta-large)

### Training
You can follow the example scripts below to get the model:
- Train predicate disambiguation model: `./scripts/train_pd.sh`
- Get predicate disambiguation results: `./scripts/get_pd.sh`
- Train role prediction model: `./scripts/train_rp.sh`
- Get role prediction results: `./scripts/get_rp.sh`
- Train Argument labeling model: `./scripts/train_al.sh`
### Evaluation
You can follow the example scripts below to evaluate the trained model:
- Predicate Disambiguation: `./scripts/eval_pd.sh`
- Role Prediction: `./scripts/eval_rp.sh`
- Argument Labeling: `./scripts/eval_al.sh`

## Acknowledgement
The data processing is partially from from [unisrl](https://github.com/bcmi220/unisrl).


## License
[Apache License 2.0](license_link_here)
