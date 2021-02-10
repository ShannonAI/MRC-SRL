# An MRC Framework for Semantic Role Labeling
This repo contains code for paper [An MRC Framework for Semantic Role Labeling](.).

## Results
Table 1: Results for predicate disambiguation.

|               | Test WSJ | Test Brown |
|---------------|----------|------------|
| Previous SOTA | 96.9     | 90.6       |
| Ours          | 97.3     | 91.3       |


Table 2: Results for argument labeling.
|               | CoNLL05 WSJ | CoNLL05 Brown | CoNLL09 WSJ | CoNLL09 Brown | CoNLL12 Test |
|---------------|-------------|---------------|-------------|---------------|--------------|
| Previous SOTA | 88.8        | 82.0          | 92.4        | 85.7          | 86.6         |
| Ours          | 89.8        | 84.2          | 93.3        | 87.1          | 88.2         |

## Usage
### Requirements
- python>=3.6
- pip install -r requirements.txt

### Dataset Preparation
#### CoNLL2005
The data is provided by: [CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html), but the original words are from the Penn Treebank dataset, which is not publicly available. If you have the PTB corpus, you can run:
./scripts/make_conll05_data.sh /path/to/ptb/

#### CoNLL2009
The data is provided by: [CoNLL-2009 Shared Task](http://ufal.mff.cuni.cz/conll2009-st/index.html), Run: 
./scripts/make_conll2009_data.sh /path/to/conll-2009

#### CoNLL2012
You have to follow the instructions below to get [CoNLL-2012](https://cemantix.org/data/ontonotes.html) data CoNLL-2012, this would result in a directory called /path/to/conll-formatted-ontonotes-5.0. Run:
./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0
#### Frame Files
You can follow the example below to get the frame files:
./scripts/get_frames.sh

### Pretrained Models Preparation
We use [Roberta-Large](https://huggingface.co/roberta-large)

### Training
#### Train Role Prediction Model
`./scripts/train_rp.sh`
#### Get Role Prediction Result
`./scripts/get_rp.sh`
#### Train Predicate Disambiguation Model
`./scripts/train_pd.sh`
#### Get Predicate Disambiguation Result
`./scripts/get_pd.sh`
### Train Argument Labeling Model
`./scripts/train_al.sh`
### Evaluation
- Role Prediction: `./scripts/eval_rp.sh`
- Predicate Disambiguation: `./scripts/eval_pd.sh`
- Argument Labeling: `./scripts/eval_al.sh`

## Acknowledgement
The data processing is partially from from [unisrl](https://github.com/bcmi220/unisrl)


## License
[Apache License 2.0](license_link_here)