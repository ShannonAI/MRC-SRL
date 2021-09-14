# An MRC Framework for Semantic Role Labeling
This repo contains code for paper [An MRC Framework for Semantic Role Labeling](.).


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
