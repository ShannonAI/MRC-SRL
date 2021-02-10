python ./module/ArgumentLabeling/train.py \
--pretrained_model_name_or_path roberta-large \
--train_path ./data/conll2005/dev.english.plabel.psense.conll05.json \
--dev_path ./data/conll2005/nev.english.plabel.psense.conll05.json  \
--dataset_tag conll2005 \
--max_tokens 1024 \
--max_epochs 20 \
--lr 1e-5 \
--max_grad_norm 1 \
--warmup_ratio 0.05 \
--arg_query_type 1 \
--argm_query_type 0 \
--gold_level 2 \
--eval \
--save \
--amp \

