nohup python ./module/ArgumentLabeling/train.py \
--dataset_tag conll2005 \
--pretrained_model_name_or_path roberta-large \
--train_path ./data/conll2005/train.english.psense.plabel.conll05.json \
--dev_path ./data/conll2005/dev.english.psense.plabel.conll05.json  \
--max_tokens 1024 \
--max_epochs 20 \
--lr 1e-5 \
--max_grad_norm 1 \
--warmup_ratio 0.01 \
--arg_query_type 2 \
--argm_query_type 1 \
--gold_level 1 \
--tensorboard \
--eval \
--save \
--amp \
--tqdm_mininterval 500 \
>log_al.txt 2>&1 &
#cat log_al.txt
