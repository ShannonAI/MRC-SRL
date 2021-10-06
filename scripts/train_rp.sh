nohup python ./module/RolePrediction/train.py \
--dataset_tag conll2005 \
--pretrained_model_name_or_path roberta-large \
--train_path ./data/conll2005/train.english.conll05.json  \
--dev_path ./data/conll2005/dev.english.conll05.json  \
--max_tokens 1024 \
--max_epochs 8 \
--lr 8e-6 \
--max_grad_norm 1 \
--warmup_ratio -1 \
--eval \
--save \
--amp \
--tqdm_mininterval 500 \
>log_rp.txt 2>&1 &
#cat log_rp.txt