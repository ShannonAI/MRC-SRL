python ./module/ArgumentLabeling/ckpt_eval.py \
--data_path ./data/conll2005/dev.english.plabel.psense.conll05.json \
--checkpoint_path ./checkpoints/conll2005/arg_labeling/2021_02_10_01_13_06/checkpoint_9.cpt \
--arg_query_type 1 \
--argm_query_type 0 \
--gold_level 2 \
--max_tokens 2048 \
--amp