python ./module/ArgumentLabeling/ckpt_eval.py \
--data_path ./data/conll2005/dev.english.psense.plabel.conll05.json \
--checkpoint_path ./checkpoints/conll2005/arg_labeling/2021_09_13_23_06_21/checkpoint_1.cpt \
--arg_query_type 2 \
--argm_query_type 1 \
--gold_level 1 \
--max_tokens 2048 \
--amp