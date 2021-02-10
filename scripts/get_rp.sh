python ./module/RolePrediction/predict.py \
--dataset_tag conll2005 \
--dataset_path ./data/conll2005/dev.english.conll05.json \
--output_path ./data/conll2005/dev.english.plabel.conll05.json \
--checkpoint_path ./checkpoints/conll2005/role_prediction/2021_02_09_20_37_45/checkpoint_3.cpt \
--max_tokens 2048 \
--alpha 5 \
--save \
--amp \