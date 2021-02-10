python ./module/RolePrediction/predict.py \
--dataset_tag conll2005 \
--dataset_path ./data/conll2005/dev.english.conll05.json \
--checkpoint_path ./checkpoints/conll2005/2021_02_09_20_22_55/checkpoint_4.cpt \
--max_tokens 2048 \
--alpha \
--amp \