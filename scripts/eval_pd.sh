python ./module/PredicateDisambiguation/predict.py \
--frames_path ./data/conll2005/frames.json \
--dataset_path ./data/conll2005/dev.english.conll05.json \
--checkpoint_path ./checkpoints/conll2005/disambiguation/2021_09_12_10_04_06/checkpoint_5.cpt \
--max_tokens 2048 \
--amp
