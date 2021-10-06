for name in train dev test_wsj test_brown
do
    python ./module/PredicateDisambiguation/predict.py \
    --frames_path ./data/conll2009/frames.json \
    --dataset_path ./data/conll2009/${name}.english.conll09.json \
    --output_path ./data/conll2009/${name}.english.psense.conll09.json \
    --checkpoint_path ./checkpoints/conll2009/disambiguation/2021_09_26_18_27_19/checkpoint_5.cpt \
    --max_tokens 2048 \
    --save \
    --amp
done