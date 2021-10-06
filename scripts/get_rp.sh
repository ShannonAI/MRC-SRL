for name in train dev test_wsj test_brown
do
    python ./module/RolePrediction/predict.py \
    --dataset_tag conll2005 \
    --dataset_path ./data/conll2005/${name}.english.psense.conll05.json \
    --output_path ./data/conll2005/${name}.english.psense.plabel.conll05.json \
    --checkpoint_path ./checkpoints/conll2005/role_prediction/2021_09_12_14_02_07/checkpoint_7.cpt  \
    --max_tokens 2048 \
    --alpha 5 \
    --save \
    --amp
done