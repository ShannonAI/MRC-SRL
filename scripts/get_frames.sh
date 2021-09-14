#conll2005
python ./scripts/frames2json.py \
--propbank_dir  /data/wangnan/CoNLL/scr/corpora/ldc/2012/LDC2012T04/LDC2012T04_CoNLL_Shared_Task_Part_2/data/CoNLL2009-ST-English/pb_frames \
--output_path ./data/conll2005/frames.json \

#conll2009
python ./scripts/frames2json.py \
--propbank_dir /data/wangnan/CoNLL/scr/corpora/ldc/2012/LDC2012T04/LDC2012T04_CoNLL_Shared_Task_Part_2/data/CoNLL2009-ST-English/pb_frames \
--nombank_dir /data/wangnan/CoNLL/scr/corpora/ldc/2012/LDC2012T04/LDC2012T04_CoNLL_Shared_Task_Part_2/data/CoNLL2009-ST-English/nb_frames \
--output_path ./data/conll2009/frames.json \

#conll2012
python ./scripts/frames2json.py \
--propbank_dir /data/wangnan/CoNLL/ontonotes-release-5.0/data/files/data/english/metadata/frames \
--output_path ./data/conll2012/frames.json \
--pos \