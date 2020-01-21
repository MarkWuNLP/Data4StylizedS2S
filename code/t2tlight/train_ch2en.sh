RUN_T2T=/home/v-wuyu/code/t2tlight

DATA_PATH=/home/v-wuyu/data/subtitle/teaching
TRAIN_SRC=/home/v-wuyu/data/subtitle/teaching/chs_total.wb.bpe.txt.shuf
TRAIN_TRG=/home/v-wuyu/data/subtitle/teaching/enu_total.wb.bpe.txt.shuf

VALID=/home/v-wuyu/data/crosslingual_style/test/informal_wc.txt.zh.wb.bpe
REFS=/home/v-wuyu/data/crosslingual_style/test/informal_wc.txt.zh.wb.bpe
TRANS=$DATA_PATH/Test/test.txt.trans
SCORE=$DATA_PATH/Test/test.txt.trans.score

VOCAB_SRC=$DATA_PATH/vocab.zh
VOCAB_TRG=$DATA_PATH/vocab.en

MODEL=/home/v-wuyu/model/Subtitle_Teaching

PARAMS_BIG="shared_source_target_embedding=False,device_list=[0,1,2,3,4,5,6,7],train_steps=300000,batch_size=6000,eval_steps_begin=2000000,eval_steps=20000,save_checkpoint_steps=4000,adam_beta2=0.997,hidden_size=1024,filter_size=4096,num_heads=16,keep_checkpoint_max=5,learning_rate=0.3,decode_alpha=1.1,decode_length=100,beam_size=18,eval_batch_size=16"

PARAMS_BASE="shared_source_target_embedding=False,device_list=[0,1],train_steps=200000,batch_size=3000,eval_steps_begin=300000,eval_steps=200000,save_checkpoint_steps=5000,adam_beta2=0.997"

MODE=train

if [ $MODE == "prepare" ]
then

echo "Shuffle Dataset........"
python $RUN_T2T/scripts/shuffle_dataset.py --input $TRAIN_SRC $TRAIN_TRG 
echo "Build Vocab........"
python $RUN_T2T/scripts/build_vocab.py $TRAIN_SRC.shuf $VOCAB_SRC --vocabsize 45000
python $RUN_T2T/scripts/build_vocab.py $TRAIN_TRG.shuf $VOCAB_TRG --vocabsize 33000

elif [ $MODE == "train" ]
then

echo "Start Training........"
/home/v-wuyu/anaconda3/bin/python3 $RUN_T2T/train.py --input $TRAIN_SRC $TRAIN_TRG \
    --output $MODEL \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --validation $VALID \
    --references $REFS \
    --parameters=$PARAMS_BASE

elif [ $MODE == "test" ]
then

echo "Start Testing........"
python3 $RUN_T2T/translate.py --input $VALID \
    --output $TRANS \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --models $MODEL \
    --parameters="device_list=[0],decode_alpha=1.2,decode_batch_size=32,beam_size=4"

elif [ $MODE == "score" ]
then

echo "Start Scoring........"
python3 $RUN_T2T/score.py --input $VALID $TRANS \
    --output $SCORE \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --model $MODEL \
    --parameters="device_list=[3]"
    
fi
