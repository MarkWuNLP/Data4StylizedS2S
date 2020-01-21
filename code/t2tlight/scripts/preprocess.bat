SET SRC="\\msra-sandvm-001\v-wuyu\Data\ACL2019\HumanLabel\MT\test\formal_wc.txt"
SET TGT="\\msra-sandvm-001\v-wuyu\Data\ACL2019\HumanLabel\MT\test\informal_wc.txt"
python "D:\ZhiruiCode\t2tlight\scripts\preprocess.py" -i %SRC% -oc %SRC%.zh -oe %SRC%.en
python "D:\ZhiruiCode\t2tlight\scripts\preprocess.py" -i %TGT% -oc %TGT%.zh -oe %TGT%.en
D:
cd D:\WordBreaker_EXE
Segment_Refine_Trans.exe -tenglish 0 -numthre 11 -trepneforchilm 0 -twn 0 -tcs 1 -tmb 0 -fi %SRC%.zh -fo %SRC%.zh.wb -dnne 1
Segment_Refine_Trans.exe -tenglish 0 -numthre 11 -trepneforchilm 0 -twn 0 -tcs 1 -tmb 0 -fi %TGT%.zh -fo %TGT%.zh.wb -dnne 1
python "D:\ZhiruiCode\t2tlight\scripts\apply_bpe.py" -i %TGT%.zh.wb -c "D:\ZhiruiCode\t2tlight\scripts\c.code" -o %TGT%.zh.wb.bpe
python "D:\ZhiruiCode\t2tlight\scripts\apply_bpe.py" -i %SRC%.zh.wb -c "D:\ZhiruiCode\t2tlight\scripts\c.code" -o %SRC%.zh.wb.bpe
python "D:\ZhiruiCode\t2tlight\scripts\apply_bpe.py" -i %TGT%.zh.wb -c "D:\project\formalityClassifier\data\SubtitleData\subtitle_ch.code" -o %TGT%.zh.wb.bpe2
python "D:\ZhiruiCode\t2tlight\scripts\apply_bpe.py" -i %SRC%.zh.wb -c "D:\project\formalityClassifier\data\SubtitleData\subtitle_ch.code" -o %SRC%.zh.wb.bpe2