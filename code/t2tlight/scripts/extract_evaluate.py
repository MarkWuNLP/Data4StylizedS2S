f = open(r"D:\project\formalityClassifier\data\enu_total.lower.wb.bpe.txt","r",encoding="utf-8")
f2 = open(r"D:\project\formalityClassifier\data\chs_total.wb.bpe.txt","r",encoding="utf-8")

fw = open(r"D:\project\formalityClassifier\data\train\chinese.bpe.txt","w",encoding="utf-8")
fw2 =open(r"D:\project\formalityClassifier\data\train\english.bpe.txt","w",encoding="utf-8")

f3 = open(r"D:\project\formalityClassifier\data\test\chinese.bpe.txt","w",encoding="utf-8")
f4 =open(r"D:\project\formalityClassifier\data\test\english.bpe.txt","w",encoding="utf-8")
import random
for line2,line in zip(f,f2):
    if random.randint(0,1000) == 333:
        f3.write(line)
        f4.write(line2)
    else:
        fw.write(line)
        fw2.write(line2)
