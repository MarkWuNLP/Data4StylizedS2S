f = open(r"D:\ZhiruiCode\t2tlight\results\formal.ref0.en","r",encoding="utf-8")
f3 =open(r"\\msranlc-ds\Shujie\Data\GYAFC_Corpus_Tok\Entertainment_Music\tune\formal.ref0","r",encoding="utf-8")
true_num = []
a = [line for line in f]
b = [line for line in f3]
for i, s in enumerate(b):
    if s in a:
        true_num.append(i)
print(len(true_num),len(b))
for i in range(4):
    f2 = open(r"\\msranlc-ds\Shujie\Data\GYAFC_Corpus_Tok\Entertainment_Music\tune\formal.ref"+str(i))
    fw = open(r"D:\ZhiruiCode\t2tlight\results\ref\formal.ref"+str(i),"w")
    count = 0
    for line in f2:
        if count in true_num:
            fw.write(line)
        else:
            pass
            #print(count, "###",line)
        count += 1
print(count)