
def back_trans_train_data():
    f = open(r"D:\Script\back_trans.informal2ch.bpe.txt","r",encoding="utf-8")
    fw = open(r"D:\Script\back_trans.informal2ch.clean.bpe.txt","w",encoding="utf-8")

    count = 0
    for line in f:
        words = line.strip().split()
        newwords = []
        prev = ""
        for w in words:
            if w != "." and w!="-":
                newwords.append(w)
                prev = w
            else:
                if prev != "." and w != "-":
                    newwords.append(w)
                    prev = w
                else:
                    continue
        fw.write(" ".join(newwords).replace("迪 幻 字幕 组",""))
        fw.write("\n")
        # if count < 1000:
        #     count+=1
        # else:
        #     break

    fw.close()
back_trans_train_data()

def back_trans_mono_data():
    f = open(r"/home/v-wuyu/data/gyafc/back.trans.txt","r",encoding="utf-8")
    fw = open(r"/home/v-wuyu/data/gyafc/back.trans.clean.txt","w",encoding="utf-8")

    count = 0
    for line in f:
        words = line.strip().split()
        newwords = []
        prev = ""
        for w in words:
            if w != ".":
                newwords.append(w)
                prev = w
            else:
                if prev != ".":
                    newwords.append(w)
                    prev = w
                else:
                    continue
        fw.write(" ".join(newwords))
        fw.write("\n")
        # if count < 1000:
        #     count+=1
        # else:
        #     break

    fw.close()