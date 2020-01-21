import nltk
def normtxt():
    f = open("../results/formal.en.pred_case_subtitle2","r",encoding="utf-8")
    fw = open("../results/formal.en.pred.norm_case_subtitle2","w",encoding="utf-8")

    for line in f:
        s = line.replace("@@ ","")
        words = s.split()
        tmp = list(words[0])
        tmp[0] = tmp[0].upper()
        words[0] = "".join(tmp)
        #line[0] = line[0].upper()
        fw.write(" ".join(words)+"\n")

def computeBLEU():
    from utils.bleu import compute_bleu
    import numpy
    totalbleu = []
    f = open("../results/1-500/formal_eng_trans", "r",encoding="utf-8")
    fw = open("../results/1-500/formal_eng_trans.norm", "w",encoding="utf-8")
    fw2 = open("../results/1-500/formal_groundtruth.norm", "w",encoding="utf-8")
    f2 = open("../results/1-500/formal_1-500_xin.en.token.txt", "r",encoding="utf-8")
    trans, ground = [], []
    for l,l2 in zip(f,f2):
        l = l.replace("@@ ","")
        fw.write(l.lower())
        fw2.write(l2.lower())
        trans.append(l.strip().lower().split())
        ground.append([l2.strip().lower().split()])
    #nltk.translate.bleu_score.sentence_bleu([])
        bleu = nltk.translate.bleu_score.sentence_bleu([l2.strip().split()],l.strip().split(), weights=(0, 0, 0, 1))
        #print(bleu)
        totalbleu.append(bleu)

    print(numpy.asarray(totalbleu).mean())
normtxt()
#computeBLEU()
