import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="inputfile")
ap.add_argument("-oc", "--outputChinese", required=True, help="outputfile Chinese")
ap.add_argument("-oe", "--outputEnglish", required=True, help="outputfile English")
args = ap.parse_args()


f = open(args.input, "r",encoding="utf-8")
fw2 = open(args.outputChinese, "w",encoding="utf-8")
fw = open(args.outputEnglish, "w",encoding="utf-8")

for line in f:
    try:
        tmp = line.strip().split("\t")
        a, b = tmp[0], tmp[1]
        fw.write(a)
        fw.write("\n")
        fw2.write(b)
        fw2.write("\n")
    except:
        print(line)
        pass

fw.close()
fw2.close()
# count = 0
# f = open(r"D:\project\formalityClassifier\data\chs_total.wb.bpe.txt","r",encoding="utf-8")
# for line in f:
#     #print(line)
#     count += 1
# print(count)