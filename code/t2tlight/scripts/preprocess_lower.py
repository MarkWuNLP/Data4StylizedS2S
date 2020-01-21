f = open(r"D:\project\formalityClassifier\data\enu_total.wb.txt","r",encoding="utf-8")
fw = open(r"D:\project\formalityClassifier\data\enu_total.lower.wb.txt","w",encoding="utf-8")
for line in f:
    line = line.strip().lower()
    fw.write(line)
    fw.write("\n")

