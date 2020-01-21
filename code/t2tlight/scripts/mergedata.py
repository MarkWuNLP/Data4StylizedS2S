f1 = open(r"\\msra-sandvm-001\v-wuyu\share\gyafc\new_exp_fr\add_data\inf_selected.txt"
          ,"r",encoding="utf-8")
f2 = open(r"\\msra-sandvm-001\v-wuyu\share\gyafc\new_exp_fr\add_data\fm_selected.txt"
          ,"r",encoding="utf-8")
fw = open(r"\\msra-sandvm-001\v-wuyu\share\gyafc\new_exp_fr\add_data\merge_selected.txt"
          ,"w",encoding="utf-8")

for line in f1:
    fw.write(line)
for line in f2:
    fw.write(line)