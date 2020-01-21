#f = open(r"D:\v-wuyu\data\process_data\title_valid_u2l.txt","r",encoding="utf-8")
f = open(r"\\msra-sandvm-001\v-wuyu\Data\EnglishTweets\ACL2019\pair2train.merge.bpe.tgt","r",encoding="utf-8")
f2 = open(r"\\msra-sandvm-001\v-wuyu\Data\EnglishTweets\ACL2019\pair2test.merge.bpe.tgt","w",encoding="utf-8")
#f3 = open(r"/home/v-wuyu/data/title_valid_u2l.txt","w",encoding="utf-8")
count = 0
for line in f:

    if count < 1000:
        f2.write(line)
    else:
        break
    # else:
    #     f3.write(line)
    count += 1