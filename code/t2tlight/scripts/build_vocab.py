""" 
Build Vocabulary
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections

def parse_args():
    parser = argparse.ArgumentParser(description="Create Vocabulary")

    parser.add_argument("inputfile", help="Dataset")
    parser.add_argument("outputfile", help="Vocabulary Name")
    parser.add_argument("--vocabsize", default=30000, type=int, help="Vocabulary Size")

    return parser.parse_args()

def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r", encoding="utf-8") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))

    return words, counts

def save_vocab(filename, vocab):
    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words = list(zip(*pairs))[0]

    with open(filename, "w",encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

def main(args):
    vocab = {}
    count = 0
    words, counts = count_words(args.inputfile)
    
    vocab["<EOS>"] = 0  #insert end-of-sentence/unknown/begin-of-sentence symbols
    vocab["<UNK>"] = 1
    vocab["<BOS>"] = 2

    for word, freq in zip(words, counts):
        if args.vocabsize and len(vocab) >= args.vocabsize:
            break
        
        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue

        vocab[word] = len(vocab)
        count += freq

    save_vocab(args.outputfile, vocab)
    
    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f" % (100.0 * count / sum(counts)))

if __name__ == "__main__":
    main(parse_args())