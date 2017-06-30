#!/usr/bin/env python
import re

def main():
    textfile = raw_input('Input the name of the textfile: ')
    words = {}
    f = open(textfile,'r')
    for line in f:
        count_words(words,line)
    f.close()
    print_words(words)

def print_words(words):
    for w in sorted(words):
        print w, words[w]

def count_words(words,line):
    m = re.search('\w.*\w',line)
    if m is None:
        return
    stripped_line = m.group(0)

    split_words = re.split('[\s\W]*',stripped_line)
    for w in split_words:
        if w in words:
            words[w] += 1
        else:
            words[w] = 1

if __name__ == '__main__':
    main()
