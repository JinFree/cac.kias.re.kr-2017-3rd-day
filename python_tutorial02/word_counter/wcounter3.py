#!/usr/bin/env python
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Count words.')
    parser.add_argument('textfile', help='textfile to read')
    parser.add_argument('-i', dest='ignore_case', action='store_true', default=False,
                       help='ignore case')
    args = parser.parse_args()

    words = {}
    with open(args.textfile,'r') as f:
        for line in f:
            count_words(words,line,args.ignore_case)
    print_words(words)

def print_words(words):
    for w in sorted(words):
        print w, words[w]

def count_words(words,line,ignore_case):
    m = re.search('\w.*\w',line)
    if m is None:
        return
    stripped_line = m.group(0)

    split_words = re.split('[\s\W]*',stripped_line)
    for w in split_words:
        if ignore_case:
            w = w.lower()
        if w in words:
            words[w] += 1
        else:
            words[w] = 1

if __name__ == '__main__':
    main()
