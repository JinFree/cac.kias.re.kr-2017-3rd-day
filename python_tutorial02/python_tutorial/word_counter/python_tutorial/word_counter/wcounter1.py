#!/usr/bin/env python

def main():
    text = raw_input('Input Text: ')
    words = {}
    count_words(words,text)
    print_words(words)

def print_words(words):
    for w in words:
        print w, words[w]

def count_words(words,line):
    for w in line.split():
        if w in words:
            words[w] += 1
        else:
            words[w] = 1

if __name__ == '__main__':
    main()
