#!/usr/bin/env python

def main():
    with open('data.txt','r') as f:
        for line in f:
            print line,

if __name__ == '__main__':
    main()

