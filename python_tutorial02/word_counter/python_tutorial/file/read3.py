#!/usr/bin/env python

def main():
    with open('data.txt','r') as f:
        lines = f.readlines()
    for line in lines:
        print line,

if __name__ == '__main__':
    main()

