#!/usr/bin/env python

def main():
    with open('data.txt','r') as f:
        text = f.read()
    print text,

if __name__ == '__main__':
    main()

