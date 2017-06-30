#!/usr/bin/env python

def main():
    with open('data.txt','w') as f:
        f.write('This is the first line.\n')
        f.write('This is the second line.\n')

if __name__ == '__main__':
    main()

