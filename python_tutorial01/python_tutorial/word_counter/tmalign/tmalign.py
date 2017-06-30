#!/usr/bin/env python
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='calculcate TM-score')
    parser.add_argument('pdb1',help='pdb file1')
    parser.add_argument('pdb2',help='pdb file2')

    args = parser.parse_args()

    tmscore1, tmscore2 = tmalign(args.pdb1,args.pdb2)

    print tmscore1, tmscore2

def tmalign(pdb1,pdb2):
    scores = []
    p = subprocess.Popen(('./tmalign',pdb1,pdb2),stdout=subprocess.PIPE)
    for line in p.stdout:
        if line.startswith('TM-score='):
            scores.append( float(line.split()[1]) )
    return tuple(scores)

if __name__ == '__main__':
    main()
