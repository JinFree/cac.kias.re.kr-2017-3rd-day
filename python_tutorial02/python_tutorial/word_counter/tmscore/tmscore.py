import tmscore_wrap
import numpy as np

NMAX = 3000

def calc_tmscore(pdbfile1,pdbfile2):
    n1,x1,y1,z1 = read_pdb_ca(pdbfile1)
    n2,x2,y2,z2 = read_pdb_ca(pdbfile2)

    l1 = n1.size 
    l2 = n2.size

    x1in = np.empty(NMAX)
    y1in = np.empty(NMAX)
    z1in = np.empty(NMAX)
    n1in = np.empty(NMAX,dtype=int)
    x2in = np.empty(NMAX)
    y2in = np.empty(NMAX)
    z2in = np.empty(NMAX)
    n2in = np.empty(NMAX,dtype=int)
    x1in[0:l1] = x1
    y1in[0:l1] = y1
    z1in[0:l1] = z1
    n1in[0:l1] = n1
    x2in[0:l2] = x2
    y2in[0:l2] = y2
    z2in[0:l2] = z2
    n2in[0:l1] = n2

    tm, rcomm, lcomm = tmscore_wrap.tmscore(l1,x1in,y1in,z1in,n1in,l2,x2in,y2in,z2in,n2in)
    return tm, rcomm, lcomm


def read_pdb_ca(pdbfile):
    f = open(pdbfile,'r')
    n = []
    xarr = []
    yarr = [] 
    zarr = [] 
    for line in f:
        record = line[:6]
        if ( record != 'ATOM  ' and record != 'HETATM' ): continue
        atomname = line[12:16]
        if atomname != ' CA ':
            continue
        resnum = int( line[22:26] )
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        n.append(resnum)
        xarr.append(x)
        yarr.append(y)
        zarr.append(z)
    f.close()

    return np.array(n), np.array(xarr), np.array(yarr), np.array(zarr)
