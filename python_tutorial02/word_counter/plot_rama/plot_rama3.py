#!/usr/bin/env python
import numpy as np
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot ramachandran chart.')
    parser.add_argument('pdb_files', metavar='pdb_file', nargs='+', help='pdb files')
    parser.add_argument('-o', default=None, dest='outpdf',metavar='pdf_file', help='output pdf file')
    args = parser.parse_args()

    # prepare the data
    data = prepare_data(args.pdb_files)

    # plot the data files
    plot_rama(data,args.outpdf)

def prepare_data(pdb_files):
    data = {}
    for pdb_file in pdb_files:
        pdb_basename = os.path.basename(pdb_file)
        coord = read_coord(pdb_file) 
        angles = get_phi_psi(coord)
        data[pdb_basename] = angles
    return data

def plot_rama(data,outpdf):
    matplotlib.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()

    colors = 'bgrcmyk'
    markers = 'o^s'

    imarker = 0
    icolor = 0
    for name in data:
        d = np.array(data[name])
        ax.plot(d[:,0],d[:,1],colors[icolor]+markers[imarker],label=name)
        imarker += 1
        icolor += 1
        if imarker == len(markers):
            imarker = 0
        if icolor == len(colors):
            icolor = 0

    ax.axis([-180.,180.,-180.,180.])
    ax.legend(numpoints=1,loc='upper right',fontsize=20)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_ticks(np.arange(-180.0,181.0,60.0))
    ax.yaxis.set_ticks(np.arange(-180.0,181.0,60.0))
    plt.xlabel('$\Phi$',fontname='Times New Roman',size=20)
    plt.ylabel('$\Psi$',fontname='Times New Roman',size=20)
    if outpdf is None:
        plt.show()
    else:
        fig.savefig(outpdf, bbox_inches='tight')

def read_coord(pdb):
    f = open(pdb,'r')
    coord = {}
    for line in f:
        if ( not line[0:6] == 'ATOM  ' ): continue
        atomname = line[12:16].strip()
        resseq = int( line[22:26] )
        c = [ float(line[30:38]), float(line[38:46]), float(line[46:54]) ]
        if ( not coord.has_key(resseq) ):
            coord[resseq] = {}
        coord[resseq][atomname] = c
  
    return coord

def get_phi_psi(coord):
    angles = []
    for res in coord:
        try: 
            c_c_prev  = np.array( coord[res-1]['C'] )
            c_ca = np.array( coord[res]['CA'] )
            c_n  = np.array( coord[res]['N'] )
            c_c  = np.array( coord[res]['C'] )
            c_n_next  = np.array( coord[res+1]['N'] )
        except:
            continue

        phi = torsion( c_c_prev, c_n, c_ca, c_c ) * 180.0/np.pi
        psi = torsion( c_n, c_ca, c_c, c_n_next ) * 180.0/np.pi
        angles.append((phi,psi))
    return angles

def torsion(c1,c2,c3,c4):
    v1 = c2-c1
    v2 = c3-c2
    v3 = c4-c3
    cp12 = np.cross(v1,v2)
    cp23 = np.cross(v2,v3)
    return np.arctan2( np.linalg.norm(v2)*np.inner(v1,cp23), np.inner(cp12,cp23) )

if __name__ == '__main__':
    main()
