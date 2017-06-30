#!/usr/bin/env python
import Gnuplot
import numpy as np
import argparse
import os
import tempfile
import shutil

def main():
    parser = argparse.ArgumentParser(description='Plot ramachandran chart.')
    parser.add_argument('pdb_files', metavar='pdb_file', nargs='+', help='pdb files')
    parser.add_argument('-o', default=None, dest='outpdf',metavar='pdf_file', help='output pdf file')
    args = parser.parse_args()

    # prepare a temporary directory
    temp_dir = tempfile.mkdtemp()

    # prepare the data files
    data_files = prepare_data_file(temp_dir,args.pdb_files)

    # plot the data files
    plot_rama(data_files,args.outpdf)

    # remove the temporary directory
    shutil.rmtree(temp_dir)

def prepare_data_file(temp_dir,pdb_files):
    data_files = []
    for pdb_file in pdb_files:
        data_file = temp_dir + os.sep + os.path.basename(pdb_file)
        coord = read_coord(pdb_file) 
        angles = get_phi_psi(coord)
        with open(data_file,'w') as f:
            for ang1, ang2 in angles:
                f.write('%f %f\n'%(ang1,ang2))
        data_files.append(data_file)
    return data_files

def plot_rama(data_files,outpdf):
    plot_items = []
    for df in data_files:
        plot_items.append(" '%s' w points ps 2.0 lw 2.0 ti '%s'"%(df,os.path.basename(df)))
    plot_line = 'plot ' + ','.join(plot_items)

    p = Gnuplot.Gnuplot(debug=1)
    if outpdf is not None:
        p("set term pdf enhanced color font 'Times-New-Roman,10' lw 2.0")
        p("set output '%s'"%outpdf)
    else:
        p('set term X11')
    p("set encoding iso_8859_1")
    p("set key top right")
    p("set size square")
    p("set xtics -180.0,60.0,180.0")
    p("set ytics -180.0,60.0,180.0")
    p("set xrange [-180.0:180.0]")
    p("set yrange [-180.0:180.0]")
    if outpdf is None:
        p('set xlabel "Phi"')
        p('set ylabel "Psi"')
    else:
        p('set xlabel "{/Symbol F}"')
        p('set ylabel "{/Symbol Y}"')
    p(plot_line)
    if outpdf is None:
        raw_input()

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
