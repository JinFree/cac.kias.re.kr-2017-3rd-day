#!/usr/bin/env python
import numpy as np

def main():
    data = np.random.random(1000)
    mask = np.logical_and( (data >= 0.2), (data < 0.4) )

#    avg = np.average( data[mask] )
    avg = np.mean( data[mask] )

    print 'Average of data between 0.2 and 0.4'
    print avg

if __name__ == '__main__':
    main()

