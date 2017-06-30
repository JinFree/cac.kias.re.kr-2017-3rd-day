#!/usr/bin/env python
import cPickle as pkl
import numpy as np

data = np.random.random((10,3))

# numpy text format
np.savetxt('data.txt',data,fmt='%.4f %.4f %.4f',header='data1 data2 data3')
# numpy binary
np.save('data',data)
# pickle
with open('data.pkl','w') as f:
    pkl.dump(data,f,2)
