import os
from glob import glob

path = 'aaa/bbb/ccc/ddd'

print os.path.dirname(path)
print os.path.basename(path)
print os.path.abspath('..')

# globbing
print glob('../*')

# path spliting
print path.split(os.sep)
