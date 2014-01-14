import numpy as np
import h5py
import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]
#arg3 = sys.argv[3]

stack_left = []
stack_right = []
og = h5py.File(arg1, mode='r')
data = og['stack']
npdata = np.array(data)
np_left = npdata[:,0:64,0:128]
#np_right = npdata[:,512:1024,:]

print np_left.shape
#print np_right.shape

f = h5py.File(arg2, 'w')
f['stack'] = np_left
f.close()

#f2 = h5py.File(arg3, 'w')
#f2['stack'] = np_right
#f2.close
