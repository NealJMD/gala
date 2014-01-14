import libtiff
from libtiff import TIFF
import numpy as np
import h5py
import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]
if len(sys.argv) == 4:
    arg3 = sys.argv[3]

stack_left = []
stack_right = []
tif = TIFF.open(arg1, mode='r')
for index, image in enumerate(tif.iter_images()):
    if len(sys.argv) == 3:
        np_image = np.array(image)
        np_left = np_image[0:64,0:128]
        stack_left.append(np_left)
    elif len(sys.argv) == 4:
        np_image = np.array(image)
        np_left = np_image[0:64,0:128]
        np_right = np_image[512:1024,:]
        stack_left.append(np_left.tolist())
        stack_right.append(np_right.tolist())

npstack_left = np.array(stack_left) / 255.0
print npstack_left.shape
npstack_right = np.array(stack_right) / 255.0
print npstack_right.shape

f = h5py.File(arg2, 'w')
grp = f.create_group("volume")
#subgrp = grp.create_group("predictions")
#subgrp.create_dataset("stack", npstack_left)
#subgrp['stack'] = npstack_left
grp['predictions'] = npstack_left
f.close()

if len(sys.argv) == 4:
    f2 = h5py.File(arg3, 'w')
    grp = f2.create_group("volume")
    #subgrp = grp.create_group("predictions")
    #subgrp.create_dataset("stack", npstack_right)
    #subgrp['stack'] = npstack_right
    grp['predictions'] = npstack_right
    f2.close() 



