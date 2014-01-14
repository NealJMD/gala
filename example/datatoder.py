import numpy as np
import h5py
import sys
import scipy
from scipy import misc

arg1 = sys.argv[1]
arg2 = sys.argv[2]

og = h5py.File(arg1, mode='r')
data = og['volume']['predictions']
npdata = np.array(data)
print npdata.shape

new_img = []

num_images = len(npdata[:])

for i in range(0,1):
    bmc = npdata[i]
    gradients = np.gradient(bmc)
    ygrad = gradients[0]
    xgrad = gradients[1]
    gradient_mags = np.sqrt(ygrad**2 + xgrad**2)
    new_img = gradient_mags

print new_img.shape
scipy.misc.imsave(arg2 + '.png', new_img)

