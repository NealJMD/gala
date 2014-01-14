import numpy as np
import h5py
import sys
from scipy.sparse import *
from scipy import *


def eval_metric(gt, prop):
    gt = h5py.File(gt, mode='r')
    prop = h5py.File(prop, mode='r')

    data_gt = np.array(gt['stack'])
    data_prop = np.array(prop['stack'])


#npdata = np.array(data)
#np_left = npdata[:,0:512,:]
#np_right = npdata[:,512:1024,:]

#print data_gt.shape
#print data_prop.shape

    segA = np.ravel(data_gt)
    segB = np.ravel(data_prop)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

#print n_labels_A
#print n_labels_B

    ones_data = np.ones(n)

#print ones_data.shape
#print segA[:].shape
#print segB[:].shape

    p_ij = csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

#print p_ij.shape

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = np.array(b.todense()) ** 2

#print a.shape
#print b.shape

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

#print a_i
#print b_i

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

#print sumA
#print sumB
#print sumAB

    prec = sumAB / sumB
    rec = sumAB / sumA

    fScore = 2.0 * prec * rec / (prec + rec)
    re = 1.0 - fScore
    return re
#print prec
#print rec
#print fScore
#print re

#f = h5py.File(arg2, 'w')
#f['stack'] = np_left
#f.close()

#f2 = h5py.File(arg3, 'w')
#f2['stack'] = np_right
#f2.close

arg1 = sys.argv[1]
arg2 = sys.argv[2:]

thresh = 0.00

for arg in arg2:
   print arg
   print thresh
   score = eval_metric(arg1, arg)
   print score
   thresh = thresh + 0.01
