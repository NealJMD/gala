import libtiff
from libtiff import TIFF
import numpy as np
import h5py
import sys
import gala
from gala import morpho
from scipy import ndimage as nd

def relabel_connected(im, connectivity=1):
    """Ensure all labels in `im` are connected.

    Parameters
    ----------
    im : array of int
        The input label image.
    connectivity : int in {1, ..., `im.ndim`}, optional
        The connectivity used to determine if two voxels are neighbors.

    Returns
    -------
    im_out : array of int
        The relabeled image.

    Examples
    --------
    >>> image = np.array([[1, 1, 2],
                          [2, 1, 1]])
    >>> im_out = relabel_connected(image)
    >>> im_out
    array([[1, 1, 2],
           [3, 1, 1]])
    """
    im_out = np.zeros_like(im)
    contiguous_segments = np.empty_like(im)
    structure = morpho.generate_binary_structure(im.ndim, connectivity)
    curr_label = 0
    labels = np.unique(im)
    if labels[0] == 0:
        labels = labels[1:]
    for label in labels:
        segment = (im == label)
        n_segments = nd.label(segment, structure,
                              output=contiguous_segments)
        seg = segment.nonzero()
        contiguous_segments[seg] += curr_label
        im_out[seg] += contiguous_segments[seg]
        curr_label += n_segments
    return im_out



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

#npstack_left = np.array(stack_left) / 255.0
npstack_left = np.array(stack_left)
print npstack_left.shape
#npstack_right = np.array(stack_right) / 255.0
#print npstack_right.shape
print morpho
npstack_left = relabel_connected(npstack_left)


f = h5py.File(arg2, 'w')
#grp = f.create_group("volume")
#subgrp = grp.create_group("predictions")
#subgrp.create_dataset("stack", npstack_left)
#subgrp['stack'] = npstack_left
f['stack'] = npstack_left
f.close()

#if len(sys.argv) == 4:
#    f2 = h5py.File(arg3, 'w')
#    grp = f2.create_group("volume")
    #subgrp = grp.create_group("predictions")
    #subgrp.create_dataset("stack", npstack_right)
    #subgrp['stack'] = npstack_right
#    grp['predictions'] = npstack_right
#    f2.close() 



