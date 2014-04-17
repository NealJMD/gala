import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange

from . import base
from histogram import Manager as HistManager
from moments import central_moments_from_noncentral_sums, ith_root

class Manager(HistManager):
    def __init__(self, radii=[25], z_resolution_factor=1, nbins=4, 
            minval=0.0, maxval=1.0, compute_percentiles=[], 
            compute_histogram=True, nmoments=4,
            normalize=False, *args, **kwargs):
        super(Manager, self).__init__()
        self.radii = radii
        self.minval = minval
        self.maxval = maxval
        self.nbins = nbins
        self.compute_histogram = compute_histogram
        self.nmoments = nmoments
        self.normalize = normalize
        self.z_resolution_factor = z_resolution_factor
        self.lazy_cache = {}
        self.invalidated = {}

        try:
            _ = len(compute_percentiles)
        except TypeError: # single percentile value given
            compute_percentiles = [compute_percentiles]
        self.compute_percentiles = compute_percentiles 

        self.hist_per_hist_len = self.nbins + len(self.compute_percentiles)
        self.hist_per_rad_len = (2*self.hist_per_hist_len+1)
        self.mom_per_node_len = self.nmoments+1
        self.mom_per_rad_len = self.mom_per_node_len*3
        self.per_rad_len = self.hist_per_rad_len + self.mom_per_rad_len
        self.total_len = self.per_rad_len * len(self.radii)
   
    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(
            fm_info['radii'],
            fm_info['nbins'],
            fm_info['minval'], 
            fm_info['maxval'],
            fm_info['compute_percentiles'],
            fm_info['compute_histogram'],
            fm_info['nmoments'],
            fm_info['normalize'])
        return obj
 
    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('histogram')
        json_fm['histogram'] = {
            'radii' : self.radii,
            'minval' : self.minval, 
            'maxval' : self.maxval, 
            'nbins' : self.nbins, 
            'compute_histogram' : self.compute_histogram, 
            'compute_percentiles' : self.compute_percentiles,
            'nmoments' : self.nmoments,
            'normalize' : self.normalize
        } 
        return json_fm

    def lazy_cache_get(self,k1,k2):
        if k1 in self.invalidated: return None
        if k2 in self.invalidated: return None
        try: return self.lazy_cache[k1][k2]
        except KeyError: pass
        try: return self.lazy_cache[k2][k1]
        except KeyError: return None

    def lazy_cache_set(self,k1,k2,val):
        try:  del self.invalidated[k1]
        except KeyError: pass
        try:  del self.invalidated[k2]
        except KeyError: pass
        if k2 < k1: k1,k2 = k2,k1
        if k1 not in self.lazy_cache:
            self.lazy_cache[k1] = {}
        self.lazy_cache[k1][k2] = val

    def lazy_cache_invalidate(self,k):
        self.invalidated[k] = 1

    def create_node_cache(self, g, n):
        return np.array([])
    def create_edge_cache(self, g, n1, n2):
        return np.array([])
    def update_edge_cache(self, g, e1, e2, dst, src):
        pass
    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        pass
    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        pass
    def compute_node_features(self, g, n, cache=None):
        return np.array([])
    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        return np.array([])

    def update_node_cache(self, g, n1, n2, dst, src):
        self.lazy_cache_invalidate(n1)
        self.lazy_cache_invalidate(n2)

    def compute_edge_features(self, g, n1, n2, cache=None):
        cached = self.lazy_cache_get(n1,n2)
        if cached != None: return cached
        out = np.ones(self.total_len,) * -1
        im = g.probabilities
        if im.ndim == 3: im = im[:,:,:,np.newaxis]
        for ii, radius in enumerate(self.radii):
            # find voxel values
            matrices = _voxels_at_intersection(g,n1,n2,radius,self.z_resolution_factor)
            if matrices[0].shape[0] == 0: continue
            if matrices[1].shape[0] == 0: continue
            pos = self.hist_per_rad_len * ii
            v1 = _vals_from_points(im,matrices[0])
            v2 = _vals_from_points(im,matrices[1])

            # compute histogram features
            hist1,ps1 = self.normalized_histogram_from_cache(self.histogram(v1),
                                            self.compute_percentiles)
            hist2,ps2 = self.normalized_histogram_from_cache(self.histogram(v2),
                                            self.compute_percentiles)
            out[pos:pos+self.hist_per_hist_len] = np.concatenate((hist1,ps1), axis=1).ravel()
            pos += self.hist_per_hist_len
            out[pos:pos+self.hist_per_hist_len] = np.concatenate((hist2,ps2), axis=1).ravel()
            pos += self.hist_per_hist_len
            out[pos] = self.JS_divergence(hist1,hist2)
            pos += 1

            # compute moment features
            s1 = _compute_moment_sums(im, matrices[0], self.nmoments)
            s2 = _compute_moment_sums(im, matrices[1], self.nmoments)
            m1 = central_moments_from_noncentral_sums(s1)
            m2 = central_moments_from_noncentral_sums(s2)
            if self.normalize: m1, m2 = map(ith_root, [m1, m2])
            md = abs(m1-m2)
            #print "s1 for %d (exp): %s" % (n1,str(s1))
            #print "m1 for %d (exp): %s" % (n1,str(m1))
            #print "s2 for %d (exp): %s" % (n2,str(s2))
            #print "m2 for %d (exp): %s" % (n2,str(m2))
            f1 = m1.ravel()[0]
            f2 = m2.ravel()[0]
            fd = md.ravel()[0]
            out[pos:(pos+self.mom_per_node_len)] = np.concatenate(([f1], m1[1:].T.ravel()))
            pos += self.mom_per_node_len
            out[pos:(pos+self.mom_per_node_len)] = np.concatenate(([f2], m2[1:].T.ravel()))
            pos += self.mom_per_node_len
            out[pos:(pos+self.mom_per_node_len)] = np.concatenate(([fd], md[1:].T.ravel()))
        self.lazy_cache_set(n1,n2,out)
        return out



@cython.boundscheck(False)
cdef _compute_moment_sums(double[:,:,:,:] ar, long[:,:] points, long nmoments):
    cdef np.ndarray[np.double_t, ndim=2] vals = np.zeros((nmoments+1,ar.shape[3]), dtype=np.double)
    cdef int power,pp,chan
    for pp in prange(points.shape[0],nogil=True,schedule='static'):
        for chan in range(ar.shape[3]):
            for power in range(nmoments+1):
                vals[power,chan] += ar[points[pp,0],points[pp,1],points[pp,2],chan] ** power
    return vals

cdef _voxels_at_intersection(g, long n1, long n2, long radius, long z_res_factor):
    cdef np.ndarray[np.int_t, ndim=1] epicenter
    cdef np.ndarray[np.int_t, ndim=2] edge_points
    edge_points = _point_matrix_from_idx_list(
            np.array(list(g[n1][n2]['boundary'])),np.array(g.watershed.shape))
    epicenter = _centroid(edge_points)
    return [_find_in_radius(g,n1,epicenter,radius,z_res_factor),
            _find_in_radius(g,n2,epicenter,radius,z_res_factor)]

@cython.boundscheck(False)
cdef inline _point_matrix_from_idx_list(long[:] idxs, long[:] shape):
    cdef int dim,ii,stride,rem
    cdef np.ndarray[np.int_t, ndim=2] ps = np.empty((idxs.shape[0],
             shape.shape[0]),dtype=np.integer)
    for ii in prange(idxs.shape[0], nogil=True, schedule='static'):
        rem = idxs[ii]
        stride = 1
        for dim in range(shape.shape[0]):
            stride *= shape[dim]
        for dim in range(shape.shape[0]):
            stride = stride / shape[dim]
            ps[ii, dim] = rem / stride # integer division ftw
            rem = rem - (ps[ii,dim] * stride)
    return ps

@cython.boundscheck(False)
cdef inline _centroid(long[:,:] ps):
    cdef int pp,dim
    cdef np.ndarray[np.int_t, ndim=1] centroid = np.empty(ps.shape[1], dtype=np.integer)
    for dim in range(ps.shape[1]):
        centroid[dim] = 0
        for pp in range(ps.shape[0]):
            centroid[dim] += ps[pp,dim]
        centroid[dim] = centroid[dim] / ps.shape[0]
    return centroid

@cython.boundscheck(False)
cdef _find_in_radius(g, long n, long[:] epicenter, long radius, long z_res_factor):
    cdef int pp,out_count,in_bounds,z_radius
    out_count = 0
    z_radius = radius / z_res_factor
    cdef np.ndarray[np.int_t,ndim=2] extent_points
    #extent_points = np.vstack(np.unravel_index(list(g.extent(n)), g.watershed.shape)).T
    extent_points = _point_matrix_from_idx_list(np.array(list(g.extent(n))),np.array(g.watershed.shape))
    cdef np.ndarray[np.int_t,ndim=2] out_points = np.empty((extent_points.shape[0],
                    extent_points.shape[1]), dtype=np.integer)
    for pp in range(extent_points.shape[0]):
        in_bounds = 1
        if extent_points[pp,0] < (epicenter[0] - z_radius) or \
           extent_points[pp,0] > (epicenter[0] + z_radius):
            continue
        for ii in range(1,extent_points.shape[1]):
            if extent_points[pp,ii] < (epicenter[ii] - radius) or \
               extent_points[pp,ii] > (epicenter[ii] + radius):
                in_bounds = 0; break
        if in_bounds == 0: continue
        for ii in range(extent_points.shape[1]):
            out_points[out_count,ii] = extent_points[pp,ii]
        out_count += 1
    return out_points[:out_count, :]

@cython.boundscheck(False)
cdef _vals_from_points(double[:,:,:,:] im, long[:,:] ps):
    cdef np.ndarray[np.double_t, ndim=2] vals = np.empty((ps.shape[0], im.shape[3]),dtype=np.double)
    cdef int pp,chan
    for chan in range(im.shape[3]):
        for pp in range(ps.shape[0]):
            vals[pp,chan] = im[ps[pp,0],ps[pp,1],ps[pp,2],chan]
    return vals
