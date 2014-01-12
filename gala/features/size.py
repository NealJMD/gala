import numpy as np
import math

from . import base


class Manager(base.Null):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__()

    @classmethod
    def load_dict(cls, fm_info):
        return cls

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('size')
        json_fm['size'] = {}
        return json_fm

    def compute_size_array(self, count):
        return np.array([count, math.log(count)])

    def compute_edge_size(self, g, n1, n2):
        return self.compute_size_array( len(g[n1][n2]['boundary']) )

    def compute_node_size(self, g, n):
        return self.compute_size_array( len(g.node[n]['extent']) )

    def create_node_cache(self, g, n):
        return self.compute_node_size(g, n)

    def create_edge_cache(self, g, n1, n2):
        return self.compute_edge_size(g, n1, n2)

    def update_node_cache(self, g, n1, n2, dst, src):
        dst += src

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst += src

    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        dst += a * self.compute_size_array( len(idxs) )

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        dst += a * self.compute_size_array( len(idxs) )

    def compute_node_features(self, g, n, cache=None):
        if cache is None:  cache = self.create_node_cache(g, n)
        return cache

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None:  cache = self.create_edge_cache(g, n1, n2)
        return cache

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None,
                                                            nthroot=False):
        if cache1 is None: cache1 = self.create_node_cache(g, n1)
        if cache2 is None: cache2 = self.create_node_cache(g, n2)
        return np.absolute(cache1 - cache2)