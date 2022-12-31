'''
@file dpq_encoder.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-29 13:27:52
@modified: 2022-12-31 23:56:06
'''

import clusterize
import numpy as np
import vquantizers as vq
from dpq.dpq_nn import DPQNetwork
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class DPQEncoder(vq.MultiCodebookEncoder):
    def __init__(self, ncodebooks, ncentroids=256, quantize_lut=True, nbits=8, upcast_every=-1, accumulate_how='sum'):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids, quantize_lut=quantize_lut,
                         nbits=nbits, upcast_every=upcast_every, accumulate_how=accumulate_how)

    def name(self):
        return "{}_{}".format('DPQ', super().name())

    def params(self):
        return {'ncodebooks': self.ncodebooks,
                'ncentroids': self.ncentroids,
                'quantize_lut': self.quantize_lut,
                'nbits': self.nbits,
                'upcast_every': self.upcast_every,
                }

    def fit(self, X, Q):
        '''
        use DPQ to learn centroids

        :param X: left matrix, just used to train
        :param Q: right matrix, should be the same as the online Q
        '''
        # TODO: query_metric, shared_centroids, tau

        # Mithral style
        # self.splits_lists, self.centroids = learn_mithral(
        #     X, self.ncodebooks, ncentroids=self.ncentroids, lut_work_const=self.lut_work_const,
        #     nonzeros_heuristic=self.nonzeros_heuristic)

        # PQ style
        self.subvect_len = int(np.ceil(X.shape[1] / self.ncodebooks))
        X = vq.ensure_num_cols_multiple_of(X, self.ncodebooks)
        # encode_algo: None
        self.centroids = vq._learn_centroids(
            X, self.ncentroids, self.ncodebooks, self.subvect_len)  # (K, C, subvect_len)
        # encode_algo: multisplit
        # self.encode_algo = "multisplits"
        # self.splits_lists, self.centroids = clusterize.learn_splits_in_subspaces(
        #     X, subvect_len=self.subvect_len, nsplits_per_subs=self.code_bits,
        #     algo=self.encode_algo)

        print(f"Using {device} device")
        model = DPQNetwork(
            ncentroids=self.ncentroids,
            ncodebooks=self.ncodebooks,
            subvect_len=self.subvect_len,
            centroids=torch.from_numpy(self.centroids.transpose((1, 0, 2))).to(device),
        ).to(device)
        X = torch.from_numpy(X.reshape((-1, self.ncodebooks, self.subvect_len))).to(device)
        model.forward(X)

    def encode_Q(self, Q):
        '''
        generate luts using centroids

        :param Q:
        '''
        # PQ style
        Q = np.atleast_2d(Q)
        Q = vq.ensure_num_cols_multiple_of(Q, self.ncodebooks)

        luts = np.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
        # print("Q shape: ", Q.shape)
        for i, q in enumerate(Q):
            lut = vq._fit_pq_lut(q, centroids=self.centroids,
                                 elemwise_dist_func=vq.dists_elemwise_dot)
            luts[i] = lut.T
        return luts

    def encode_X(self, X):
        '''
        encode left matrix online, currently all from PQEncoder

        :param X: online left matrix
        '''
        # PQ style
        X = vq.ensure_num_cols_multiple_of(X, self.ncodebooks)
        # encode_algo: None
        idxs = vq.pq._encode_X_pq(X, codebooks=self.centroids)
        # encode_algo: multisplit
        # idxs = clusterize.encode_using_splits(X, self.subvect_len, self.splits_lists, "multi")

        # self.offsets is set in MultiCodebookEncoder.__init__
        return idxs + self.offsets


if __name__ == "__main__":
    DPQEncoder().fit()
