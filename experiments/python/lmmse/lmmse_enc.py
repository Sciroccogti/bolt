'''
@file lmmse_enc.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-04-12 12:43:28
@modified: 2023-04-19 13:27:23
'''

import vquantizers as vq
import numpy as np
import clusterize


class LMMSEEncoder(vq.PQEncoder):
    def __init__(
        self, ncodebooks, ncentroids: int = 16, elemwise_dist_func=vq.dists_elemwise_sq, preproc='PQ',
        encode_algo: str | None = None, quantize_lut: bool = True, nbits=8, upcast_every: int = -1,
        accumulate_how='sum', SNRs: np.ndarray = np.array([5, 10, 15]),
        **preproc_kwargs
    ):
        self.SNRs = SNRs
        super().__init__(ncodebooks, ncentroids, elemwise_dist_func, preproc, encode_algo,
                         quantize_lut, nbits, upcast_every, accumulate_how, **preproc_kwargs)

    def name(self):
        return "{}_{}".format("LMMSE", super().name())

    def params(self):
        return {
            'ncodebooks': self.ncodebooks,
            'ncentroids': self.ncentroids,
            'quantize_lut': self.quantize_lut,
            'nbits': self.nbits,
            'upcast_every': self.upcast_every,
        }

    def fit(self, X: np.ndarray):
        """
        Learn the centroids and the splits_lists for the encoder.
        :param X: the data to learn from
        """
        # PQ style
        self.subvect_len = int(np.ceil(X.shape[1] / self.ncodebooks))
        X = self._pad_ncols(X)

        if self.encode_algo in ('splits', 'multisplits'):
            assert self.encode_algo == "multisplits"
            self.splits_lists, self.centroids = \
                clusterize.learn_splits_in_subspaces(
                    X, subvect_len=self.subvect_len,
                    nsplits_per_subs=self.code_bits, algo=self.encode_algo)
        else:
            # (K, C, D*)
            self.centroids = vq._learn_centroids(
                X, self.ncentroids, self.ncodebooks, self.subvect_len)

    def genLUT(self) -> np.ndarray:
        """
        Generate the LUT for the encoder.
        :return: the LUT: (C, K, len(SNRs), Npilot, Npilot)
        """
        Ncarrier = 512
        df = 1 / Ncarrier
        Npilot = 56

        luts = np.empty(shape=(len(self.SNRs), self.ncodebooks, self.ncentroids, Npilot, Npilot),
                        dtype=np.complex128)
        for i in range(len(self.SNRs)):
            snr = self.SNRs[i]
            for k in range(self.ncentroids):
                for c in range(self.ncodebooks):
                    RDS = self.centroids[k][c][0]
                    K3 = np.repeat(np.array([[i for i in range(Npilot)]]).T, Npilot, axis=1)
                    K4 = np.repeat(np.array([[i for i in range(Npilot)]]), Npilot, axis=0)
                    rf2 = 1 / (1 + 2j * np.pi * RDS * df * (Ncarrier / Npilot) * (K3 - K4))

                    W = rf2 @ np.linalg.inv(rf2 + (1 / snr) * np.eye(Npilot))
                    luts[i][c][k] = W
        return luts

    def dists_enc(self, X_enc, Q_luts, unquantize=True,
                  offset=None, scale=None):
        # for A(N x D) and B(D x M), C codebooks, K=16 buckets
        X_enc = np.ascontiguousarray(X_enc)  # has shape (N x C) with values in [0, K-1]
        # Q_luts has shape (M x C x K)
        # print(Q_luts.shape)

        if unquantize:
            offset = self.total_lut_offset if offset is None else offset
            scale = self.scale_by if scale is None else scale

        all_dists = np.empty((len(Q_luts), np.shape(Q_luts)[2], np.shape(Q_luts)[3]), dtype=np.complex128)  # (M x N)
        for i, lut in enumerate(Q_luts):
            centroid_dists = lut[X_enc.ravel()]  # shape (NC,)
            dists = centroid_dists
            # dists = centroid_dists.reshape(X_enc.shape)  # shape (N x C)
            # dists = dists.sum(axis=-1)  # shape (N,)

            if self.quantize_lut and unquantize:
                # dists = (dists / self.scale_by) + self.total_lut_offset
                dists = (dists / scale) + offset
            all_dists[i] = dists

        return all_dists
