'''
@file lmmse_filter.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-04-12 12:11:13
@modified: 2023-04-19 13:10:22
'''

from vq_amm import VQMatmul
import vquantizers as vq
# from lmmse. import DPQEncoder, sliceData
from collections.abc import Callable
import numpy as np
from lmmse.lmmse_enc import LMMSEEncoder


class LMMSEFilter(VQMatmul):
    def __init__(
        self, ncodebooks, ncentroids: int = 16, quantize_lut=True, nbits=8, upcast_every=-1,
        SNRs: np.ndarray=np.array([5, 10, 15]),
        # genLUTFunc: Callable[[int, float, np.ndarray | None],
        #                       tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = sliceData,
    ):
        if (quantize_lut or upcast_every != -1):
            raise NotImplementedError("quantize and upcast not yet available for DPQ!")
        self.ncodebooks = ncodebooks
        self.ncentroids = (self._get_ncentroids() if ncentroids is None
                           else ncentroids)
        self.quantize_lut = quantize_lut
        self.nbits = nbits
        self.upcast_every = upcast_every
        self.SNRs = SNRs
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    def _get_ncentroids(self):
        return self.ncentroids

    def _create_encoder(self, ncodebooks) -> LMMSEEncoder:
        return LMMSEEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            quantize_lut=self.quantize_lut, upcast_every=self.upcast_every,
                            SNRs=self.SNRs,
                            **self._get_encoder_kwargs())

    def fit(self, A: np.ndarray):
        """
        get centroids (and splits_lists)
        """
        self.enc.fit(A)
        self.luts = self.enc.genLUT()

    def set_A(self, A):
        self.A_enc = self.enc.encode_X(A)

    def set_B(self, B):
        pass

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids,
                'quantize_lut': self.quantize_lut, 'nbits': self.nbits,
                'upcast_every': self.upcast_every}

    def __call__(self, A, B):
        """
        :param A: X
        :param B: snr
        """
        if self.A_enc is None:
            self.set_A(A)
        nearest_SNR = np.argmin(np.abs(self.SNRs - B))
        return self.enc.dists_enc(self.A_enc, self.luts[nearest_SNR], self.quantize_lut)

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        return super().get_speed_metrics(A, B, fixedA, fixedB)
