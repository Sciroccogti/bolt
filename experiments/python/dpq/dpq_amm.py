'''
@file dpq_amm.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-29 14:05:49
@modified: 2023-01-26 13:12:32
'''

from vq_amm import VQMatmul
from dpq.dpq_encoder import DPQEncoder, sliceData
from collections.abc import Callable
import numpy as np
from torch import Tensor


class DPQMatmul(VQMatmul):
    def __init__(
        self, ncodebooks, ncentroids: int = 16, quantize_lut=True, nbits=8, upcast_every=-1,
        genDataFunc: Callable[[int, float, np.ndarray | None],
                              tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = sliceData,
    ):
        if (quantize_lut or upcast_every != -1):
            raise NotImplementedError("quantize and upcast not yet available for DPQ!")
        self.genDataFunc = genDataFunc
        self.ncodebooks = ncodebooks
        self.ncentroids = (self._get_ncentroids() if ncentroids is None
                           else ncentroids)
        self.quantize_lut = quantize_lut
        self.nbits = nbits
        self.upcast_every = upcast_every
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    def _get_ncentroids(self):
        return self.ncentroids

    def _create_encoder(self, ncodebooks) -> DPQEncoder:
        assert ncodebooks == self.ncodebooks,\
            "ncodebooks for _create_encoder does not match the one for DPQMatmul"
        return DPQEncoder(
            ncodebooks=self.ncodebooks,
            ncentroids=self.ncentroids,
            quantize_lut=self.quantize_lut,
            nbits=self.nbits,
            upcast_every=self.upcast_every,
            genDataFunc=self.genDataFunc
        )

    def set_B(self, B):
        return super().set_B(B)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids,
                'quantize_lut': self.quantize_lut, 'nbits': self.nbits,
                'upcast_every': self.upcast_every}

    def __call__(self, A, B):
        # return super().__call__(A, B)
        return self.enc.product(A)

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        return super().get_speed_metrics(A, B, fixedA, fixedB)
