'''
@file dpq_amm.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-29 14:05:49
@modified: 2023-01-02 13:55:52
'''

from vq_amm import VQMatmul
from dpq.dpq_encoder import DPQEncoder


class DPQMatmul(VQMatmul):
    def __init__(self, ncodebooks, ncentroids: int = 16, quantize_lut=True, nbits=8, upcast_every=-1):
        if (quantize_lut or upcast_every != -1):
            raise NotImplementedError("quantize and upcast not yet available for DPQ!")
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids,
                         quantize_lut=quantize_lut, nbits=nbits, upcast_every=upcast_every)
        self.ncentroids = ncentroids

    def _get_ncentroids(self):
        return self.ncentroids

    def _create_encoder(self, ncodebooks):
        assert ncodebooks == self.ncodebooks,\
            "ncodebooks for _create_encoder does not match the one for DPQMatmul"
        return DPQEncoder(
            ncodebooks=self.ncodebooks,
            ncentroids=self.ncentroids,
            quantize_lut=self.quantize_lut,
            nbits=self.nbits,
            upcast_every=self.upcast_every,
        )

    def set_B(self, B):
        return super().set_B(B)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids,
                'quantize_lut': self.quantize_lut, 'nbits': self.nbits,
                'upcast_every': self.upcast_every}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        return super().get_speed_metrics(A, B, fixedA, fixedB)
