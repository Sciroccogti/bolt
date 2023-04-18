'''
@file lmmse_enc.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-04-12 12:43:28
@modified: 2023-04-12 16:26:53
'''

import vquantizers as vq


class LMMSEEncoder(vq.MultiCodebookEncoder):
    def __init__(
        self, ncodebooks, ncentroids: int = 16,
        quantize_lut: bool = True, nbits: int = 8, upcast_every: int = -1,
        accumulate_how='sum'
    ):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids, quantize_lut=quantize_lut,
                         nbits=nbits, upcast_every=upcast_every, accumulate_how=accumulate_how)

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
