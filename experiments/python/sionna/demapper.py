'''
@file autoencoder.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-09-08 14:36:48
@modified: 2022-09-08 15:16:48
'''
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims
from sionna.channel import AWGN
import sionna
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from vars import *


class NeuralDemapper(Layer):

    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        # The feature correspond to the LLRs for every bits carried by a symbol
        self._dense_3 = Dense(num_bits_per_symbol, None)

    def call(self, inputs):
        y, no = inputs

        # Using log10 scale helps with the performance
        no_db = log10(no)

        # Stacking the real and imaginary components of the complex received samples
        # and the noise variance
        # [batch size, num_symbols_per_codeword]
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword])
        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y),
                      no_db], axis=2)  # [batch size, num_symbols_per_codeword, 3]
        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        # [batch size, num_symbols_per_codeword, num_bits_per_symbol]
        llr = self._dense_3(llr)

        return llr


class Baseline(Model):

    def __init__(self):
        super().__init__()

        ################
        # Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n)
        constellation = Constellation(
            "qam", num_bits_per_symbol, trainable=False)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)

        ################
        # Channel
        ################
        self._channel = AWGN()

        ################
        # Receiver
        ################
        self._demapper = Demapper("app", constellation=constellation)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db, perturbation_variance=tf.constant(0.0, tf.float32)):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        # Transmitter
        ################
        b = self._binary_source([batch_size, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)  # x [batch size, num_symbols_per_codeword]

        ################
        # Channel
        ################
        y = self._channel([x, no])  # [batch size, num_symbols_per_codeword]

        ################
        # Receiver
        ################
        llr = self._demapper([y, no])
        # Outer decoding
        b_hat = self._decoder(llr)
        return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation
