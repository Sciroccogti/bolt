'''
@file autoencoder.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-09-08 14:36:48
@modified: 2022-09-14 21:29:19
'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

from sionna.channel import AWGN
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.mapping import Constellation, Demapper, Mapper
from sionna.utils import (BinarySource, ebnodb2no, expand_to_rank, insert_dims,
                          log10)
from .vars import *


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


class BoltDemapper(Layer):

    def __init__(self, model, model_weights_path: str) -> None:
        """
        @param model:
        @param model_weights_path: 
        """
        super().__init__()
        self._ests_ = []  # store est
        self._weights_ = []
        self._biases_ = []

        # 读取权重
        with open(model_weights_path, 'rb') as f:
            weights = pickle.load(f)
            assert len(weights) % 2 == 0  # 必须是一个 weight 一个 bias 对应
            for i in range(0, len(weights), 2):
                self._weights_.append(weights[i])
                self._biases_.append(weights[i + 1])
            print(weights)
        # 在这里读取权重，训练乘法器

    def call(self, inputs):
        y, no = inputs


if __name__ == "__main__":

    model_conventional = E2ESystemConventionalTraining(training=False)
    print(model_conventional)
    demapper = BoltDemapper(
        None, model_weights_path=model_weights_path_conventional_training)
