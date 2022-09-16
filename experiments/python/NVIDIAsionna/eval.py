'''
@file eval.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-09-08 14:55:02
@modified: 2022-09-15 13:49:40
'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sionna.channel import AWGN
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.mapping import Constellation, Demapper, Mapper
from sionna.utils import (BinarySource, ebnodb2no, expand_to_rank, insert_dims,
                          log10, sim_ber)
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

from .conventional import E2ESystemConventionalTraining, load_weights
from .vars import *


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

# Utility function to load and set weights of a model

if __name__ == "__main__":
    # Range of SNRs over which the systems are evaluated
    ebno_dbs = np.arange(ebno_db_min,  # Min SNR for evaluation
                         ebno_db_max,  # Max SNR for evaluation
                         0.5)  # Step

    # Dictionnary storing the results
    BLER = {}

    # model_baseline = Baseline()
    # _,bler = sim_ber(model_baseline, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=10000)
    # BLER['baseline'] = bler.numpy()

    model_conventional = E2ESystemConventionalTraining(training=False)
    load_weights(model_conventional, model_weights_path_conventional_training)
    _,bler = sim_ber(model_conventional, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=10000)
    BLER['autoencoder-conv'] = bler.numpy()

    with open(results_filename, 'wb') as f:
        pickle.dump((ebno_dbs, BLER), f)

    plt.figure(figsize=(10,8))
    # Baseline - Perfect CSI
    plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')
    # Autoencoder - conventional training
    plt.semilogy(ebno_dbs, BLER['autoencoder-conv'], 'x-.', c=f'C1', label=f'Autoencoder - conventional training')
    # # Autoencoder - RL-based training
    # plt.semilogy(ebno_dbs, BLER['autoencoder-rl'], 'o-.', c=f'C2', label=f'Autoencoder - RL-based training')

    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim((1e-4, 1.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig("result.svg")
