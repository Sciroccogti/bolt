'''
@file boltConventional.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief
@date 2022-09-11 14:46:11
@modified: 2022-09-16 15:44:10
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

import amm_methods
from math_util import relu
from matmul import estFactory, eval_matmul
from matmul_datasets import MatmulTask
from NVIDIAsionna.conventional import (E2ESystemConventionalTraining,
                                       load_weights)
from NVIDIAsionna.demapper import NeuralDemapper
from NVIDIAsionna.eval import Baseline
from NVIDIAsionna.vars import *


class E2EBoltConventionalTraining(Model):

    def __init__(self, training: bool, model_weights_path_conventional_training: str, method: str):
        super().__init__()

        model_conventional = E2ESystemConventionalTraining(training=False)
        load_weights(model_conventional,
                     model_weights_path_conventional_training)
        # read weights in for bolt
        with open(model_weights_path_conventional_training, 'rb') as f:
            self.weights_ = pickle.load(f)

        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(
            shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        b = model_conventional._binary_source(
            [training_batch_size, k])  # (128, 750)
        c = model_conventional._encoder(b)  # (128, 1500)
        x = model_conventional._mapper(c)  # (128, 250)
        y = model_conventional._channel([x, no])  # (128, 250)

        llr = model_conventional._demapper([y, no])  # (128, 250, 6)
        llr = tf.reshape(llr, [128, n])  # (128, 1500)

        no_db = log10(no)
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword])
        z = tf.stack([tf.math.real(y), tf.math.imag(y), no_db],
                     axis=2)  # (128, 250, 3)
        layer1out = z @ self.weights_[1]  # 第一层乘法输出

        # self.est1 = estFactory(methods=[amm_methods.METHOD_MITHRAL], tasks=[
        #     MatmulTask(z[0].numpy(), layer1out[0].numpy(), None, None, self.weights_[1], name="Layer1")])

        layer2in = tf.nn.relu(layer1out + self.weights_[2])
        layer2out = layer2in @ self.weights_[3]
        self.est2 = estFactory(methods=[method], tasks=[
            MatmulTask(layer2in[0].numpy(), layer2out[0].numpy(), None, None, self.weights_[3], name="Layer2")])

        layer3in = tf.nn.relu(layer2out + self.weights_[4])
        layer3out = layer3in @ self.weights_[5]
        self.est3 = estFactory(methods=[method], tasks=[
            MatmulTask(layer3in[0].numpy(), layer3out[0].numpy(), None, None, self.weights_[5], name="Layer3")])

        self._training = training

        ################
        # Transmitter
        ################
        self._binary_source = model_conventional._binary_source
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._encoder = model_conventional._encoder
        # Trainable constellation
        self.constellation = model_conventional.constellation
        self._mapper = model_conventional._mapper

        ################
        # Channel
        ################
        self._channel = model_conventional._channel

        ################
        # Receiver
        ################
        # We use the previously defined neural network for demapping
        self._demapper = model_conventional._demapper
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._decoder = model_conventional._decoder

        #################
        # Loss function
        #################
        if self._training:
            self._bce = model_conventional._bce

    # @tf.function(jit_compile=True) # TODO
    def call(self, batch_size, ebno_db):
        assert self._training == False, "This model cannot be trained"

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        # Transmitter
        ################
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, n])
        else:
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
        no_db = log10(no)
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword])
        z = tf.stack([tf.math.real(y), tf.math.imag(y), no_db], axis=2)
        layer1out = z @ self.weights_[1]  # 第一层乘法输出
        layer2in = tf.nn.relu(layer1out + self.weights_[2])
        llrs_ = []

        for i in range(batch_size):
            # layer2out = layer2in[i] @ self.weights_[3]
            layer2out = tf.convert_to_tensor(eval_matmul(
                self.est2, layer2in[i], self.weights_[3]))
            layer3in = tf.nn.relu(layer2out + self.weights_[4])
            layer3out = tf.convert_to_tensor(
                eval_matmul(self.est3, layer3in, self.weights_[5]))
            llrs_.append(layer3out + self.weights_[6])
        llr = tf.stack(llrs_)
        
        llr = tf.reshape(llr, [batch_size, n])
        # If training, outer decoding is not performed and the BCE is returned
        if self._training:
            loss = self._bce(c, llr)
            return loss
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation


def conventional_train_bolt(model_weights_path_conventional_training):
    # init model with trained weights
    model = E2EBoltConventionalTraining(
        training=False, model_weights_path_conventional_training=model_weights_path_conventional_training)


if __name__ == "__main__":
    # Range of SNRs over which the systems are evaluated
    ebno_dbs = np.arange(ebno_db_min,  # Min SNR for evaluation
                         ebno_db_max,  # Max SNR for evaluation
                         0.5)  # Step

    # Dictionnary storing the results
    BLER = {}

    # model_exact = E2EBoltConventionalTraining(training=False, model_weights_path_conventional_training="./NVIDIAsionna/" +
    #                                          model_weights_path_conventional_training, method=amm_methods.METHOD_EXACT)
    # _, bler = sim_ber(model_exact, ebno_dbs, batch_size=128,
    #                   num_target_block_errors=10, max_mc_iter=1000)
    # BLER['autoencoder-exact'] = bler.numpy()

    model_bolt = E2EBoltConventionalTraining(training=False, model_weights_path_conventional_training="./NVIDIAsionna/" +
                                             model_weights_path_conventional_training, method=amm_methods.METHOD_MITHRAL)
    _, bler = sim_ber(model_bolt, ebno_dbs, batch_size=128,
                      num_target_block_errors=20, max_mc_iter=1000)
    BLER['autoencoder-maddness'] = bler.numpy()

    # model_bolt = E2EBoltConventionalTraining(training=False, model_weights_path_conventional_training="./NVIDIAsionna/" +
    #                                          model_weights_path_conventional_training, method=amm_methods.METHOD_PQ)
    # _, bler = sim_ber(model_bolt, ebno_dbs, batch_size=128,
    #                   num_target_block_errors=20, max_mc_iter=1000)
    # BLER['autoencoder-PQ'] = bler.numpy()

    model_baseline = Baseline()
    _, bler = sim_ber(model_baseline, ebno_dbs, batch_size=128,
                      num_target_block_errors=20, max_mc_iter=1000)
    BLER['baseline'] = bler.numpy()

    model_conventional = E2ESystemConventionalTraining(training=False)
    load_weights(model_conventional, "./NVIDIAsionna/" +
                 model_weights_path_conventional_training)
    _, bler = sim_ber(model_conventional, ebno_dbs, batch_size=128,
                      num_target_block_errors=20, max_mc_iter=1000)
    BLER['autoencoder-conv'] = bler.numpy()

    with open(results_filename, 'wb') as f:
        pickle.dump((ebno_dbs, BLER), f)

    plt.figure(figsize=(10, 8))
    # Baseline - Perfect CSI
    plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')
    plt.semilogy(ebno_dbs, BLER['autoencoder-conv'], 'x-.',
                 c=f'C1', label=f'Autoencoder - conventional origin')
    # plt.semilogy(ebno_dbs, BLER['autoencoder-exact'], 's-.',
    #              c=f'C2', label=f'Autoencoder - conventional exact')
    plt.semilogy(ebno_dbs, BLER['autoencoder-maddness'], '^-.',
                 c=f'C3', label=f'Autoencoder - conventional maddness')
    # plt.semilogy(ebno_dbs, BLER['autoencoder-PQ'], '^-.',
    #              c=f'C4', label=f'Autoencoder - conventional PQ')
    # Autoencoder - conventional training
    # # Autoencoder - RL-based training
    # plt.semilogy(ebno_dbs, BLER['autoencoder-rl'], 'o-.', c=f'C2', label=f'Autoencoder - RL-based training')

    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim((1e-4, 1.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig("result.svg")
