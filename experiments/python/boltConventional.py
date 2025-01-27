'''
@file boltConventional.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief
@date 2022-09-11 14:46:11
@modified: 2022-10-16 13:48:06
'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sionna.utils import (BinarySource, ebnodb2no, expand_to_rank, insert_dims,
                          log10, sim_ber)
from tensorflow.keras import Model

import amm_methods
from matmul import estFactory, eval_matmul
from matmul_datasets import MatmulTask
from NVIDIAsionna.conventional import (E2ESystemConventionalTraining,
                                       load_weights)
from NVIDIAsionna.eval import Baseline
from NVIDIAsionna.vars import *


class E2EBoltConventionalTraining(Model):
    """
    该 class 构造时使用了 E2ESystemConventionalTraining 来初始化整个系统，
    仅在 call 时替换了 demapper，因此需要在构造时将权重读入到 E2ESystemConventionalTraining
    模型中，而不是构造后将权重读入到自身
    """

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
            [training_batch_size, k])  # (training_batch_size, 750)
        c = model_conventional._encoder(b)  # (training_batch_size, 1500)
        x = model_conventional._mapper(c)  # (training_batch_size, 250)
        y = model_conventional._channel([x, no])  # (training_batch_size, 250)

        llr = model_conventional._demapper([y, no])  # (training_batch_size, 250, 6)
        llr = tf.reshape(llr, [training_batch_size, n])  # (training_batch_size, 1500)

        no_db = log10(no)
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword])
        z = tf.stack([tf.math.real(y), tf.math.imag(y), no_db],
                     axis=2)  # (training_batch_size, 250, 3)
        layer1out = z @ self.weights_[1]  # 第一层乘法输出

        # self.est1 = estFactory(methods=[amm_methods.METHOD_MITHRAL], tasks=[
        #     MatmulTask(z[0].numpy(), layer1out[0].numpy(), None, None, self.weights_[1], name="Layer1")])

        layer2in = tf.nn.relu(layer1out + self.weights_[2])
        layer2out = layer2in @ self.weights_[3]
        # self.est2 = estFactory(methods=[method], tasks=[
        #     MatmulTask(layer2in[0].numpy(), layer2out[0].numpy(), None, None, self.weights_[3], name="Layer2")])

        layer3in = tf.nn.relu(layer2out + self.weights_[4])
        layer3out = layer3in @ self.weights_[5]
         # 把三维矩阵变成二维矩阵在第一个维度上的堆叠
        layer3inReshaped = np.reshape(layer3in.numpy(), (-1, layer3in.numpy().shape[2]))
        layer3outReshaped = np.reshape(layer3out.numpy(), (-1, layer3out.numpy().shape[2]))
        self.est3 = estFactory(methods=[method], tasks=[
            MatmulTask(layer3inReshaped, layer3outReshaped, None, None, self.weights_[5], name="Layer3")], ncentroids=4096)

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

        layer2out = layer2in @ self.weights_[3]
        # self.est2 = estFactory(methods=[amm_methods.METHOD_MITHRAL], tasks=[
        #     MatmulTask(layer2in[i].numpy(), layer2out.numpy(), None, None, self.weights_[3], name="Layer2")])
        # layer2out = tf.convert_to_tensor(eval_matmul(
        #     self.est2, layer2in[i], self.weights_[3]))

        layer3in = tf.nn.relu(layer2out + self.weights_[4])
         # 把三维矩阵变成二维矩阵在第一个维度上的堆叠
        layer3inReshaped = np.reshape(layer3in.numpy(), (-1, layer3in.numpy().shape[2]))
        # layer3out = layer3in @ self.weights_[5]
        # self.est3 = estFactory(methods=[amm_methods.METHOD_PQ], tasks=[
        #     MatmulTask(layer3in.numpy(), layer3out.numpy(), None, None, self.weights_[5], name="Layer3")], ncentroids=64)
        layer3out = tf.convert_to_tensor(
            eval_matmul(self.est3, layer3inReshaped, self.weights_[5]))
        llr = layer3out + self.weights_[6]
        
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

    model_bolt = E2EBoltConventionalTraining(training=False, model_weights_path_conventional_training=
                                             model_weights_path_conventional_training, method=amm_methods.METHOD_MITHRAL)
    _, bler = sim_ber(model_bolt, ebno_dbs, batch_size=1024,
                      num_target_block_errors=20, max_mc_iter=2000)
    BLER['autoencoder-maddness'] = bler.numpy()

    model_bolt = E2EBoltConventionalTraining(training=False, model_weights_path_conventional_training=
                                             model_weights_path_conventional_training, method=amm_methods.METHOD_PQ)
    _, bler = sim_ber(model_bolt, ebno_dbs, batch_size=1024,
                      num_target_block_errors=20, max_mc_iter=2000)
    BLER['autoencoder-PQ'] = bler.numpy()

    model_exact = E2EBoltConventionalTraining(training=False, model_weights_path_conventional_training=
                                             model_weights_path_conventional_training, method=amm_methods.METHOD_EXACT)
    _, bler = sim_ber(model_exact, ebno_dbs, batch_size=1024,
                      num_target_block_errors=20, max_mc_iter=2000)
    BLER['autoencoder-exact'] = bler.numpy()

    model_baseline = Baseline()
    _, bler = sim_ber(model_baseline, ebno_dbs, batch_size=1024,
                      num_target_block_errors=20, max_mc_iter=2000)
    BLER['baseline'] = bler.numpy()

    model_conventional = E2ESystemConventionalTraining(training=False)
    load_weights(model_conventional, model_weights_path_conventional_training)
    _, bler = sim_ber(model_conventional, ebno_dbs, batch_size=1024,
                      num_target_block_errors=20, max_mc_iter=2000)
    BLER['autoencoder-conv'] = bler.numpy()

    with open(results_filename, 'wb') as f:
        pickle.dump((ebno_dbs, BLER), f)

    plt.figure(figsize=(6, 4.5))
    # Baseline - Perfect CSI
    plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')
    plt.semilogy(ebno_dbs, BLER['autoencoder-conv'], 'x-.',
                 c=f'C1', label=f'Autoencoder - conventional origin')
    plt.semilogy(ebno_dbs, BLER['autoencoder-exact'], 's-.',
                 c=f'C2', label=f'Autoencoder - conventional exact')
    plt.semilogy(ebno_dbs, BLER['autoencoder-maddness'], '^-.',
                 c=f'C3', label=f'Autoencoder - conventional maddness')
    plt.semilogy(ebno_dbs, BLER['autoencoder-PQ'], '^-.',
                 c=f'C4', label=f'Autoencoder - conventional PQ')
    # Autoencoder - conventional training
    # # Autoencoder - RL-based training
    # plt.semilogy(ebno_dbs, BLER['autoencoder-rl'], 'o-.', c=f'C2', label=f'Autoencoder - RL-based training')

    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim(top=1)
    plt.legend()
    plt.tight_layout()
    # plt.title("自编码器的误块率")
    plt.savefig("result.svg")
