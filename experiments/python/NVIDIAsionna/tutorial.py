'''
@file tutorial.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-09-05 20:40:03
@modified: 2022-09-06 17:32:28
'''
# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np

# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn

# For plotting
import matplotlib.pyplot as plt

# For saving complex Python data structures efficiently
import pickle

# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()

# 256-QAM constellation
NUM_BITS_PER_SYMBOL = 6
# The constellation is set to be trainable
constellation = sn.mapping.Constellation(
    "qam", NUM_BITS_PER_SYMBOL, trainable=True)

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = sn.channel.AWGN()

BATCH_SIZE = 128  # How many examples are processed by Sionna in parallel
EBN0_DB = 17.0  # Eb/N0 in dB

optimizer = tf.keras.optimizers.Adam(1e-4)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
# test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

loss = 1

while loss > 1e-5:
    with tf.GradientTape() as tape:
        # 生成比特序列，通过信道
        bits = binary_source([BATCH_SIZE, 1200])  # Blocklength
        x = mapper(bits)
        # Coderate set to 1 as we do uncoded transmission here
        no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        y = awgn_channel([x, no])
        llr = demapper([y, no])
        # 计算 loss
        loss = bce(bits, llr)
        print(loss)
    # print(tape.watched_variables())
    gradient = tape.gradient(loss, tape.watched_variables())
    optimizer.apply_gradients(zip(gradient, tape.watched_variables()))
    train_loss(loss)
    
