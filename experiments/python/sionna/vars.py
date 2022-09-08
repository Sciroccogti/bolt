'''
@file vars.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-09-08 14:47:41
@modified: 2022-09-08 17:34:24
'''

import tensorflow as tf

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 5.0
ebno_db_max = 8.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6  # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5  # Coderate for the outer code
n = 1500  # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
# Number of modulated baseband symbols per codeword
num_symbols_per_codeword = n//num_bits_per_symbol
k = int(n*coderate)  # Number of information bits per codeword

###############################################
# Training configuration
###############################################
# Number of training iterations for conventional training
num_training_iterations_conventional = 10000
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(128, tf.int32)  # Training batch size
# Variance of the perturbation used for RL-based training of the transmitter
rl_perturbation_var = 0.01
# Filename to save the autoencoder weights once conventional training is done
model_weights_path_conventional_training = "awgn_autoencoder_weights_conventional_training"
# Filename to save the autoencoder weights once RL-based training is done
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training"

###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results"  # Location to save the results
