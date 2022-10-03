'''
@file train.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-09-08 14:50:22
@modified: 2022-10-03 14:58:28
'''

import tensorflow as tf

from .conventional import (E2ESystemConventionalTraining, conventional_training,
                          save_weights)
from sionna.mapping import Constellation, Demapper, Mapper
from .vars import *

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0  # Index of the GPU to use
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    # Fix the seed for reprodu  cible trainings
    tf.random.set_seed(1)
    # Instantiate and train the end-to-end system
    model = E2ESystemConventionalTraining(training=True)
    conventional_training(model)
    # Save weights
    save_weights(model, model_weights_path_conventional_training)
    fig = model.constellation.show()
    fig.suptitle("Conventional training")
    fig.savefig("Conv-constellation.svg")
    
    constellation = Constellation(
        "qam", num_bits_per_symbol, trainable=False)
    fig = constellation.show()
    fig.suptitle("Baseline")
    fig.savefig("Base-constellation.svg")
