import numpy as np
import os
import matmul as mm
import math_util as mu
import scipy.io as io
from amm_methods import *

snrs_ = ["8db/", "8.5db/", "9db/", "9.5db/", "10db/", "10.5db/", "11db/"] #, "12db/", "14db/"]

method = METHOD_MITHRAL

for snr in snrs_:
    print(snr)

    # est0 = mm.estFactory(X_path="x0.npy", W_path="w0.npy", Y_path="y0.npy", dir= os.path.join("../python/data/train/", snr), methods=[method])
    # x_test = np.load("data/" + "input_test%sB.npy" % (snr[:-3]))
    # w_test = np.load("data/" + "weight_hidden1.npy", allow_pickle=True)
    # bias = np.load("data/" + "bias_hidden1.npy")
    # y_out_matmul = mm.eval_matmul(est0, x_test, w_test) # MADDNESS乘法的结果
    # y_out_last = mu.relu(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果
    # x_test = y_out_last

    est1 = mm.estFactory(X_path="x1.npy", W_path="w1.npy", Y_path="y1.npy", dir= os.path.join("../python/data/train/", snr), methods=[method])
    x_test = np.load("data/" + "hidden1_output_test%sB.npy" % (snr[:-3]))
    w_test = np.load("data/" + "weight_hidden1-hidden2.npy", allow_pickle=True)
    bias = np.load("data/" + "bias_hidden1-hidden2.npy")
    y_out_matmul = mm.eval_matmul(est1, x_test, w_test) # MADDNESS乘法的结果
    y_out_last = mu.relu(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果
    x_test = y_out_last

    est2 = mm.estFactory(X_path="x2.npy", W_path="w2.npy", Y_path="y2.npy", dir= os.path.join("../python/data/train/", snr), methods=[method])
    # x_test = np.load("data/" + "hidden2_output_test%sB.npy" % (snr[:-3]))
    w_test = np.load("data/" + "weight_hidden2-hidden3.npy", allow_pickle=True)
    bias = np.load("data/" + "bias_hidden2-hidden3.npy")
    y_out_matmul = mm.eval_matmul(est2, x_test, w_test) # MADDNESS乘法的结果
    y_out_last = mu.relu(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果
    x_test = y_out_last

    est3 = mm.estFactory(X_path="x.npy", W_path="w.npy", Y_path="y.npy", dir= os.path.join("ldpc", snr), methods=[method])
    # x_test = np.load("data/" + "hidden3_output_test%sB.npy" % (snr[:-3]))
    w_test = np.load("data/" + "weight_hidden3-out.npy", allow_pickle=True)
    bias = np.load("data/" + "bias_hidden3-out.npy")
    y_out_matmul = mm.eval_matmul(est3, x_test, w_test) # MADDNESS乘法的结果
    y_out_last = mu.softmax(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果

    # np.save("LDPC_decoder_NET_testdata/" + snr + "nomul_matmul_yout_matmul", y_out_matmul)
    # np.save("LDPC_decoder_NET_testdata/" + snr + "nomul_matmul_yout_last", y_out_last)

    io.savemat("data/" + method + "nc16@1_output1_threshold0.3_Tr15_SNR%s_ot.mat" % (snr[:-3]), {"NN_output_buffer": y_out_last})