import numpy as np
import math_util
import os
import matmul as mm
import math_util as mu

snr = "20db/"

est = mm.estFactory(X_path="x.npy", W_path="w.npy", Y_path="y.npy", dir= os.path.join("ldpc", snr))

x_test = np.load("LDPC_decoder_NET_testdata/" + snr + "input.npy")
w_test = np.load("LDPC_decoder_NET_testdata/" + snr + "weight.npy")
bias = np.load("LDPC_decoder_NET_testdata/" + snr + "bias.npy")

y_out_matmul = mm.eval_matmul(est, x_test, w_test) # MADDNESS乘法的结果
y_out_last = mu.softmax(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果

np.save("LDPC_decoder_NET_testdata/" + snr + "nomul_matmul_yout_matmul", y_out_matmul)
np.save("LDPC_decoder_NET_testdata/" + snr + "nomul_matmul_yout_last", y_out_last)