# %% [markdown]
# decoder transformer层的linear1层（dtl1）替换为近似矩阵乘法, 32\*32\*64, 64\*128, 32\*32\*128

# %%
import numpy as np
import os
import sys
dir_now = os.getcwd()
sys.path.append(dir_now)
sys.path.append(os.path.join(dir_now, '../'))
import matmul as mm
import math_util as mu
# import scipy.io as io
from amm_methods import *
import socket # Obtain the current host name, which can be used to select different data directories and result saving directories

# %%
# method = METHOD_MITHRAL
method = METHOD_PQ
# method = METHOD_EXACT
# method = METHOD_SCALAR_QUANTIZE
nbits = 4 # 量化比特数

# %%
linear_name = 'dtl1'
feedback_bits = 256
# ncodebooks = 32 # max:64
ncentroids = 64
if method == METHOD_MITHRAL:
    ncentroids = 16
train_sam_num = 3000 # 训练集样本数
if method == METHOD_EXACT:
    ncodebooks = 0
    ncentroids = 0
test_sam_num = 1000 # 测试集样本数
batch_size = 32

for ncodebooks in [16,32,64]:
    # %%
    host_name = socket.gethostname()
    if host_name == 'DESKTOP-PLRL7TK':
        dir_train = 'E:\\hdr\\研一\\华为-深度学习\\intermediate\\intermediate8dbfc1'
        dir_result = ''
    elif host_name == 'DESKTOP-6FOH47P':
        dir_train = 'F:\\Projects\\python\\PQ\\intermediate8dbfc1'
        dir_result = 'F:\\Projects\\python\\PQ\\res'
        data_to_fcpath_train= ''
        featurepath_train= ''
        data_to_fcpath_test = ''
        featurepath_test = ''
    elif host_name == 'jm-System-Product-Name':
        dir_train = os.path.join('/data/hdr/transformer_data/joined', 'train', 'f'+str(feedback_bits))
        dir_test = os.path.join('/data/hdr/transformer_data/joined', 'test', 'f'+str(feedback_bits))
        dir_result = '/data/hdr/pq/res'
        data_to_fcpath_train= 'dx_linear1in_train_f%i_sam%i.npy' % (feedback_bits, train_sam_num)
        y_train = 'dx_linear1_y_train_f%i_sam%i.npy' % (feedback_bits, train_sam_num)
        featurepath_train= 'dx_linear1out_train_f%i_sam%i.npy' % (feedback_bits, train_sam_num)
        data_to_fcpath_test = 'dx_linear1in_test_f%i_sam%i.npy' % (feedback_bits, test_sam_num)
        featurepath_test = 'dx_linear1out_test_f%i_sam%i.npy' % (feedback_bits, test_sam_num)
        # y_test = 'y_test.npy'
    else:
        raise NameError("You are running the script in a new computer, please define dirs")


    weightpath = 'dx_linear1_w_f256.npy'
    biaspath = 'dx_linear1_b_f256.npy'
    dir_result = os.path.join(dir_result, method, "f%i" % feedback_bits, linear_name)
    try:
        os.mkdir(dir_result)
    except FileExistsError:
        pass 


    # %%
    est3 = mm.estFactory(X_path=data_to_fcpath_train, W_path=weightpath, Y_path=y_train, dir= dir_train, ncodebooks=ncodebooks, ncentroids=ncentroids, methods=[method], nbits=nbits)


    # %%
    # type(est3)

    # %%
    x_test = np.load(dir_test+'/'+data_to_fcpath_test)
    w_test = np.load(dir_train+'/'+weightpath)
    bias = np.load(dir_train+'/'+biaspath)
    # print(type(est3))
    y_out_matmul = mm.eval_matmul(est3, x_test, w_test) # MADDNESS乘法的结果
    # y_out_last = mu.softmax(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果
    y_out_last = y_out_matmul + bias.T # MADDNESS替换后当前层输出，即+bias并不需要激活函数后的结果

    # %%
    print(y_out_last)
    print("y_out_last.shape: ", y_out_last.shape)
    y_out_last_re = y_out_last.reshape(test_sam_num, batch_size, -1, y_out_last.shape[-1]) #AMM字典模式需要复原y大小
    print("y_out_last_re.shape: ", y_out_last_re.shape)
    # np.save("LDPC_decoder_NET_testdata/" + snr + "nomul_matmul_yout_matmul", y_out_matmul)
    # np.save(dir_result+'/'+method+'fc1_fb256_cb%i_ct%i.npy' % (ncodebooks, ncentroids), y_out_matmul)
    if method == METHOD_EXACT:
        train_sam_num = 0 # 训练集样本数
    if method == METHOD_SCALAR_QUANTIZE:
        np.save(os.path.join(dir_result, '%s%s_trsam%i_tesam%i_fb%i_nbits%i.npy' % (method, linear_name, train_sam_num, test_sam_num, feedback_bits, nbits)), y_out_last_re.astype(np.float32))
    else:
        np.save(os.path.join(dir_result, '%s%s_trsam%i_tesam%i_fb%i_cb%i_ct%i.npy' % (method, linear_name, train_sam_num, test_sam_num, feedback_bits, ncodebooks, ncentroids)), y_out_last_re)

    # %%



