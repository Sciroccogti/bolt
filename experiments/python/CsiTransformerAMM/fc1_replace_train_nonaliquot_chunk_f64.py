#!/bin/env/python
# %%
import numpy as np
import os
import sys
dir_now = os.getcwd()
sys.path.append(dir_now)
sys.path.append(os.path.join(dir_now, '../'))
import matmul as mm
import math_util as mu
import scipy.io as io
from amm_methods import *
import socket # Obtain the current host name, which can be used to select different data directories and result saving directories

# %%
method = METHOD_MITHRAL
# method = METHOD_PQ
# method = METHOD_EXACT
# method = METHOD_SCALAR_QUANTIZE

# %%
feedback_bits = 64

# ncodebooks=128
ncentroids=256
if method == METHOD_MITHRAL:
    ncentroids=16
train_sam_num = 3000 # 训练集样本数
split_Br_frac = [1/2,1/2] # 切分后B矩阵每分块的行数依次占总行数比例
split_Bc_frac = [1/2,1/2] # 切分后B矩阵每分块的行数依次占总行数比例
if sum(split_Br_frac)!=1 or sum(split_Bc_frac)!=1:
    raise ValueError("split_Br_frac或split_Bc_frac的和必须为1")
n_split_Br = len(split_Br_frac) # A的列、B的行分割成n_split_Br份
n_split_Bc = len(split_Bc_frac) # B的列分割成n_split_Bc份


# %%
host_name = socket.gethostname()
if host_name == 'DESKTOP-PLRL7TK':
    dir_train = 'E:\\hdr\\研一\\华为-深度学习\\intermediate\\intermediate8dbfc1'
    dir_result = ''
elif host_name == 'DESKTOP-6FOH47P':
    dir_train = 'F:\\Projects\\python\\PQ\\intermediate8dbfc1'
    dir_result = 'F:\\Projects\\python\\PQ\\res'
    data_to_fcpath_train= 'data_to_fc_e39_7999.npy'
    featurepath_train= 'feature_e39_7999.npy'
    data_to_fcpath_test = 'data_to_fc_e39_7999.npy'
    featurepath_test = 'feature_e39_7999.npy'
elif host_name == 'jm-System-Product-Name':
    dir_train = os.path.join('/data/hdr/transformer_data/joined', 'train', 'f'+str(feedback_bits))
    dir_test = os.path.join('/data/hdr/transformer_data/joined', 'test', 'f'+str(feedback_bits))
    dir_result = '/data/hdr/pq/res'
    data_to_fcpath_train= 'data_to_fc_train_f%i_sam%i.npy' % (feedback_bits, train_sam_num)
    y_train = 'y_train_f%i_sam%i.npy' % (feedback_bits, train_sam_num)
    featurepath_train= 'feature_train_f%i_sam%i.npy' % (feedback_bits, train_sam_num)
    data_to_fcpath_test = 'data_to_fc_test_f%i.npy' % feedback_bits
    featurepath_test = 'feature_test_f%i.npy' % feedback_bits
    y_test = 'y_test_f%i.npy' % feedback_bits
else:
    raise NameError("You are running the script in a new computer, please define dir_train")

# 将split_Br_frac和split_Bc_frac的内容都取1/x然后用"_"连接成字符串split_Br_frac_inv_str和split_Bc_frac_inv_str
split_Br_frac_inv = [str(int(1/i)) for i in split_Br_frac]
# print(split_Br_frac_inv)
split_Br_frac_inv_str = '_'.join(split_Br_frac_inv)
# print(split_Br_frac_inv_str)
split_Bc_frac_inv = [str(int(1/i)) for i in split_Bc_frac]
# print(split_Bc_frac_inv)
split_Bc_frac_inv_str = '_'.join(split_Bc_frac_inv)
# print(split_Bc_frac_inv_str)

dir_train_split = os.path.join(dir_train, 'split_Br'+split_Br_frac_inv_str+'_split_Bc'+split_Bc_frac_inv_str)
try:
    os.mkdir(dir_train_split)
except FileExistsError:
    pass
print(dir_train_split)
dir_test_split = os.path.join(dir_test, 'split_Br'+split_Br_frac_inv_str+'_split_Bc'+split_Bc_frac_inv_str)
try:
    os.mkdir(dir_test_split)
except FileExistsError:
    pass 
dir_result = os.path.join(dir_result, method, "f%i" % feedback_bits, "fc1")
try:
    os.mkdir(dir_result)
except FileExistsError:
    pass 

weightpath = 'encoder_fcw_f%i.npy' % feedback_bits
biaspath = 'encoder_fcb_f%i.npy' % feedback_bits

# print(dir_result)


# %% [markdown]
# ### 切分训练集

# %% [markdown]
# 切分训练集A矩阵

for ncodebooks in (128, 256, 512):
    for amm_ind_list in ([(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0), (1, 1)]):#[(0,0), (0, 1), (1, 0), (1, 1)] # 使用近似矩阵乘的小矩阵索引（针对矩阵B）
    # amm_ind_list = [(0, 0)]
    # amm_ind_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 0)]
        amm_simple_ind_list = []
        for ind in amm_ind_list:
            amm_simple_ind_list.append(ind[0]*n_split_Bc + ind[1])
        # %%
        data_to_fc_train = np.load(os.path.join(dir_train, data_to_fcpath_train))
        print(data_to_fc_train.shape)
        Ab = data_to_fc_train.shape[1]
        # split_size_Ac = int(data_to_fc_train.shape[1] / n_split_Br) # split后每个分块训练集A的列数
        split_size_Ac = [int(i * Ab) for i in split_Br_frac] # split后每个分块训练集A的列数
        data_to_fc_train_split_path_list = []
        for i in range(n_split_Br):
            data_to_fc_train_split_path_list.append(os.path.join(dir_train_split, 'data_to_fc1_train_f'+str(feedback_bits)+'_split'+str(n_split_Br)+'_'+str(i)+'.npy'))
            np.save(data_to_fc_train_split_path_list[i], data_to_fc_train[np.ix_(range(data_to_fc_train.shape[0]), range(sum(split_size_Ac[:i]), sum(split_size_Ac[:(i+1)])))])

        # %% [markdown]
        # 切分训练集B矩阵

        # %%
        weight_train = np.load(os.path.join(dir_train, weightpath))
        print(weight_train.shape)
        Bb = weight_train.shape[1]
        split_size_Br = split_size_Ac # split后每个分块B的行数
        # split_size_Bc = int(weight_train.shape[1] / n_split_Bc) # split后B的列数
        split_size_Bc = [int(i * Bb) for i in split_Bc_frac] # split后每个分块训练集B的列数
        # split_total_num_B =  *  # B矩阵分成了多少块
        weight_train_split_path_list = []
        B_split_ind = 0 # B的分块的序号，从0开始
        for i in range(n_split_Br):
            for j in range(n_split_Bc):
                weight_train_split_path_list.append(os.path.join(dir_train_split, 'weight_fc1_train_f'+str(feedback_bits)+'_sr'+str(n_split_Br)+'_'+str(i)+'_sc'+str(n_split_Bc)+'_'+str(j)+'.npy'))
                np.save(weight_train_split_path_list[i*n_split_Bc + j], weight_train[np.ix_(range(sum(split_size_Br[:i]), sum(split_size_Br[:(i+1)])), range(sum(split_size_Bc[:j]), sum(split_size_Bc[:(j+1)])))])
                

        # %% [markdown]
        # 训练集A与B相乘

        # %%
        y_train_split_path_list = []
        for i in range(n_split_Br):
            for j in range(n_split_Bc):
                y_train_split_path_list.append(os.path.join(dir_train_split, 'y_fc1_train_f'+str(feedback_bits)+'_sr'+str(n_split_Br)+'_'+str(i)+'_sc'+str(n_split_Bc)+'_'+str(j)+'.npy'))
                xx = np.load(data_to_fc_train_split_path_list[i])
                ww = np.load(weight_train_split_path_list[i*n_split_Bc + j])
                np.save(y_train_split_path_list[i*n_split_Bc + j], np.matmul(xx, ww))

        # %% [markdown]
        # ### 切分测试集

        # %%
        data_to_fc_test = np.load(os.path.join(dir_test, data_to_fcpath_test))
        print(data_to_fc_test.shape)
        # split_size = int(data_to_fc_train.shape[1] / split) # split后单个训练集A的列数
        data_to_fc_test_split_path_list = []
        for i in range(n_split_Br):
            data_to_fc_test_split_path_list.append(os.path.join(dir_test_split, 'data_to_fc1_test_f'+str(feedback_bits)+'_split'+str(n_split_Br)+'_'+str(i)+'.npy'))
            np.save(data_to_fc_test_split_path_list[i], data_to_fc_test[np.ix_(range(data_to_fc_test.shape[0]), range(sum(split_size_Ac[:i]), sum(split_size_Ac[:(i+1)])))])

        # %%
        # # 输入与weight相乘
        # y_test_split_path_list = []
        # for i in range(split):
        #     y_test_split_path_list.append(os.path.join(dir_test_split, 'y_fc1_test_f'+str(feedback_bits)+'_split'+str(split)+'_'+str(i)+'.npy'))
        #     xx = np.load(data_to_fc_test_split_path_list[i])
        #     ww = np.load(weight_train_split_path_list[i])
        #     np.save(y_test_split_path_list[i], np.matmul(xx, ww))

        # %% [markdown]
        # ### AMM训练

        # %%
        est_list = []
        for i in range(n_split_Br):
            for j in range(n_split_Bc):
                if (i, j) in amm_ind_list:
                    dir_est, X_path = os.path.split(data_to_fc_train_split_path_list[i])
                    dir_est, W_path = os.path.split(weight_train_split_path_list[i*n_split_Bc + j])
                    dir_est, Y_path = os.path.split(y_train_split_path_list[i*n_split_Bc + j])
                    est3 = mm.estFactory(X_path=X_path, W_path=W_path, Y_path=Y_path, dir= dir_est, ncodebooks=ncodebooks, ncentroids=ncentroids, methods=[method])
                    est_list.append(est3)



        # for i in range(split):
        #     dir_est, X_path = os.path.split(data_to_fc_train_split_path_list[i])
        #     dir_est, W_path = os.path.split(weight_train_split_path_list[i])
        #     dir_est, Y_path = os.path.split(y_train_split_path_list[i])
        #     est3 = mm.estFactory(X_path=X_path, W_path=W_path, Y_path=Y_path, dir= dir_est, ncodebooks=ncodebooks, ncentroids=ncentroids, methods=[method])
        #     est_list.append(est3)


        # %%
        y_split_out_matmul_list = []
        i_est_list = 0 # est_list的索引
        for i in range(n_split_Br):
            for j in range(n_split_Bc):
                x_test = np.load(data_to_fc_test_split_path_list[i])
                w_test = np.load(weight_train_split_path_list[i*n_split_Bc + j])
                if (i, j) in amm_ind_list: # 用近似矩阵乘法的分块
                    y_split_out_matmul = mm.eval_matmul(est_list[i_est_list], x_test, w_test)
                    i_est_list += 1
                else:
                    y_split_out_matmul = np.matmul(x_test, w_test)
                y_split_out_matmul_list.append(y_split_out_matmul)

        # %%
        # split 的矩阵结果合成
        # 先纵向相加
        sum_col_list = []
        for j in range(n_split_Bc):
            sum_col = np.zeros((y_split_out_matmul_list[0].shape[0], split_size_Bc[j]))
            for i in range(n_split_Br):
                sum_col += y_split_out_matmul_list[i * n_split_Bc + j]
            sum_col_list.append(sum_col)

        #再横向拼接
        y_out_matmul = sum_col_list[0]
        if n_split_Bc > 1:
            for j in range(1, n_split_Bc):
                y_out_matmul = np.append(y_out_matmul, sum_col_list[j], axis=1)


        # %%
        bias = np.load(dir_train+'/'+biaspath)
        y_out_last = y_out_matmul + bias.T # MADDNESS替换后当前层输出，即+bias并不需要激活函数后的结果

        # %%
        amm_simple_ind_str= ''
        for ind in amm_simple_ind_list:
            amm_simple_ind_str += str(ind)

        # print(amm_simple_ind_str)

        # %%
        print(y_out_last)
        print(y_out_last.shape)
        # np.save("LDPC_decoder_NET_testdata/" + snr + "nomul_matmul_yout_matmul", y_out_matmul)
        # np.save(dir_result+'/'+method+'fc1_fb256_cb%i_ct%i.npy' % (ncodebooks, ncentroids), y_out_matmul)

        np.save(dir_result+'/'+method+'fc1_sr%s_sc%s_amm%s_fb%i_cb%i_ct%i.npy' % (split_Br_frac_inv_str, split_Bc_frac_inv_str, amm_simple_ind_str, feedback_bits, ncodebooks, ncentroids), y_out_last.astype(np.float32))
        # io.savemat(dir_result+'\\fc1_256.mat', {"NN_output_buffer": y_out_last})

        # %%



