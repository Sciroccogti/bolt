# %% [markdown]
# encoder transformer层的linear2层（etl2）替换为近似矩阵乘法

import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # 防止jupyter爆内存
import matmul as mm
from NNutils import *
from amm_methods import *
import socket # Obtain the current host name, which can be used to select different data directories and result saving directories

# method = METHOD_MITHRAL
# method = METHOD_PQ
# method = METHOD_PLUTO
# method = METHOD_MITHRALPQ
# method = METHOD_EXACT
# method = METHOD_SCALAR_QUANTIZE
quantize_lut = True


linear_name = 'etl2'
feedback_bits = 256
linear_name_full = "ex_linear2"

auto_train = True # 是否根据已运行的训练性能结果自动训练，（train_sam_num取已训练的最大值）
nbits_trained = 8
nbits_goal = 12
if quantize_lut == False:
    nbits_goal = 0
nbits = nbits_goal # 要运行的量化比特数
test_sam_num = 1000 # 测试集样本数(如需修改，请同时修改下面的读取文件，现文件默认1000个样本)



for method in [METHOD_MITHRAL,METHOD_PQ]:
    if not auto_train:
        ncodebooks = 128 # max:512
        ncentroids = 16
        train_sam_num = 1000 # 训练集样本数
    else:
        cb_ct_ntr_combinations_unique = change_nbits_auto_run_list(linear_name, method, feedback_bits, nbits_trained, nbits_goal)
        print(cb_ct_ntr_combinations_unique)
    # 遍历每个cb、ct、n_train_sam组合
    for _, row_ref in cb_ct_ntr_combinations_unique.iterrows():
        ncodebooks = int(row_ref['cb'])
        ncentroids = int(row_ref['ct'])
        train_sam_num = int(row_ref['n_train_sam'])

        batch_size = 32
        if method == METHOD_EXACT:
            ncodebooks = 0
            ncentroids = 0

        host_name = socket.gethostname()
        if host_name == 'DESKTOP-PLRL7TK':
            dir_train = 'E:\\hdr\\研一\\华为-深度学习\\intermediate\\intermediate8dbfc1'
            dir_result = ''
        elif host_name == 'DESKTOP-6FOH47P':
            dir_train = 'F:\\Projects\\python\\PQ\\intermediate8dbfc1'
            dir_result = 'F:\\Projects\\python\\PQ\\res'
            linearin_path_train= ''
            linearout_path_train= ''
            linearin_path_test = ''
            linearout_path_test = ''
        elif host_name == 'jm-System-Product-Name':
            dir_joined = '/data/hdr/transformer_data/joined'
            dir_train = os.path.join(dir_joined, 'train', 'f'+str(feedback_bits))
            dir_test = os.path.join(dir_joined, 'test', 'f'+str(feedback_bits))
            dir_result = '/data/hdr/pq/res'
            linearin_path_train= '%sin_train_f%i_sam%i.npy' % (linear_name_full, feedback_bits, train_sam_num)
            y_train = '%s_y_train_f%i_sam%i.npy' % (linear_name_full, feedback_bits, train_sam_num)
            linearout_path_train= '%sout_train_f%i_sam%i.npy' % (linear_name_full, feedback_bits, train_sam_num)
            linearin_path_test = '%sin_test_f%i_sam%i.npy' % (linear_name_full, feedback_bits, test_sam_num)
            linearout_path_test = '%sout_test_f%i_sam%i.npy' % (linear_name_full, feedback_bits, test_sam_num)
        else:
            raise NameError("You are running the script in a new computer %s, please define dirs" % host_name)


        weightpath = '%s_w_f%i.npy' % (linear_name_full, feedback_bits)
        biaspath = '%s_b_f%i.npy' % (linear_name_full, feedback_bits)
        dir_result = os.path.join(dir_result, method, "f%i" % feedback_bits, linear_name)
        try:
            os.mkdir(dir_result)
        except FileNotFoundError:
            os.makedirs(dir_result)
        except FileExistsError:
            pass 


        dataset_prepare(dir_joined, linear_name_full, feedback_bits, [train_sam_num, test_sam_num], batch_size, S1 = S1_dict[linear_name])

        if method == METHOD_PLUTO:
            est3 = mm.estFactory(X_path=linearin_path_train, W_path=weightpath, Y_path=y_train, dir= dir_train,\
                                ncodebooks=ncodebooks, ncentroids=ncentroids, methods=[method], nbits=nbits, \
                                quantize_lut = quantize_lut, bias_path=biaspath)
        else:
            est3 = mm.estFactory(X_path=linearin_path_train, W_path=weightpath, Y_path=y_train, dir= dir_train,\
                                ncodebooks=ncodebooks, ncentroids=ncentroids, methods=[method], nbits=nbits, \
                                quantize_lut = quantize_lut)


        x_test = np.load(dir_test+'/'+linearin_path_test)
        w_test = np.load(dir_train+'/'+weightpath)
        bias = np.load(dir_train+'/'+biaspath)
        y_out_matmul = mm.eval_matmul(est3, x_test, w_test) # MADDNESS乘法的结果
        if method == METHOD_PLUTO:
            y_out_last = y_out_matmul
        else:
            y_out_last = y_out_matmul + bias.T # MADDNESS替换后当前层输出，即+bias并不需要激活函数后的结果

        print(y_out_last)
        print("y_out_last.shape: ", y_out_last.shape)
        y_out_last_re = y_out_last.reshape(test_sam_num, batch_size, -1, y_out_last.shape[-1]) #AMM字典模式需要复原y大小
        print("y_out_last_re.shape: ", y_out_last_re.shape)
        if method == METHOD_SCALAR_QUANTIZE:
            np.save(os.path.join(dir_result, '%s%s_trsam%i_tesam%i_fb%i_nbits%i.npy' % (method, linear_name, train_sam_num, test_sam_num, feedback_bits, nbits)), y_out_last_re.astype(np.float32))
        elif method == METHOD_MITHRAL or method == METHOD_PQ or method == METHOD_PLUTO or method == METHOD_MITHRALPQ:
            np.save(os.path.join(dir_result, '%s%s_ql%i_nbits%i_trsam%i_tesam%i_fb%i_cb%i_ct%i.npy' % (method, linear_name, quantize_lut, nbits, train_sam_num, test_sam_num, feedback_bits, ncodebooks, ncentroids)), y_out_last_re)
        else:
            np.save(os.path.join(dir_result, '%s%s_trsam%i_tesam%i_fb%i_cb%i_ct%i.npy' % (method, linear_name, train_sam_num, test_sam_num, feedback_bits, ncodebooks, ncentroids)), y_out_last_re)

