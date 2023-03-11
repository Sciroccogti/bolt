# %% [markdown]
# encoder transformer层的linear2层（etl2）替换为近似矩阵乘法

# %%
import numpy as np
import os
import sys
# 获取当前文件所在的文件夹路径
if "__file__" in globals():
    # 获取__file__变量的值
    file_path = __file__
    # 获取当前文件所在的文件夹路径
    dir_now = os.path.dirname(file_path)
else:
    # 获取当前工作目录
    dir_now = os.getcwd()
sys.path.append(dir_now)
sys.path.append(os.path.join(dir_now, '../'))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # 防止jupyter爆内存
import matmul as mm
import math_util as mu
from NNutils import *
# import scipy.io as io
from amm_methods import *
import socket # Obtain the current host name, which can be used to select different data directories and result saving directories

# %%
# method = METHOD_MITHRAL
# method = METHOD_PQ
# method = METHOD_PLUTO
# method = METHOD_MITHRALPQ
# method = METHOD_EXACT
# method = METHOD_SCALAR_QUANTIZE
quantize_lut = True
for method in [METHOD_PQ]:

    linear_name = 'etl2'
    feedback_bits = 256
    linear_name_full = "ex_linear2"

    auto_train_change_nbits = True # 是否根据已运行的训练性能结果改变nbits自动训练，（train_sam_num取已训练的最大值）
    auto_train_change_upcast = False # 是否根据已运行的训练性能结果改变upcast自动训练，（train_sam_num取已训练的最大值）

    if auto_train_change_upcast == True:
        if method == METHOD_MITHRAL:
            upcast_trained = -1
            upcast_goal = 16
        else:
            upcast_trained = -1
            upcast_goal = 16
    else:
        if method == METHOD_MITHRAL:
            upcast_goal = 16
        else:
            upcast_goal = -1


    nbits_trained = 0
    
    nbits_goal = 10
    if quantize_lut == False:
        nbits_goal = 0
    nbits = nbits_goal # 要运行的量化比特数
    upcast_every = upcast_goal # 要运行的upcast
    lut_work_const = -1

    test_sam_num = 1000 # 测试集样本数(如需修改，请同时修改下面的读取文件，现文件默认1000个样本)

    if not auto_train_change_nbits and not auto_train_change_upcast:
        ncodebooks = 512 # max:512
        ncentroids = 16
        train_sam_num = 50 # 训练集样本数
    elif auto_train_change_nbits:
        param2change = "nbits"
        param_trained = nbits_trained
        param_goal = nbits_goal
        cb_ct_ntr_combinations_unique = change_param_auto_run_list(linear_name, method, feedback_bits, param2change, param_trained, param_goal, "upcast_every", upcast_every)
        print(cb_ct_ntr_combinations_unique)
        # 遍历每个cb、ct、n_train_sam组合
        # for _, row_ref in cb_ct_ntr_combinations_unique.iterrows():
        #     ncodebooks = int(row_ref['cb'])
            # ncentroids = int(row_ref['ct'])
            # train_sam_num = int(row_ref['n_train_sam'])
    elif auto_train_change_upcast:
        param2change = "upcast_every"
        param_trained = upcast_trained
        param_goal = upcast_goal

        cb_ct_ntr_combinations_unique = change_param_auto_run_list(linear_name, method, feedback_bits, param2change, param_trained, param_goal, "nbits", nbits_goal)
        print(cb_ct_ntr_combinations_unique)
    i=0
    # 遍历每个cb、ct、n_train_sam组合
    for _, row_ref in cb_ct_ntr_combinations_unique.iterrows():
        if i in range(1):
            i+=1
            # continue
        ncodebooks = int(row_ref['cb'])
        ncentroids = int(row_ref['ct'])
        train_sam_num = int(row_ref['n_train_sam'])

        batch_size = 32
        if method == METHOD_EXACT:
            ncodebooks = 0
            ncentroids = 0

        AMM_train_dirs = get_AMM_train_dirs(linear_name, linear_name_full, method, feedback_bits, train_sam_num, test_sam_num)
        create_dir(AMM_train_dirs["dir_result"])

        dataset_prepare(AMM_train_dirs["dir_joined"], linear_name_full, feedback_bits, [train_sam_num, test_sam_num], 
                        batch_size, S1 = S1_dict[linear_name])

        if method == METHOD_PLUTO:
            est3 = mm.estFactory(X_path=AMM_train_dirs["linearin_path_train"], W_path=AMM_train_dirs["weightpath"], 
                                Y_path=AMM_train_dirs["y_train"], dir= AMM_train_dirs["dir_train"], ncodebooks=ncodebooks, 
                                ncentroids=ncentroids, methods=[method], nbits=nbits, quantize_lut = quantize_lut, 
                                upcast_every=upcast_every, bias_path=AMM_train_dirs["biaspath"],lut_work_const=-1)
        elif method == METHOD_MITHRAL:
            est3 = mm.estFactory(X_path=AMM_train_dirs["linearin_path_train"], W_path=AMM_train_dirs["weightpath"], 
                                Y_path=AMM_train_dirs["y_train"], dir= AMM_train_dirs["dir_train"], ncodebooks=ncodebooks, 
                                ncentroids=ncentroids, methods=[method], nbits=nbits, quantize_lut = quantize_lut,
                                upcast_every=upcast_every, lut_work_const=lut_work_const)
        else:
            est3 = mm.estFactory(X_path=AMM_train_dirs["linearin_path_train"], W_path=AMM_train_dirs["weightpath"], 
                                Y_path=AMM_train_dirs["y_train"], dir= AMM_train_dirs["dir_train"], ncodebooks=ncodebooks, 
                                ncentroids=ncentroids, methods=[method], nbits=nbits, quantize_lut = quantize_lut,
                                upcast_every=upcast_every)

        x_test = np.load(AMM_train_dirs["dir_test"]+'/'+AMM_train_dirs["linearin_path_test"])
        w_test = np.load(AMM_train_dirs["dir_train"]+'/'+AMM_train_dirs["weightpath"])
        bias = np.load(AMM_train_dirs["dir_train"]+'/'+AMM_train_dirs["biaspath"])
        # print(type(est3))
        y_out_matmul = mm.eval_matmul(est3, x_test, w_test) # MADDNESS乘法的结果
        # y_out_last = mu.softmax(y_out_matmul + bias.T) # MADDNESS替换后当前层输出，即+bias并激活函数后的结果
        if method == METHOD_PLUTO:
            y_out_last = y_out_matmul
        else:
            y_out_last = y_out_matmul + bias.T # MADDNESS替换后当前层输出，即+bias并不需要激活函数后的结果

        print(y_out_last)
        print("y_out_last.shape: ", y_out_last.shape)
        y_out_last_re = y_out_last.reshape(test_sam_num, batch_size, -1, y_out_last.shape[-1]) #AMM字典模式需要复原y大小
        print("y_out_last_re.shape: ", y_out_last_re.shape)
        if method == METHOD_SCALAR_QUANTIZE:
            np.save(os.path.join(AMM_train_dirs["dir_result"], '%s%s_trsam%i_tesam%i_fb%i_nbits%i.npy' % 
                                                                (method, linear_name, train_sam_num, test_sam_num, feedback_bits, nbits)), 
                                                                y_out_last_re.astype(np.float32))
        elif method == METHOD_MITHRAL or method == METHOD_PLUTO:
            np.save(os.path.join(AMM_train_dirs["dir_result"], '%s%s_trsam%i_tesam%i_fb%i_cb%i_ct%i_ql%i_nb%i_uc%i_lwc%i.npy' % 
                                                                (method, linear_name, train_sam_num, test_sam_num, feedback_bits, 
                                                                ncodebooks, ncentroids, quantize_lut, nbits, upcast_every, lut_work_const)), y_out_last_re)
        elif method == METHOD_PQ or method == METHOD_MITHRALPQ:
            np.save(os.path.join(AMM_train_dirs["dir_result"], '%s%s_trsam%i_tesam%i_fb%i_cb%i_ct%i_ql%i_nb%i_uc%i.npy' % 
                                                                (method, linear_name, train_sam_num, test_sam_num, feedback_bits, 
                                                                ncodebooks, ncentroids, quantize_lut, nbits, upcast_every)), y_out_last_re)
        else:
            np.save(os.path.join(AMM_train_dirs["dir_result"], '%s%s_trsam%i_tesam%i_fb%i_cb%i_ct%i.npy' % 
                                                                (method, linear_name, train_sam_num, test_sam_num, feedback_bits, 
                                                                ncodebooks, ncentroids)), y_out_last_re)


