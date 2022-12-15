# 有关transformer数据集的工具
import numpy as np
import os
import socket
import re # 正则表达式

bits = 256
batch_size = 32
S1 = [32,32,1,32,32,1] # ex_linear1in大小为batch_size*S1*S2;etl1,etl2,fc1,dtl1,dtl2,fc2的S1大小
S1_dict = {"etl1":32, "etl2":32, "fc1":1, "dtl1":32, "dtl2":32, "fc2":1}
nbits = 8 # # METHOD_SCALAR_QUANTIZE的量化比特数
# whole_train_sam_num = 7000 # 完整的训练集样本数
# smaller_train_sam_num = 3000 # 减小内存消耗的训练集样本数
# smallerer_train_sam_num = 1000
# smallererer_train_sam_num = 50
AMM_name_tran = {"etl1":"ex_linear1", "etl2":"ex_linear2", "fc1":"fc1", "dtl1":"dx_linear1", "dtl2":"dx_linear2", "fc2":"fc2"} # 顺序不变，与S1的顺序一致
host_name = socket.gethostname()
if host_name == 'DESKTOP-PLRL7TK':
    dir_intermediate = ''
elif host_name == 'DESKTOP-6FOH47P':
    dir_intermediate = 'F:\\Projects\\python\\PQ\\intermediate8dbfc1\\'
elif host_name == 'jm-System-Product-Name':
    dir_intermediate = '/data/hdr/transformer_data/intermediate/'
    dir_train = os.path.join('/data/hdr/transformer_data/joined', 'train', 'f'+str(bits))
    dir_test = os.path.join('/data/hdr/transformer_data/joined', 'test', 'f'+str(bits))
    # dir1 = '/data/hdr/transformer_data/joined/'
else:
    raise NameError("You are running the script in a new computer, please define dir_intermediate")

def create_dir(directory): # 创建（尚未存在的）空目录函数
    try:
        os.mkdir(directory)
    except FileNotFoundError:
        os.makedirs(directory)
    except FileExistsError:
        pass 

def del_linear_suffix(intermediate_name:str): # 删除全连接层名称的尾缀（in、out、_y）
    if intermediate_name.endswith('_y') or intermediate_name.endswith('in'):
        linear_name = intermediate_name[:-2]
    elif intermediate_name.endswith('out'):
        linear_name = intermediate_name[:-3]
    else:
        linear_name = intermediate_name
    return linear_name

def get_AMM_train_dirs(linear_name, linear_name_full, method, feedback_bits, train_sam_num, test_sam_num):
    AMM_train_dirs = {}
    dir_now = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在目录
    AMM_train_dirs["dir_joined"] = os.path.join(dir_now, "../../../../transformer_data/joined")
    AMM_train_dirs["dir_train"] = os.path.join(AMM_train_dirs["dir_joined"], 'train', 'f'+str(feedback_bits))
    AMM_train_dirs["dir_test"] = os.path.join(AMM_train_dirs["dir_joined"], 'test', 'f'+str(feedback_bits))
    AMM_train_dirs["dir_result"] = os.path.join(dir_now, "../../../res", method, "f%i" % feedback_bits, linear_name)
    AMM_train_dirs["linearin_path_train"] = '%sin_train_f%i_sam%i.npy' % (linear_name_full, feedback_bits, train_sam_num)
    AMM_train_dirs["y_train"] = '%s_y_train_f%i_sam%i.npy' % (linear_name_full, feedback_bits, train_sam_num)
    AMM_train_dirs["linearout_path_train"]= '%sout_train_f%i_sam%i.npy' % (linear_name_full, feedback_bits, train_sam_num)
    AMM_train_dirs["linearin_path_test"] = '%sin_test_f%i_sam%i.npy' % (linear_name_full, feedback_bits, test_sam_num)
    AMM_train_dirs["linearout_path_test"] = '%sout_test_f%i_sam%i.npy' % (linear_name_full, feedback_bits, test_sam_num)

    AMM_train_dirs["weightpath"] = '%s_w_f%i.npy' % (linear_name_full, feedback_bits)
    AMM_train_dirs["biaspath"] = '%s_b_f%i.npy' % (linear_name_full, feedback_bits)

    return AMM_train_dirs

# 从单batch样本合成大样本集（样本从序号0开始），方便AMM训练 #j1代表合并第一维
def join_from_intermediate_j1(dir_intermediate, dir_t, dire_train, bits, intermediate_name, sam_num, trainortest):
    #sam_num:合并的样本数;trainortest:合成训练集填"train",测试集填"test"
    linearpath0= os.path.join(dir_intermediate, str(bits), intermediate_name+'_f%i_e39_0.npy' % bits)#例：此处intermediate_name为linear
    linear0 = np.load(linearpath0)
    print("生成的数据集前缀: ", intermediate_name, trainortest)
    print("样本合并第一维前大小：", linear0.shape)
    #把第一维合并
    linear0_join1 = np.reshape(linear0, (-1, linear0.shape[-1]))
    print("样本合并第一维后大小：", linear0_join1.shape)
    if trainortest == "test":
        add = 7000
    else:
        add = 0
    for i in range(1+add, sam_num+add):
        linearpath1= os.path.join(dir_intermediate, str(bits), intermediate_name+'_f%i_e39_%i.npy' % (bits, i) )
        linear1 = np.load(linearpath1)
        linear1_join1 = np.reshape(linear1, (-1, linear1.shape[-1]))
        if linear1_join1.shape[0]!=1024:
            print("i",str(i),",shape",str(linear1_join1.shape[0]))
        linear0_join1 = np.append(linear0_join1, linear1_join1, axis=0)
    print("合并后数据集大小: ", linear0_join1.shape)
    np.save(os.path.join(dir_t, '%s_%s_f%i_sam%i.npy' % (intermediate_name,trainortest,bits,sam_num)), linear0_join1) 
    print("intermediate_name[-3:] == out:",intermediate_name[-3:] == "out")
    if intermediate_name[-3:] == "out": # 如果合并的是out数据集，顺带把y=out-bias也合并了
        linear_name=intermediate_name[:-3]
        bias = np.load(os.path.join(dire_train, "%s_b_f%i.npy" % (linear_name, bits)))
        y = linear0_join1 - bias
        np.save(os.path.join(dir_t, '%s_y_%s_f%i_sam%i.npy' % (linear_name, trainortest, bits, sam_num)), y) 
    

# 从单batch样本合成大样本集（样本从序号0开始），方便AMM训练 #不需要合并第一维
def join_from_intermediate(dir_intermediate, dir_t, dire_train, bits, intermediate_name, sam_num, trainortest):
    #sam_num:合并的样本数;trainortest:合成训练集填"train",测试集填"test"
    linearinpath0= os.path.join(dir_intermediate, str(bits), intermediate_name+'_f%i_e39_0.npy' % bits)#例：此处intermediate_name为linearin
    linearin0 = np.load(linearinpath0)
    print("生成的数据集前缀: ", intermediate_name, trainortest)
    print("原样本大小: ", linearin0.shape)
    if trainortest == "test":
        add = 7000
    else:
        add = 0
    for i in range(1+add, sam_num+add):
        linearinpath1= os.path.join(dir_intermediate, str(bits), intermediate_name+'_f%i_e39_%i.npy' % (bits, i) )
        linearin1 = np.load(linearinpath1)
        linearin1 = np.reshape(linearin1, (-1, linearin1.shape[-1]))
        if linearin1.shape[0]!=32:
            print("i",str(i),",shape",str(linearin1.shape[0]))
        linearin0 = np.append(linearin0, linearin1, axis=0)
    print("合并后数据集大小: ", linearin0.shape)
    np.save(os.path.join(dir_t, '%s_%s_f%i_sam%i.npy' % (intermediate_name,trainortest,bits,sam_num)), linearin0) 
    if intermediate_name[-3:] == "out": #如果合并的是out数据集，顺带把y=out-bias也合并了
        linear_name=intermediate_name[:-3]
        bias = np.load(os.path.join(dire_train, "%s_b_f%i.npy" % (linear_name, bits)))
        y = linearin0 - bias
        np.save(os.path.join(dir_t, '%s_y_%s_f%i_sam%i.npy' % (linear_name, trainortest, bits, sam_num)), y) 
        

# 从已合成训练/测试集中提取更小的1个训练/测试集
def join_from_joined(dir_t, intermediate_name, bits, joined_sam_num, sam_num, batch_size, trainortest, S1 = 1):# 从已合成训练/测试集中提取更小的训练测试集
    # 例:intermediate_name:'ex_linear1in'
    # joined_sam_num：已合成训练/测试集的样本数
    # sam_num：提取出的训练/测试集的样本数
    # S1对于transformer子模块外的全连接层为1，对于transformer子模块内的全连接层不为1
    linear_whole = np.load(os.path.join(dir_t,  '%s_%s_f%i_sam%i.npy' % (intermediate_name, trainortest, bits,joined_sam_num)))
    print("生成的数据集前缀: ", intermediate_name, trainortest)
    print("原数据集大小: ", linear_whole.shape)
    linear_smaller = linear_whole[np.ix_(range(sam_num*batch_size*S1), range(linear_whole.shape[1]))]
    print("提取后数据集大小: ", linear_smaller.shape)
    np.save(os.path.join(dir_t, '%s_train_f%i_sam%i.npy' % (intermediate_name, bits, sam_num)), linear_smaller) 
    if intermediate_name[-3:] == "out":
        linear_name=intermediate_name[:-3]
        bias = np.load(os.path.join(dir_train, "%s_b_f%i.npy" % (linear_name, bits)))
        y = linear_smaller - bias
        np.save(os.path.join(dir_t, '%s_y_%s_f%i_sam%i.npy' % (linear_name, trainortest, bits, sam_num)), y) 

# 从已合成训练/测试集中分割的多个等大小的训练/测试集
def split_from_joined(dir_t, intermediate_name, bits, joined_sam_num, sam_num, batch_size, trainortest, n_split: int = -1, S1 = 1):# 从已合成训练/测试集中提取更小的训练测试集
    # 例:intermediate_name:'ex_linear1in'
    # joined_sam_num：已合成训练/测试集的样本数
    # sam_num：分割出的*每个*训练/测试集的样本数
    # n_split: 分割出的训练/测试集的个数。-1表示最大，其他可为正整数
    # S1对于transformer子模块外的全连接层为1，对于transformer子模块内的全连接层不为1
    assert joined_sam_num % sam_num == 0 # 大样本集的样本数必须是分割成的小样本集的样本数的整数倍
    n_split_max = joined_sam_num / sam_num
    assert (n_split==-1 or n_split>0) and n_split<=n_split_max # nsplit需要为-1或小于等于最大值的正整数
    if n_split == -1:# n_split: 分割出的训练/测试集的个数。-1表示最大，其他正整数
        n_split = n_split_max
    linear_whole = np.load(os.path.join(dir_t,  '%s_%s_f%i_sam%i.npy' % (intermediate_name, trainortest, bits,joined_sam_num)))
    print("生成的数据集前缀: %s_%s" % (intermediate_name, trainortest))
    print("原数据集大小: ", linear_whole.shape)
    linear_name = del_linear_suffix(intermediate_name)
    dir_split = os.path.join(dir_t, intermediate_name, 'sam_num%i' % sam_num)
    create_dir(dir_split)
    bias = np.load(os.path.join(dir_train, "%s_b_f%i.npy" % (linear_name, bits)))
    for i_split in range(n_split):
        linear_smaller = linear_whole[np.ix_(range(i_split*sam_num*batch_size*S1, (i_split+1)*sam_num*batch_size*S1), range(linear_whole.shape[1]))]
        if i_split == 0:
            print("提取后每个数据集大小: ", linear_smaller.shape)
            print("提取后小数据集个数: ", n_split)
        np.save(os.path.join(dir_split, '%s_%s_f%i_sam%i_i%i.npy' % (intermediate_name, trainortest, bits, sam_num, i_split)), linear_smaller) 
        if intermediate_name[-3:] == "out":
            y = linear_smaller - bias
            np.save(os.path.join(dir_split, '%s_y_%s_f%i_sam%i_i%i.npy' % (linear_name, trainortest, bits, sam_num, i_split)), y) 


def findfiles(dire, file_str):
    #查找dir目录下文件名含有file_str的文件，返回符合要求的完整文件路径列表path_result和符合要求的文件名列表file_result
    path_result = []
    file_result = []
    for root,_,files in os.walk(dire):
        for file_name in files:
            if file_str in file_name:
                path_result.append(os.path.join(root,file_name).replace('\\','/')) # 符合要求的完整文件路径列表
                file_result.append(file_name) # 符合要求的文件名列表
    return path_result, file_result


def find_max_dataset(dire, file_str):
    # 查找dir目录下文件名含有file_str的文件中数据集最大的样本数
    _, file_result = findfiles(dire, file_str)
    str2find = "_sam" # 文件名中代表样本数的字符串标示
    sam_num_list = []
    for file_name in file_result:
        idx = file_name.rfind(str2find) # str2find在每一个文件名中的位置索引
        sam_num = re.findall("\d+",file_name[(idx+len(str2find)):]) # 获得文件名中代表样本数的字符串标示后的数字，代表样本数
        sam_num = int(sam_num[0])
        sam_num_list.append(sam_num)

    if sam_num_list: #找到了符合名称的数据集，输出最大样本数
        max_sam_num = max(sam_num_list)
        return max_sam_num
    else:
        return 0


def dataset_prepare(direc, linear_name_full, feedback_bits, sam_num_list, batch_size, S1 = 1):
    # 准备全连接层linear_name_full在CSI压缩后长度为feedback_bits的数据集，训练样本数为sam_num_list[0], 测试样本数为sam_num_list[1]，样本大小为batch_size
    # S1对于transformer子模块外的全连接层为1，对于transformer子模块内的全连接层不为1
    in_transformer_list = ["ex_linear1", "ex_linear2", "dx_linear1", "dx_linear2"]
    out_transformer_list = ["fc1", "fc2"]
    data_place_list = ["in", "out", "_y"]
    tt_list = ["train", "test"]
    for data_place in data_place_list:
        for train_or_test in tt_list:
            dire = os.path.join(direc, train_or_test, 'f'+str(feedback_bits))
            dire_train = os.path.join(direc, "train", 'f'+str(feedback_bits))
            if train_or_test == "train":
                sam_num = sam_num_list[0]
            else:
                sam_num = sam_num_list[1]
            dataset_name = '%s%s_%s_f%i_sam%i.npy' % (linear_name_full, data_place, train_or_test, feedback_bits, sam_num)
            
            dataset_path = os.path.join(dire, dataset_name)
            if not os.path.exists(dataset_path): #不存在则创建数据集
                if data_place == "_y": # y数据集在生成out数据集时同时生成
                    data_place = "out"
                intermediate_name = '%s%s'% (linear_name_full, data_place)
                # 查找是否有更大的数据集
                print("查找以此为前缀的数据集：", '%s%s_%s_f%i' % (linear_name_full, data_place, train_or_test, feedback_bits))
                max_sam_num = find_max_dataset(dire, '%s%s_%s_f%i' % (linear_name_full, data_place, train_or_test, feedback_bits))
                
                if sam_num < max_sam_num:# 有更大的数据集
                    print("有比输入样本数更大的数据集，从中提取新数据集")
                    join_from_joined(dire, intermediate_name, feedback_bits, max_sam_num, sam_num, batch_size, train_or_test, S1)
                else:# 没有更大的数据集
                    print("没有比输入样本数更大的数据集，从样本合成新数据集")
                    if linear_name_full in in_transformer_list: # transformer子模块内的全连接层需要把数据集第一维合并，外的不需要合并
                        join_from_intermediate_j1(dir_intermediate, dire, dire_train, feedback_bits, intermediate_name, sam_num, train_or_test)
                    elif linear_name_full in out_transformer_list:
                        join_from_intermediate(dir_intermediate, dire, dire_train, feedback_bits, intermediate_name, sam_num, train_or_test)
                

