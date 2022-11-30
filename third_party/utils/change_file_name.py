# 指定文件夹中修改文件名前缀的脚本
import os
 
path = input('请输入文件夹路径：')
prefix_pre = input('请输入修改前文件名前缀：')
prefix_after = input('请输入修改后文件名前缀：')
 
# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
 
m = 1  # python中input函数默认返回一个字符串，需强制转化为整数
n = m

for inner_file in fileList:
    if inner_file.startswith(prefix_pre):
        n += 1

print("有%i个文件以%s为开头。" %(n-m, prefix_pre))
change_or_not = input('请确认是否修改（Y、y/N、n）：')

if change_or_not == "Y" or change_or_not == "y":
    for inner_file in fileList:
        # 获取旧文件名（就是路径+文件名）
        old_name = path + os.sep + inner_file  # os.sep添加系统分隔符
        if os.path.isdir(old_name):  # 如果是目录则跳过
            continue
    
        # 设置新文件名
        if inner_file.startswith(prefix_pre):
            new_name = path + os.sep + prefix_after + inner_file[len(prefix_pre):]
            os.rename(old_name, new_name)  # 用os模块中的rename方法对文件改名
    print("修改完成")
else:
    print("未修改")

    
