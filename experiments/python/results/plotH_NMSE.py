'''
@file plotDFT.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-07-06 15:07:45
@modified: 2022-07-06 16:10:47
'''

import os
import re
from matplotlib import ticker
import matplotlib.pyplot as plt


colors_ = ["r", "g", "b", "orange", "c", "m", "steelblue", "k"]

fig, ax = plt.subplots()
plt.rcParams["font.sans-serif"] = ["Sarasa Mono SC Nerd"]
plt.title(r"DFT-IDFT 的信道估计较精确乘法估计信道差距")
plt.xlabel(r"$SNR$")
plt.ylabel(r"信道 $NMSE$")
# plt.yscale("log")
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='all', numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(True, which="major", ls="-", color="grey")
ax.grid(True, which="minor", ls="--")
ax.xaxis.set_tick_params(direction='in', which='both')  # 刻度线向内
ax.yaxis.set_tick_params(direction='in', which='both')


path = "./"
files_ = os.listdir(path)
colorCnt = 0
for file in files_:
    match = None
    if match == None:
        match = re.match(r"\*.*?\.txt", file)

    if match != None:
        snr_ = [0, 3, 6, 9, 12, 15, 18, 21]

        fin = open(path + file, "r")
        lines_ = fin.readlines()

        method = lines_[12].split("'")[3]
        ncodebooks = int(
            re.match(r" 'ncodebooks': ([0-9]+)", lines_[13]).group(1))
        ber_ = [float(i) for i in lines_[18].split()]
        fer_ = [float(i) for i in lines_[21].split()]
        nmdr_DFT_ = [float(i) for i in lines_[24].split()]
        nmse_IDFT_ = [float(i) for i in lines_[27].split()]
        nmse_H_ = [float(i) for i in lines_[30].split()]
        plt.plot(snr_, nmse_H_, label=method, color=colors_[colorCnt], linestyle="-", marker="o")
        colorCnt += 1
plt.legend()
plt.show()