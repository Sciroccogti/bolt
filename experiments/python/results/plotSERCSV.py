'''
@file plotDFT.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-07-06 15:07:45
@modified: 2023-04-20 13:34:06
'''

import os
import re
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np


colors_ = ["r", "g", "b", "orange", "c", "m", "steelblue", "grey", "brown", "k"]
markers_ = ["o", "x", "s", "d", "+", "*", "v", "^", ">", "D"]

fig, [axBER, axNMSE] = plt.subplots(ncols=2, figsize=(12, 4.5))
plt.rcParams["font.sans-serif"] = ["Sarasa Mono SC Nerd"]
axBER.set_title(r"DFT-IDFT 的信道估计误比特率")
axBER.set_xlabel(r"$SNR$")
axBER.set_ylabel(r"$SER$")
axBER.set_ylim(top=.1, bottom=5e-4)
axBER.set_yscale("log")
axBER.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='all', numticks=10))
axBER.yaxis.set_minor_formatter(ticker.NullFormatter())
axBER.grid(True, which="major", ls="-", color="grey")
axBER.grid(True, which="minor", ls="--")
axBER.xaxis.set_tick_params(direction='in', which='both')  # 刻度线向内
axBER.yaxis.set_tick_params(direction='in', which='both')

axNMSE.set_title(r"DFT-IDFT 的信道估计较精确乘法估计信道差距")
axNMSE.set_xlabel(r"$SNR$")
axNMSE.set_ylabel(r"信道 $NMSE$")
axNMSE.set_ylim(top=2e-2, bottom=1e-3)
axNMSE.set_yscale("log")
axNMSE.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='all', numticks=10))
axNMSE.yaxis.set_minor_formatter(ticker.NullFormatter())
axNMSE.grid(True, which="major", ls="-", color="grey")
axNMSE.grid(True, which="minor", ls="--")
axNMSE.xaxis.set_tick_params(direction='in', which='both')  # 刻度线向内
axNMSE.yaxis.set_tick_params(direction='in', which='both')

axBER.set_xlim([0, 15])
# axBER.set_xlim([-5, 2.5])
# axBER.set_xticks(np.arange(-10, 5, step=2.5))

axNMSE.set_xlim([0, 15])
# axNMSE.set_xlim([-5, 2.5])
# axNMSE.set_xticks(np.arange(-10, 5, step=2.5))

path = "./dft/"
files_ = os.listdir(path)
files_.sort()
colorCnt = 0
for file in files_:
    match = None
    try:
        # if match == None:
        #     match = re.match(r"\+[0-9][a-z,A-Z].*?\.csv", file)
        # if match == None:
        #     match = re.match(r"\+\+\+[0-9][a-z,A-Z].*?\.csv", file)
        if match == None:
            match = re.match(r"\+rms75\+[0-9][a-z,A-Z].*?\.csv", file)
        # if match == None:
        #     match = re.match(r"\+DPQ[0-9]\+[a-z,A-Z].*?\.csv", file)
        if match != None:

            fin = open(path + file, "r")
            lines_ = fin.readlines()

            method = lines_[18].split(",")[1].strip()
            nbits = lines_[16].split(",")[1].strip()
            ncentroids = int(lines_[14].split(",")[1].strip())
            ncodebooks = int(lines_[13].split(",")[1].strip())
            if method == "Mithral":
                method = "MAD-(K=" + str(ncentroids) + " C=" + str(ncodebooks) + ")"
            if (int(nbits) <= 8 or int(nbits) > 16) and method.startswith("MAD"):
                continue
            curColor = colors_[colorCnt]
            curLine = "-"
            if method.startswith("Exact"):
                curColor = "k"
                curLine = "-."
            if method.startswith("SQ"):
                curLine = "-."
                continue
            if method.startswith("PQ"):
                curLine = ":"
                continue
            
            snr_ = []
            ber_ = []
            rawNMSE_ = []
            for line in lines_[20:]:
                try:
                    snr_.append(float(line.split(",")[0]))
                    if float(line.split(",")[1]) == 0:
                        ber_.append(None)
                    else:
                        ber_.append(float(line.split(",")[1]))
                    rawNMSE_.append(float(line.split(",")[6]))
                except:
                    break
            axBER.plot(snr_, ber_, label=method, color=curColor, linestyle=curLine, marker=markers_[colorCnt])
            axNMSE.plot(snr_, rawNMSE_, label=method+"-"+nbits+"bits", color=curColor, linestyle=curLine, marker=markers_[colorCnt])
            colorCnt += 1
    except Exception as e:
        print(file + ": ", e)
axBER.legend()
axNMSE.legend()
plt.show()