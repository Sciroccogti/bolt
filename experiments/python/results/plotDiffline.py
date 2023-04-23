'''
@file plotDiffline.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-04-20 10:27:12
@modified: 2023-04-20 14:12:34
'''
'''
@file plotDiff.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-03-27 15:12:36
@modified: 2023-04-13 14:27:34
'''




import os
import re
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
def complexity(method: str, ncodebooks: int, ncentroids: int) -> float:
    M = 16
    if method.startswith("MAD"):
        return (800 * ncodebooks * M + 100 * D * np.log2(ncentroids)
                + 1.2 * M * ncodebooks * ncentroids)
    elif method.startswith("PQ"):
        return (6000 * D * ncentroids + 800 * ncodebooks * M - 2900 * ncentroids * ncodebooks
                + 1.2 * M * ncodebooks * ncentroids)
    elif method.startswith("Exact"):
        return 5500 * D * M


def evalNMSEroot(snr_: list, nmse_: list, refNMSE: float) -> float:
    """
    polyfit lg(nmse_) - lg(refNMSE) = f(snr_), and get the root of snr_
    """
    p = np.polyfit(snr_, np.log10(nmse_) - np.log10(refNMSE), deg=7)
    roots = np.roots(p)
    ammSNRs_ = roots[np.where(np.logical_and(np.isreal(roots), abs(roots) < 15))]
    assert len(ammSNRs_) == 1, ammSNRs_
    return np.real(ammSNRs_[-1])


colors_ = ["r", "g", "b", "orange", "c", "m", "steelblue", "grey", "brown", "k"]
cbColors_ = {
    1: "C0",
    2: "C1",
    4: "C2",
    8: "C3",
}
markers_ = ["1", "v", "o", "s", "*", "+", "x", "D", "|", "^", "2", ]
# markers_ = ["o", "x", "s", "d", "+", "*", "v", "^", ">", "D"]

rms = 75
D = 256

fig, axLoss = plt.subplots(ncols=1, figsize=(5, 4.5))
if type(axLoss) == list:
    axLoss = axLoss[0]
plt.rcParams["font.sans-serif"] = ["Sarasa Mono SC Nerd"]
axLoss.set_title(r"$rms=%se^{-9}$时替换 IDFT 后信道估计损失与复杂度关系" % rms)
axLoss.set_xlabel(r"复杂度占精确乘法比例（%）")
axLoss.set_ylabel(r"$NMSE$ 为 $1e^{-2}$ 时的 $SNR$ 损失（dB）")
axLoss.xaxis.set_tick_params(direction='in', which='both')  # 刻度线向内
axLoss.yaxis.set_tick_params(direction='in', which='both')
axLoss.grid(True, which="major", ls="--", color="grey")
axLoss.grid(True, which="minor", ls="--")
axLoss.set_xscale("log")

path = "./dft/"
files_ = os.listdir(path)
files_.sort(reverse=True)
colorCnt = 7
exactSNR = 0
maxLoss = 0

plots_ = {}

for file in files_:
    match = None
    try:
        # if match == None:
        #     match = re.match(r"\+[0-9][a-z,A-Z].*?\.csv", file)
        # if match == None:
        #     match = re.match(r"\+\+\+[0-9][a-z,A-Z].*?\.csv", file)
        if match == None:
            match = re.match(r"\+rms%s\+[0-9][a-z,A-Z].*?\.csv" % rms, file)
        if match != None:

            fin = open(path + file, "r")
            lines_ = fin.readlines()

            ncodebooks = int(lines_[13].split(",")[1].strip())
            ncentroids = int(lines_[14].split(",")[1].strip())
            nbits = lines_[16].split(",")[1].strip()
            method = lines_[18].split(",")[1].strip()
            if method.startswith("Mithral"):
                method = "MAD-(K=" + str(ncentroids) + " C=" + str(ncodebooks) + ")" + method[7:]
            curColor = colors_[colorCnt]
            curLine = "-"
            if method.startswith("Exact"):
                curColor = "k"
                curLine = "-."
            if method.startswith("SQ"):
                curLine = "-."
            if method.startswith("PQ"):
                curLine = ":"
                continue
            if int(nbits) <= 8 and method.startswith("MAD"):
                continue
            # if (int(nbits) <= 8 or int(nbits) > 16) and method.startswith("MAD"):
            #     continue
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

            if method.startswith("Exact"):
                exactSNR = evalNMSEroot(snr_, rawNMSE_, 1e-2)
            if method.startswith("MAD"):
                method += "-"+nbits+"bits"
                ammSNR = evalNMSEroot(snr_, rawNMSE_, 1e-2)
                loss = ammSNR - exactSNR
                if loss > maxLoss:
                    maxLoss = loss
                comp = complexity(method, ncodebooks, ncentroids) / (5500 * 256 * 16) * 100
                if D // ncodebooks not in plots_:
                    plots_[D//ncodebooks] = []
                plots_[D//ncodebooks].append((comp, loss, int(np.log2(ncentroids))))
                # axLoss.scatter(comp, loss, c=cbColors_[ncodebooks],
                #                marker=markers_[colorCnt], label=method)
                colorCnt -= 1
    except Exception as e:
        print(file + ": ", e)

cnt = 0
# plots[x][0]: (comp, loss, int(np.log2(ncentroids)))
for x in plots_.keys():
    plots_[x].sort()
    x_ = []
    y_ = []
    i = 0
    while i < len(plots_[x]):
        if i > 0 and plots_[x][i-1][0] == plots_[x][i][0] and plots_[x][i-1][1] < plots_[x][i][1]:
            plots_[x].pop(i)
        else:
            x_.append(plots_[x][i][0])
            y_.append(plots_[x][i][1])
            axLoss.text(plots_[x][i][0]+0.1, plots_[x][i][1]+0.1, plots_[x][i][2], fontsize=10, color=cbColors_[x])
            i += 1
    axLoss.plot(x_, y_, c=cbColors_[x], label=r"VQ(%d, $n_e$)" % x, marker=markers_[cnt])
    cnt += 1
# for plot in plots_:
#     axLoss.plot(*zip(*plots_[plot]), c=cbColors_[plot], label="VQ()")
axLoss.legend()
axLoss.set_ylim(top=np.ceil(maxLoss), bottom=0)
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0, wspace=0)
# plt.margins(0, 0)
# axLoss.margins(0, 0)
# plt.margins(0, 0)
plt.show()
