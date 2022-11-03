'''
@file plotDFT.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief https://github.com/originlab/Python-Samples
@date 2022-07-06 15:07:45
@modified: 2022-11-01 16:37:13
'''

import os
import re
import originpro as op
import matplotlib as plt


# Very useful, especially during development, when you are
# liable to have a few uncaught exceptions.
# Ensures that the Origin instance gets shut down properly.
# Note: only applicable to external Python.
import sys
def origin_shutdown_exception_hook(exctype, value, traceback):
    '''Ensures Origin gets shut down if an uncaught exception'''
    op.exit()
    sys.__excepthook__(exctype, value, traceback)


if op and op.oext:
    sys.excepthook = origin_shutdown_exception_hook

# Set Origin instance visibility.
# Important for only external Python.
# Should not be used with embedded Python. 
if op.oext:
    op.set_show(True)

src_opju = os.getcwd() + "\\dft-diffPG.opju"
if not op.open(file = src_opju, readonly=False):
    print(src_opju + " not found!")
    exit(-1)

# Simple syntax to find a worksheet.
sheet_BER = op.find_sheet('w', '[Book1]Sheet1')
sheet_NMSE = op.find_sheet('w', '[Book2]Sheet1')

graph_BER = op.find_graph("Graph1")[0]
graph_NMSE = op.find_graph("Graph2")[0]

SNR_ = sheet_BER.to_list(0)

colors_ = ["r", "g", "b", "orange", "c", "m", "steelblue", "k", "y"]
markers_ = [1, 2, 3, 4, 5, 18, 19, 6, 7] # https://www.originlab.com/doc/LabTalk/ref/List-of-Symbol-Shapes

for plot in graph_BER.plot_list():

    graph_BER.remove_plot(plot)

for plot in graph_NMSE.plot_list():
    graph_NMSE.remove_plot(plot)

path = "./results/"

files_ = os.listdir(path)
files_.sort()
colorCnt = 0
for file in files_:
    match = None
    try:
        if match == None:
            match = re.match(r"\+\+[0-9][a-z,A-Z].*?\.csv", file)

        if match != None:

            fin = open(path + file, "r")
            lines_ = fin.readlines()

            method = lines_[15].split(",")[1].strip()
            curColor = colors_[colorCnt]
            curLine = "-"
            
            snr_ = []
            ber_ = []
            rawNMSE_ = []
            for line in lines_[17:]:
                try:
                    snr_.append(float(line.split(",")[0]))
                    if float(line.split(",")[1]) == 0:
                        ber_.append(None)
                    else:
                        ber_.append(float(line.split(",")[1]))
                    rawNMSE_.append(float(line.split(",")[6]))
                except:
                    break
            assert SNR_[:len(snr_)] == snr_, "SNR not matched!"
            sheet_BER.from_list(colorCnt + 1, ber_, lname=method)
            sheet_NMSE.from_list(colorCnt + 1, rawNMSE_, lname=method)
            for sheet, graph in [(sheet_BER, graph_BER), (sheet_NMSE, graph_NMSE)]:
                plot = graph.add_plot(sheet, coly=colorCnt+1, colx=0, type="y")
                plot.colormap = "Color4Line"
                plot.symbol_size = 12
                plot.symbol_kind = markers_[colorCnt]
                plot.color = plt.colors.to_hex(colors_[colorCnt])
                plot.set_int("line.width", 3)
                # https://www.originlab.com/doc/LabTalk/ref/List-of-Line-Styles
                plot.set_int("line.type", 1) # 从1开始

                if method.startswith("Exact"):
                    plot.color = "#000000"
                    plot.set_int("line.type", 8)

                if method.startswith("PQ"):
                    plot.set_int("line.type", 3)

            colorCnt += 1
    except Exception as e:
        print(file + ": ", e)

for graph in [graph_BER, graph_NMSE]:
    graph.rescale()
    graph.set_xlim(begin=-10, end=2.5)
    # graph.set_ylim(end=1)


op.save(src_opju)

# Exit running instance of Origin.
# Required for external Python but don't use with embedded Python.
# if op.oext:
#     op.exit()