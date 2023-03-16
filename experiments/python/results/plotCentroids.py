'''
@file plotCentroids.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-03-08 21:12:21
@modified: 2023-03-08 23:49:16
'''
import matplotlib.pyplot as plt
import numpy as np

titles_ = ["K128C128", "K64C128", "K16C128"]
colors_ = ["g", "purple", "r", "g", "b", "orange", "c", "m", "steelblue", "grey", "brown", "k"]
markers_ = ["o", "x", "s", "d", "+", "*", "v", "^", ">", "D"]
cmaps_ = ["Greens", "Purples", "Reds"]

lim = 0

fig, ax = plt.subplots()
handles_ = []
labels_ = []

for i in range(len(titles_)):
    centroids = np.transpose(np.load(titles_[i] + ".npy"))
    cnts = np.load(titles_[i] + "-cnt" + ".npy")
    lim = max(np.ceil(np.max(abs(centroids))), lim)
    legend = plt.scatter(centroids[0], centroids[1], c=cnts, label=titles_[
        i], marker=markers_[i], cmap=cmaps_[i], alpha=0.7).legend_elements(num=1, fmt="%s" % titles_[i])
    handles_ += legend[0]
    labels_ += legend[1]
    plt.colorbar(label=titles_[i])
plt.grid()
plt.axis("square")
plt.xlim((-lim, lim))
plt.ylim((-lim, lim))
plt.legend(handles=handles_, labels=labels_)
plt.show()
