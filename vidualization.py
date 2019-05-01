import json, os
import random
import cv2
import numpy as np
import pandas as pd

import umap
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.externals import joblib

if __name__=="__main__":
    dim = 3

    features = np.load('./models/features.npy')
    reducer = umap.UMAP(n_neighbors=5, n_components=dim, metric='euclidean', random_state=10)
    # reducer = PCA(n_components=dim)
    features = reducer.fit_transform(features)

    # モデルを保存
    filename = './models/umap_model.sav'
    joblib.dump(reducer, filename)

    print(features)
    print(len(features))
    
    data = [row.strip().split(',') for row in open('./models/label.csv', 'r', encoding='utf8')]
    l_dict = {row[1]: 0 for row in data}
    count = 0
    for key in l_dict.keys():
        l_dict[key] = count
        count += 1
    nums = []
    prev, count = data[0][1], 0
    for row in data:
        if row[1] != prev:
            nums.append(count)
            prev, count = row[1], 1
        else:
            count += 1
    nums.append(count)

    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ["r", "g", "b", "c", "m", "y", "b", "#377eb8"]
        start = 0
        for idx in range(len(nums)):
            end = start + nums[idx]
            print(start, end)
            ax.scatter3D(features[start: end, 0], features[start: end, 1], features[start: end, 2], c=colors[idx])
            start = end
        plt.show()
    elif dim == 2:
        colors = ["r", "g", "b", "c", "m", "y", "b", "#377eb8"]
        start = 0
        for idx in range(len(nums)):
            end = start + nums[idx]
            print(start, end)
            plt.scatter(features[start: end, 0], features[start: end, 1], c=colors[idx])
            start = end
        plt.show()
    else:
        print('Specifiable dimensions is only 2 or 3')