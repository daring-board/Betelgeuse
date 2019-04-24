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
    features = np.load('./models/features.npy')
    # reducer = umap.UMAP(n_neighbors=5, n_components=32, metric='cosine', random_state=10)
    reducer = PCA(n_components=32)
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

    np.save('./models/reduced_features.npy', features)
    l_list = [l_dict[data[idx][1]] for idx in range(len(data))]
    with open('./models/label_num.csv', 'w', encoding="utf8") as f:
        for idx in range(len(data)):
            f.write('%d,%d\n'%(idx, l_dict[data[idx][1]]))
    l_list = np.asarray(l_list)

    # clf = RFC(n_estimators=100, max_depth=3, random_state=0)
    clf = ETC()
    clf.fit(features, l_list)

    # モデルを保存
    filename = './models/randumforest_model.sav'
    joblib.dump(clf, filename)
    
    pred = clf.predict(features)
    for idx in range(len(pred)):
        print(pred[idx], l_list[idx])

    prob = clf.predict_proba(features)
    for idx in range(len(prob)):
        print(pred[idx], prob[idx])