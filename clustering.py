import json, os
import random
import cv2
import numpy as np
import pandas as pd

import umap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.externals import joblib

if __name__=="__main__":
    features = np.load('./models/features.npy')
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
    l_list = [l_dict[data[idx][1]] for idx in range(len(data))]
    with open('./models/label_num.csv', 'w', encoding="utf8") as f:
        for idx in range(len(data)):
            f.write('%d,%d\n'%(idx, l_dict[data[idx][1]]))
    l_list = np.asarray(l_list)

    reducer = umap.UMAP(n_neighbors=10, n_components=8, metric='euclidean', random_state=10)
    features1 = reducer.fit_transform(features)
    # モデルを保存
    filename = './models/e_umap_model.sav'
    joblib.dump(reducer, filename)

    reducer = umap.UMAP(n_neighbors=10, n_components=8, metric='cosine', random_state=10)
    features2 = reducer.fit_transform(features)
    # モデルを保存
    filename = './models/c_umap_model.sav'
    joblib.dump(reducer, filename)

    reducer = PCA(n_components=8)
    features3 = reducer.fit_transform(features)
    # モデルを保存
    filename = './models/pca_model.sav'
    joblib.dump(reducer, filename)

    reducer = LDA(n_components=8)
    features4 = reducer.fit_transform(features, l_list)
    # モデルを保存
    filename = './models/lda_model.sav'
    joblib.dump(reducer, filename)

    features = np.concatenate([features1, features2, features3, features4], 1)
    # features = np.concatenate([features1, features3], 1)

    print(features)
    print(len(features))

    np.save('./models/reduced_features.npy', features)

    # clf = RFC(n_estimators=100, max_depth=3, random_state=0)
    # clf = ETC()
    clf = GBC(n_estimators=20)
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

    importance = clf.feature_importances_
    for idx in range(len(importance)):
        print(idx, importance[idx])
    fig = plt.figure()
    plt.bar(range(len(importance)), list(importance))
    plt.show()