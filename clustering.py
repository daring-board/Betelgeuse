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
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib

import hdbscan

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

    r_hist = np.load('./models/r_hist.npy')
    reducer = LDA(n_components=2)
    r_hist = reducer.fit_transform(r_hist, y=l_list)
    # モデルを保存
    filename = './models/r_umap_model.sav'
    joblib.dump(reducer, filename)

    g_hist = np.load('./models/g_hist.npy')
    reducer = LDA(n_components=2)
    g_hist = reducer.fit_transform(g_hist, y=l_list)
    # モデルを保存
    filename = './models/g_umap_model.sav'
    joblib.dump(reducer, filename)

    b_hist = np.load('./models/b_hist.npy')
    reducer = LDA(n_components=2)
    b_hist = reducer.fit_transform(b_hist, y=l_list)
    # モデルを保存
    filename = './models/b_umap_model.sav'
    joblib.dump(reducer, filename)

    reducer = umap.UMAP(n_neighbors=5, n_components=6, metric='cosine', random_state=10)
    features = reducer.fit_transform(features, y=l_list)
    # モデルを保存
    filename = './models/c_umap_model.sav'
    joblib.dump(reducer, filename)

    features = np.concatenate([features, r_hist, g_hist, b_hist], 1)

    print(features)
    print(len(features))

    np.save('./models/reduced_features.npy', features)

    # clf = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    clf = GBC()
    clf.fit(features, l_list)

    # モデルを保存
    filename = './models/randumforest_model.sav'
    joblib.dump(clf, filename)

    # pred = hdbscan.approximate_predict(clf, features)
    pred = clf.predict(features)
    # ラベル変換
    # filename = './models/convert_label.csv'
    # con_label = {idx: -1 for idx in l_dict.values()}
    # for idx in range(len(pred[0])):
    #     if pred[0][idx] == -1: continue
    #     if con_label[pred[0][idx]] != -1: continue 
    #     con_label[pred[0][idx]] = l_list[idx]
    # print(con_label)
    # with open(filename, 'w', encoding='utf8') as f:
    #     for key, value in con_label.items():
    #         f.write('%d,%d\n'%(key, value))

    # for idx in range(len(pred[0])):
    #     print(l_list[idx], pred[0][idx], pred[1][idx])
 
    prob = clf.predict_proba(features)
    for idx in range(len(prob)):
        print(pred[idx], prob[idx])

    importance = clf.feature_importances_
    for idx in range(len(importance)):
        print(idx, importance[idx])
    fig = plt.figure()
    plt.bar(range(len(importance)), list(importance))
    plt.show()