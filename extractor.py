import json, os
import random
import cv2
import numpy as np
import pandas as pd

from keras.utils import np_utils, Sequence
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

class DataSequence(Sequence):
    def __init__(self, data_path, label):
        self.batch = 4
        self.data_file_path = data_path
        self.datagen = ImageDataGenerator(
                            rotation_range=30,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.5
                        )
        d_list = os.listdir(self.data_file_path)
        self.f_list = []
        for dir in d_list:
            if dir == 'empty': continue
            for f in os.listdir(self.data_file_path+'/'+dir):
                self.f_list.append(self.data_file_path+'/'+dir+'/'+f)
        self.label = label
        self.length = len(self.f_list)

    def __getitem__(self, idx):
        warp = self.batch
        aug_time = 3
        datas, labels = [], []
        label_dict = self.label

        # for f in random.sample(self.f_list, warp):
        for f in self.f_list[warp * idx: warp * (idx+1)]:
            img = cv2.imread(f)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            datas.append(img)
            label = f.split('/')[2].split('_')[-1]
            labels.append(label_dict[label])
            # Augmentation image
            for num in range(aug_time):
                tmp = self.datagen.random_transform(img)
                datas.append(tmp)
                labels.append(label_dict[label])

        datas = np.asarray(datas)
        labels = pd.DataFrame(labels)
        labels = np_utils.to_categorical(labels, len(label_dict))
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

if __name__=="__main__":
    shape = (224, 224, 3)
    input_tensor = Input(shape=shape)

    '''
    学習済みモデルのロード(base_model)
    '''
    base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=input_tensor)

    '''
    学習用画像のロード
    '''
    label_dict = {}
    count = 0
    for d_name in os.listdir('./train'):
        if d_name == 'empty': continue
        if d_name == '.DS_Store': continue
        d_name = d_name.split('_')[-1]
        label_dict[d_name] = count
        count += 1
    train_gen = DataSequence('./train', label_dict)

    added_layer = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=added_layer)

    model.summary()

    file_all = train_gen.length
    steps = file_all / 4

    '''
    特徴ベクトル抽出
    '''
    features = model.predict_generator(
         train_gen,
         steps=steps,
    )
    np.save('./models/features.npy', features)
    with open('./models/label.csv', 'w', encoding="utf8") as f:
        for idx in range(train_gen.length):
            for n in range(4):
                f.write('%d,%s\n'%(idx, train_gen.f_list[idx].split('/')[-2]))

    '''
    カラーヒストグラム抽出
    '''
    datagen = train_gen.datagen
    r_hists, g_hists, b_hists = [], [], []
    for f in train_gen.f_list:
        img = cv2.imread(f)
        img = cv2.resize(img, (224, 224))
        r_hist = cv2.calcHist([img], [0], None, [256], [0,256])
        g_hist = cv2.calcHist([img], [1], None, [256], [0,256])
        b_hist = cv2.calcHist([img], [2], None, [256], [0,256])
        r_hists.append([item[0] for item in r_hist])
        g_hists.append([item[0] for item in g_hist])
        b_hists.append([item[0] for item in b_hist])
        # Augmentation image
        for num in range(3):
            tmp = datagen.random_transform(img)
            r_hist = cv2.calcHist([tmp], [0], None, [256], [0,256])
            g_hist = cv2.calcHist([tmp], [1], None, [256], [0,256])
            b_hist = cv2.calcHist([tmp], [2], None, [256], [0,256])
            r_hists.append([item[0] for item in r_hist])
            g_hists.append([item[0] for item in g_hist])
            b_hists.append([item[0] for item in b_hist])

    r_hists = np.asarray(r_hists)
    np.save('./models/r_hist.npy', r_hists)
    g_hists = np.asarray(g_hists)
    np.save('./models/g_hist.npy', g_hists)
    b_hists = np.asarray(b_hists)
    np.save('./models/b_hist.npy', b_hists)