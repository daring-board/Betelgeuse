import cv2, json, os
import requests
import numpy as np
import configparser
from threading import Thread

import tensorflow as tf
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
from keras.layers import Input
from keras.models import Sequential, Model, load_model
from keras.layers.core import Flatten
from keras.applications.vgg16 import VGG16

from sklearn.externals import joblib

app = Flask(__name__)

UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
graph1 = None
model1 = None
reducer1 = None
reducer2 = None
reducer3 = None
reducer4 = None
classifier = None
label_dict = {}

URL = 'http://127.0.0.1:5000'
app.config['MOBILENET_URL'] = URL

def classify_process():
    global model1, graph1
    global reducer1, reducer2, reducer3, reducer4, classifier
    tmp_dict = {row.strip().split(',')[1]: 0 for row in open('./models/label.csv', 'r')}
    print(tmp_dict)
    count = 0
    for key in tmp_dict.keys():
        label_dict[count] = key
        count += 1

    graph1 = tf.get_default_graph()
    with graph1.as_default():
        shape = (224, 224, 3)
        input_tensor = Input(shape=shape)
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        added_layer = Flatten()(base_model.output)
        model1 = Model(inputs=base_model.input, outputs=added_layer)

    reducer1 = joblib.load('./models/e_umap_model.sav')
    reducer2 = joblib.load('./models/c_umap_model.sav')
    reducer3 = joblib.load('./models/pca_model.sav')
    reducer4 = joblib.load('./models/lda_model.sav')
    classifier = joblib.load('./models/randumforest_model.sav')

@app.route('/', methods = ["GET", "POST"])
def root():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        f = request.files['FILE']
        f_path = save_img(f)
        files = {'FILE': (f.filename, open(f_path, 'rb'))}
        response = requests.post(app.config['MOBILENET_URL']+'/predict', files=files)
        pred1 = json.loads(response.content)['data']
        pred2 = predict_core([f_path]).data.decode('utf-8')
        pred2 = json.loads(pred2)['data']
        print(pred1)
        print(pred2)
        result = make_result(pred1, pred2, [f_path])

        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return render_template(
                'index.html',
                filepath=path,
                predict=result[0]
            )

@app.route('/upload/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def make_result(x1, x2, path_list):
    pred = (np.array(x1) + np.array(x2)) / 2
    names = [item.split('/')[-1] for item in path_list]

    result = []
    for idx in range(len(pred)):
        order = pred[idx].argsort()
        cl1 = order[-1]
        cl2 = order[-2]
        item = {
            'name': names[idx],
            'class1': (label_dict[cl1], str(pred[idx][cl1])),
            'class2': (label_dict[cl2], str(pred[idx][cl2])),
        }
        result.append(item)
    print(result)
    return result

def save_img(f):
    stream = f.stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    f_path = UPLOAD_FOLDER+'/'+f.filename
    cv2.imwrite(f_path, img)
    return f_path

def predict_core(path_list):
    global model1, graph1
    global reducer1, reducer2, reducer3, reducer4, classifier
    data = preprocess(path_list)
    names = [item.split('/')[-1] for item in path_list]

    with graph1.as_default():
        features = model1.predict(data)
    print(features)

    features1 = reducer1.transform(features)
    features2 = reducer2.transform(features)
    features3 = reducer3.transform(features)
    features4 = reducer4.transform(features)
    reduced_features = np.concatenate([features1, features2, features3, features4], 1)
    print(reduced_features)

    pred = classifier.predict_proba(reduced_features)
    print(pred)

    return jsonify({
            'status': 'OK',
            'data': pred.tolist()
        })

def preprocess(f_list):
    datas = []
    for f_path in f_list:
        print(f_path)
        img = cv2.imread(f_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        datas.append(img)
    datas = np.asarray(datas)
    return datas

def abortWithInvalidParams(reason, debug={}):
    abort(400, {
        'errorCode': 1,
        'description': 'invalid params',
        'reason': reason,
        'debug': debug,
    })


def abortWithNoItem(reason, debug={}):
    abort(404, {
        'errorCode': 2,
        'description': 'no item',
        'reason': reason,
        'debug': debug,
    })


def abortWithServerError(reason, debug={}):
    abort(500, {
        'errorCode': 3,
        'description': 'server error',
        'reason': reason,
        'debug': debug,
    })

if __name__ == "__main__":
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0',port=5001)
