import os
import numpy as np
from imp import reload
from PIL import Image

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from . import keys
from . import densenet

reload(densenet)

characters = keys.alphabet[:]
characters = characters[0:] + u'╝'
nclass = len(characters)

model_height = 60
method_type = 0

input = Input(shape=(model_height, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

cwd = os.getcwd()
modelPath = os.path.join(cwd, 'densenet/models/weights_densenet.h5')
print("modelPath = ", modelPath)
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)


def decode(pred):
    char_list = []
    percent_list = []

    pred_argmax = pred.argmax(axis=2)
    pred_text = pred_argmax[0]
    len_pred_text = len(pred_text)
    for i in range(len_pred_text):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
            percent_list.append("%.3f" % (pred[0][i][pred_text[i]]))

    return char_list, percent_list


def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / model_height
    width = int(width / scale)
    img = img.resize([width, model_height], Image.ANTIALIAS)
    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    X = img.reshape([1, model_height, width, 1])
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]
    return decode(y_pred)
