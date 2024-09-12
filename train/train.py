import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
import densenet


img_h = 60
img_w = 200
batch_size = 8
max_label_length = 5
data_test_path = "data_test.txt"
data_train_path = "data_train.txt"


def get_session(gpu_fraction=0.8):

    num_threads = os.environ.get("OMP_NUM_THREADS")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(" ")
        dic[p[0]] = p[1:]
    return dic


class random_uniform_num():
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batch_size):
        r_n = []
        if self.index + batch_size > self.total:
            r_n_1 = self.range[self.index: self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batch_size) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batch_size]
            self.index = self.index + batch_size

        return r_n


def gen(data_file, image_path, batch_size=8, max_label_length=5, imagesize=(60, 200)):
    image_label = readfile(data_file)
    image_file = [i for i, j in image_label.items()]
    x = np.zeros((batch_size, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batch_size, max_label_length]) * 10000
    input_length = np.zeros([batch_size, 1])
    label_length = np.zeros([batch_size, 1])

    r_n = random_uniform_num(len(image_file))
    image_file = np.array(image_file)
    while 1:
        shuffle_image_file = image_file[r_n.get(batch_size)]
        for i, j in enumerate(shuffle_image_file):
            img1 = Image.open(os.path.join(image_path, j)).convert("L")
            img = np.array(img1, "f") / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            str_label = image_label[j]
            label_length[i] = len(str_label)

            if len(str_label) <= 0:
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str_label)] = [int(k) - 1 for k in str_label]

        inputs = {"the_input": x,
                  "the_labels": labels,
                  "input_length": input_length,
                  "label_length": label_length,
                  }
        outputs = {"ctc": np.zeros([batch_size])}
        yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_h, n_class):
    input = Input(shape=(img_h, None, 1), name="the_input")
    y_pred = densenet.dense_cnn(input, n_class)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name="the_labels", shape=[None], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer="adam", metrics=["accuracy"])

    return basemodel, model


if __name__ == "__main__":
    char_set = open("char_std_30.txt", "r", encoding="utf-8").readlines()

    char_set = "".join([ch.strip("\n") for ch in char_set][1:] + ["╝"])
    # print(char_set)
    n_class = len(char_set)
    # print(n_class)

    tf.compat.v1.keras.backend.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, n_class)

    modelPath = "./pretrain_model/keras.h5"
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print("done!")

    train_loader = gen("data_train.txt", "./images", batch_size=batch_size, max_label_length=max_label_length, imagesize=(img_h, img_w))
    test_loader = gen("data_test.txt", "./images", batch_size=batch_size, max_label_length=max_label_length, imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath="./models/weights_densenet-{epoch:02d}-{accuracy:.4f}.h5",
                                 monitor="acc",
                                 save_best_only=False,
                                 save_weights_only=True)

    total_epochs = 10
    test_num_lines = sum(1 for line in open(data_test_path))
    train_num_lines = sum(1 for line in open(data_train_path))

    #lr_schedule = lambda epoch: 0.005 * 0.9**epoch
    #lr_schedule = lambda epoch: 0.0005 * 0.65 ** epoch
    lr_schedule = lambda epoch: 0.0005 * 0.4**epoch

    learning_rate = np.array([lr_schedule(i) for i in range(total_epochs)])
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(monitor="accuracy", patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir="./models/logs", write_graph=True)

    print("---------Start training----------with test number =", test_num_lines, ", train number = ", train_num_lines)
    model.fit(train_loader,
              steps_per_epoch=train_num_lines // batch_size,
              epochs=total_epochs,
              initial_epoch=0,
              validation_data=test_loader,
              validation_steps=test_num_lines // batch_size,
              callbacks=[checkpoint, early_stop, change_lr, tensorboard])

