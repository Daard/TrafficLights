import cv2
import numpy as np
import pandas as pd
import re
import warnings
from distutils.version import LooseVersion

import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Lambda,  Dropout
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical

from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import timeit, read_data
import time

from tensorflow.contrib.keras.python.keras.callbacks import TensorBoard as keras_tb
from typing import *

MODEL = "./model.ckpt"
GRAPH = MODEL + ".meta"
LOGITS = "logits"


"""use this link to download TL dataset"""
# http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/

# Check TensorFlow Version
assert LooseVersion(tf.__version__) == LooseVersion('1.3.0'), 'Please use TensorFlow version 1.3!  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def pre_process(x_train: pd.DataFrame, ind: int, crop=True) -> Any:
    # Crop
    x_train = x_train.as_matrix()
    file = x_train[ind, 0]
    p = re.compile('dayClip\d+')
    span = p.search(file).span()
    clip = file[span[0]: span[1]]
    formatted = file.replace(clip, clip + "/frames/" + clip)
    img = cv2.imread("./" + formatted)
    if crop:
        x1, y1, x2, y2 = x_train[ind, 1], x_train[ind, 2], x_train[ind, 3], x_train[ind, 4]
        cropped = img[y1:y2, x1:x2]
        # Resize
        resized = cv2.resize(cropped, (52, 114), interpolation=cv2.INTER_AREA)
    else:
        shape = img.shape
        # Crop
        cropped = img[0:shape[0] // 2]
        resized = cv2.resize(cropped, (300, 200), interpolation=cv2.INTER_AREA)
    # Blur
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    final_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    return final_image


def generator(samples: pd.DataFrame, crop=True):
    def inner(batch_size: int, infinite=False):
        nonlocal samples
        mapping = {'stop': 0, 'warning': 1, 'go': 2, 'stopLeft': 3, 'warningLeft': 4, 'goLeft': 5}
        num_samples = len(samples)
        samples = shuffle(samples)
        while 1:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                labels = batch_samples['Annotation tag'].apply(lambda x: mapping[x]).values
                for ind in range(len(batch_samples)):
                    image = pre_process(batch_samples, ind, crop)
                    images.append(image)
                x_train = np.array(images)
                # one hot activation
                y_train = to_categorical(labels, num_classes=6)
                yield shuffle(x_train, y_train)
            if not infinite:
                break
    return inner

"""simple cnn for traffic lights classification, works well, but we need to add TL detector method"""
def layers(keep_prob: Any, input_shape=(114, 52, 3), add_softmax=False):
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Conv2D(filters=24, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
    model.add(Conv2D(filters=36, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
    model.add(Flatten())
    model.add(Dense(50))
    if keep_prob < 1:
        model.add(Dropout(keep_prob))
    model.add(Dense(10))
    if keep_prob < 1:
        model.add(Dropout(keep_prob))
    if add_softmax:
        model.add(Dense(6, activation='softmax', name=LOGITS))
    else:
        model.add(Dense(6, name=LOGITS))
    return model


@timeit
def compile_keras(train_samples: pd.DataFrame, validation_samples:pd.DataFrame):
    K.set_learning_phase(True)
    size = 16
    steps_per_epoch = len(train_samples) // size
    validation_steps = len(validation_samples) // size
    train_generator = generator(train_samples, crop=False)(size, infinite=True)
    validation_generator = generator(validation_samples, crop=False)(size, infinite=True)
    model = layers(keep_prob=0.5, input_shape=(200, 300, 3), add_softmax=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    tensorboard = keras_tb(log_dir="./logs/{}".format(time.time()))
    history_object = model.fit_generator(train_generator,
                                         # samples_per_epoch=6 * len(train_samples),
                                         validation_data=validation_generator,
                                         # nb_val_samples=6 * len(validation_samples),
                                         epochs=10, callbacks=[tensorboard],
                                         validation_steps=validation_steps,
                                         steps_per_epoch=steps_per_epoch)
    model.save_weights('model.h5')

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])

    print('Validation Loss')
    print(history_object.history['val_loss'])


if __name__ == "__main__":
    # train_nn(epochs=10, batch_size=256)
    # data = read_data()
    # train_samples, validation_samples = train_test_split(data, test_size=0.2)
    # images, labels = next(generator(train_samples, crop=True)(batch_size=5))
    # for image in images:
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)
    # compile_keras(train_samples, validation_samples)
    model = layers(keep_prob=0.5)
    print(model.summary())