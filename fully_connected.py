
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.models import Sequential, load_model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Lambda, Reshape, Flatten, Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.contrib.keras.python.keras import optimizers
from sklearn.model_selection import train_test_split
import warnings
from distutils.version import LooseVersion
from scrapping import *
from utils import read_data
from tensorflow.contrib.keras import backend as K
from scrapping import show
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint
import h5py
from utils import timeit

LOGITS = "logits"
RESHAPED = "reshaped"

assert LooseVersion(tf.__version__) == LooseVersion('1.3.0'), 'Please use TensorFlow version 1.3!  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



"""try different approach, but stuck here"""
def fcn(num_classes=2, learning=True) -> Sequential:
    K.set_learning_phase(learning)
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    kernel_size = 3
    pad = 'same'
    act2 = 'relu'
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(200, 400, 3)))
    model.add(Conv2D(32, kernel_size, 1, padding=pad, activation=act2, kernel_regularizer=reg))
    model.add(Conv2D(32, kernel_size, 1, padding=pad, activation=act2, kernel_regularizer=reg))
    model.add(MaxPooling2D((2, 2), 2))
    model.add(Conv2D(64, kernel_size, 1, padding=pad, activation=act2, kernel_regularizer=reg))
    model.add(Conv2D(64, kernel_size, 1, padding=pad, activation=act2, kernel_regularizer=reg))
    model.add(MaxPooling2D((2, 2), 2))
    model.add(Conv2D(128, kernel_size, 1, padding=pad, activation=act2, kernel_regularizer=reg))
    model.add(Conv2D(128, kernel_size, 1, padding=pad, activation=act2, kernel_regularizer=reg))
    model.add(MaxPooling2D((2, 2), 2))
    model.add(Conv2D(num_classes, 1, 1, padding=pad, kernel_regularizer=reg))
    model.add(Conv2DTranspose(num_classes, kernel_size, strides=2, activation=act2, padding=pad))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(num_classes, kernel_size, strides=2, activation=act2, padding=pad))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(num_classes, kernel_size, strides=2, activation=act2, padding=pad,
                              name=LOGITS))
    model.add(Reshape((-1, num_classes)))
    return model


"""tried to use cnn for whole image classification, loss was 0.7 after 7 epochs"""
def cnn(keep_prob=0.5, input_shape=(200, 400, 3)):
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Conv2D(filters=15, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
    model.add(Conv2D(filters=25, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
    model.add(Conv2D(filters=35, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
    model.add(Flatten())
    model.add(Dense(50))
    if keep_prob < 1:
        model.add(Dropout(keep_prob))
    model.add(Dense(10))
    if keep_prob < 1:
        model.add(Dropout(keep_prob))
        model.add(Dense(6, activation='softmax', name=LOGITS))
    return model


def compile(model: Sequential, train_samples: pd.DataFrame, validation_samples:pd.DataFrame, gen, type='img'):

    # model.add(Reshape((-1, num_classes), name=RESHAPED))
    size = 5
    steps_per_epoch = len(train_samples) // size
    validation_steps = len(validation_samples) // size
    train_generator = gen(train_samples, type)(size, infinite=True)
    validation_generator = gen(validation_samples, type)(size, infinite=True)

    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    # checkpoint
    # filepath = "weights-improvement-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='epoch', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    history_object = model.fit_generator(train_generator,
                                         validation_data=validation_generator,
                                         epochs=5, callbacks=None,
                                         validation_steps=validation_steps,
                                         steps_per_epoch=steps_per_epoch)

    model.save_weights('fcn_weights_f2.h5')
    # model.save('fcn_model.h5')

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])

    print('Validation Loss')
    print(history_object.history['val_loss'])


"""change file and model layers for check"""
def predict(path, imgs):
    K.set_learning_phase(False)
    model = fcn(learning=False)
    model.load_weights(path)
    print(model.summary())

    @timeit
    def inner(imgs):
        return model.predict(imgs)
    return inner(imgs)


def train(builder, type='img'):
    data = read_data(13)
    print(data.index)
    print(len(data))
    train_samples, validation_samples = train_test_split(data, test_size=0.2)
    model = builder()
    print(model.summary())
    compile(model, train_samples, validation_samples, synt_generator, type)


def check(path):
    sim_list = os.listdir('./sim')
    images = []
    for file in sim_list:
        img = cv2.imread('./sim/' + file)
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        resized = cv2.resize(cvt, (400, 200))
        images.append(resized)
        # cv2.imshow("check", resized)
        # cv2.waitKey(0)

    labels = predict(path, np.array(images))
    show(images, labels)


def pre_trained():
    model = fcn(learning=True)
    #Don't use this model
    model.load_weights('./fcn_weights_final.h5')
    return model


if __name__ == "__main__":
    # train(fcn)
    # train(pre_trained)

    data = read_data(13)
    imgs, _ = next(synt_generator(data)(15))
    # USe this model
    labels = predict('./fcn_weights_f2.h5', imgs)
    show(imgs, labels)


    # check('./fcn_weights_f2.h5')












