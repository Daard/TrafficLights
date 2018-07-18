
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Lambda, Reshape, Flatten, Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras import optimizers
from sklearn.model_selection import train_test_split
import warnings
from distutils.version import LooseVersion
from scrapping import *
from utils import read_data
from tensorflow.contrib.keras import backend as K
from scrapping import show

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
    model = Sequential()
    kernel_size = [5, 5, 5]
    stride = 1

    # model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(200, 250, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(200, 400, 3)))
    pad = 'same'
    act = 'relu'

    #TODO: add max_pooling, try new act functions

    model.add(Conv2D(5, kernel_size[0], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(10, kernel_size[0], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(25, kernel_size[1], 2, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(35, kernel_size[2], 2, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(45, kernel_size[2], 4, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Reshape((5, 10, 2)))
    model.add(Conv2D(num_classes, 1, 1, padding=pad, kernel_regularizer=reg))
    model.add(Conv2DTranspose(num_classes, kernel_size=kernel_size[2], strides=5, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(num_classes, kernel_size=kernel_size[1], strides=4, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(num_classes, kernel_size=kernel_size[0], strides=2, activation='tanh', padding=pad, kernel_regularizer=reg,
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
    model.add(Conv2D(filters=35, kernel_size=5, strides=2, padding='same', activation='relu', kernel_regularizer=reg))
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
    size = 32
    steps_per_epoch = len(train_samples) // size
    validation_steps = len(validation_samples) // size
    train_generator = gen(train_samples, type)(size, infinite=True)
    validation_generator = gen(validation_samples, type)(size, infinite=True)

    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    history_object = model.fit_generator(train_generator,
                                         validation_data=validation_generator,
                                         epochs=10, callbacks=None,
                                         validation_steps=validation_steps,
                                         steps_per_epoch=steps_per_epoch)

    model.save_weights('fully_connected.h5')

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])

    print('Validation Loss')
    print(history_object.history['val_loss'])


"""change file and model layers for check"""
def predict(imgs):
    file='./fcn2.h5'
    model = fcn(learning=False)
    model.load_weights(file)
    print(model.summary())
    return model.predict(imgs)


def train(builder, type):
    data = read_data(13)
    print(data.index)
    print(len(data))
    train_samples, validation_samples = train_test_split(data, test_size=0.2)
    model = builder()
    print(model.summary())
    compile(model, train_samples, validation_samples, synt_generator, type)


def check():
    sim_list = os.listdir('./sim')
    images = []
    for file in sim_list:
        img = cv2.imread('./sim/' + file)
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        resized = cv2.resize(cvt, (250, 200))
        images.append(resized)
        # cv2.imshow("check", resized)
        # cv2.waitKey(0)

    labels = predict(np.array(images))
    show(images, labels)


if __name__ == "__main__":
    train(cnn, 'one_hot')









