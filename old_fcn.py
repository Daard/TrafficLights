from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Lambda, Reshape, Flatten, Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras import backend as K
import tensorflow as tf

LOGITS = "logits"

"""old fcn model, the loss was about 0.3 after 5 epochs, I used syntectic dataset for training,
 but it is very unstable,
 did not manage to reduce loss below 0.5 in the next training process after 12 epochs, maybe it is caused by dataset randomness"""
def old_fcn(num_classes=2, learning=True):
    K.set_learning_phase(learning)
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    model = Sequential()
    pad = 'same'
    act = 'relu'
    kernel_size = [5, 5, 5]
    model.add(Conv2D(5, kernel_size[0], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(10, kernel_size[0], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(25, kernel_size[1], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(35, kernel_size[2], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(45, kernel_size[2], 1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(Conv2D(num_classes, 1, 1, padding=pad, kernel_regularizer=reg))
    model.add(Conv2DTranspose(num_classes, kernel_size=kernel_size[2], strides=1, activation=act, padding=pad))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(num_classes, kernel_size=kernel_size[1], strides=1, activation=act, padding=pad, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(num_classes, kernel_size=kernel_size[0], strides=1, activation=act, padding=pad, kernel_regularizer=reg,
                              name=LOGITS))
    model.add(Reshape((-1, num_classes)))
    return model