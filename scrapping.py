
import pandas as pd
import os, random
import cv2
import re
from sklearn.utils import shuffle
import numpy as np
from typing import *
from utils import read_data
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical


"""you need to load road images into road directory, try to use p12 udacity project data"""
def random_road():
    dir = os.listdir('./road')
    file = random.choice(dir)
    img = cv2.imread('./road/' + file)
    return img[75:375, 200:1000]


"""dowload TL dataset from http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/"""
def crop_lights(x_train: pd.DataFrame, ind: int) -> Any:
    # Crop
    x_train = x_train.values
    file = x_train[ind, 0]
    p = re.compile('dayClip\d+')
    span = p.search(file).span()
    clip = file[span[0]: span[1]]
    formatted = file.replace(clip, clip + "/frames/" + clip)
    img = cv2.imread("./" + formatted)
    x1, y1, x2, y2 = x_train[ind, 1], x_train[ind, 2], x_train[ind, 3], x_train[ind, 4]
    cropped = img[y1:y2, x1:x2]
    # tried to add different TL sizes
    sizes = [(150, 200), (100, 150), (50, 100), (25, 50)]
    return cv2.resize(cropped, random.choice(sizes))

"""one hot labels for cnn and classification"""
def one_hot(batch_samples):
    mapping = {'stop': 0, 'warning': 1, 'go': 2, 'stopLeft': 3, 'warningLeft': 4, 'goLeft': 5}
    labels = batch_samples['Annotation tag'].apply(lambda x: mapping[x]).values
    y_train = to_categorical(labels, num_classes=6)
    return y_train


"""image-like labels for TL detection"""
def img_labels(i_shape, r_shape, x, y):
    # create one-hot image-shape array of labels for a picture
    label = np.ndarray((r_shape[0], r_shape[1], 2))
    label[:, :, :] = [1, 0]
    label[y:y + i_shape[0], x:x + i_shape[1], :] = [0, 1]
    label_img = cv2.resize(label, (400, 200))
    return np.reshape(label_img, (-1, 2))

"""the sizes of images must be setted according to input shape of NN"""
def synt_generator(samples: pd.DataFrame, type='img'):
    def inner(batch_size: int, infinite=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        nonlocal samples
        num_samples = len(samples)
        samples = shuffle(samples)
        while 1:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                labels = []
                for ind in range(len(batch_samples)):
                    image = crop_lights(batch_samples, ind)
                    road = random_road()
                    r_shape, i_shape = road.shape, image.shape
                    max_y, max_x = r_shape[0] - i_shape[0], r_shape[1] - i_shape[1]
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    road[y:y+i_shape[0], x:x+i_shape[1]] = image
                    blurred = cv2.GaussianBlur(road, (5, 5), 0)
                    cvt = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
                    #!!! dont forget to change image shape if you want to try new NN architecture
                    images.append(cv2.resize(cvt, ((400, 200))))
                    if type == 'img':
                        # for detecting
                        label = img_labels(i_shape, r_shape, x, y)
                    else:
                        # for classification
                        label = one_hot(batch_samples)
                    labels.append(label)
                # I don't why, but with very big arrays numpy shape begins producing unstable results (32, 400, 400, 2) or (32, ),
                # I was trying to solve this problem, found some stackoveroflow topics, but did not manage to solve it
                # Thus, do not set very big images
                x_train = np.array(images)
                y_train = np.array(labels)
                yield x_train, y_train
            if not infinite:
                break

    return inner


"""images from traffic light dataset"""
def real_gen(samples: pd.DataFrame):

    def pre_process(x_train: pd.DataFrame, ind: int):

        def labels(shape: Tuple, target_shape: Tuple):
            # create one-hot image-shape array of labels for a picture
            array = np.ndarray((shape[0], shape[1], 2))
            array[:, :, :] = [1, 0]
            array[y1:y2 + 1, x1:x2 + 1, :] = [0, 1]
            image = cv2.resize(array, (target_shape[1], target_shape[0]))
            return np.reshape(image, (-1, 2))

        # Read image from data
        x_train = x_train.as_matrix()
        file = x_train[ind, 0]
        x1, y1, x2, y2 = x_train[ind, 1], x_train[ind, 2], x_train[ind, 3], x_train[ind, 4]
        p = re.compile('dayClip\d+')
        span = p.search(file).span()
        clip = file[span[0]: span[1]]
        formatted = file.replace(clip, clip + "/frames/" + clip)
        img = cv2.imread("./" + formatted)
        shape = img.shape
        # Crop
        cropped = img[0:shape[0] // 2]
        # Resize
        resized = cv2.resize(cropped, (400, 200), interpolation=cv2.INTER_AREA)
        # Blur
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        # Convert color space
        final_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
        return final_image, labels(cropped.shape, resized.shape)

    def inner(batch_size: int, infinite=False):
        num_samples = len(samples)
        while 1:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                labels = []
                for ind in range(len(batch_samples)):
                    image, image_labels = pre_process(batch_samples, ind)
                    images.append(image)
                    labels.append(image_labels)
                x_train = np.array(images)
                y_train = np.array(labels)
                yield x_train, y_train
            if not infinite:
                break
    return inner


"""use this method, if ypu you want to check your generator outputs, """
def show(images, labels):
    for image, label in zip(images, labels):
        bgr = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', gray)
        cv2.waitKey(0)
        label_img = np.reshape(label, (200, 400, 2))
        cv2.imshow('label', 0.5 * np.argmax(label_img, axis=2))
        cv2.waitKey(0)


if __name__ == "__main__":
    data = read_data(6)
    images, labels = next(synt_generator(data, 'img')(15))
    print(labels.shape)

    # show(images, labels)
    # print(images.shape)
    # show(images, labels)
    # images1, labels1 = next(generatorrrr(data)(5))


