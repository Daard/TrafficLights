from utils import timeit

from tensorflow.contrib.keras.python.keras.models import Model
from model import generator, read_data, layers
import numpy as np
import cv2
from sklearn.metrics import precision_recall_fscore_support as score
import tensorflow as tf
from utils import load_graph


def model_predict(file='./models/my_model_weights.h5'):
    @timeit
    def load():
        model = layers(1, add_softmax=True)
        model.load_weights(file)
        print(model.summary())
        return model
    model = load()
    @timeit
    def inner(imgs):
        return model.predict(imgs)
    return inner


def graph_predict(file='./models/optimized.pb'):
    sess, _ = load_graph(file, use_xla=False)
    @timeit
    def inner(images):
        graph = sess.graph
        logits = graph.get_operation_by_name('logits/Softmax').outputs[0]
        image_input = graph.get_operation_by_name('lambda_1_input').outputs[0]
        init_map = {image_input: images}
        p_labels = sess.run([tf.nn.softmax(logits)], init_map)
        return p_labels[0]
    return inner


@timeit
def run(predict):
    samples = read_data()
    mapping = {'stop': 0, 'warning': 1, 'go': 2, 'stopLeft': 3, 'warningLeft': 4, 'goLeft': 5}
    X, labels = next(generator(samples)(1))
    p_labels = predict(X)
    print(p_labels)
    for img, l, p_l in zip(X, labels, p_labels):
        final_image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        indx = np.argmax(p_l)
        label = list(mapping.keys())[list(mapping.values()).index(indx)]
        cv2.imshow(label, final_image)
        cv2.waitKey(0)


@timeit
def check_metrics(_model: Model):
    samples = read_data()
    batch_size = 100
    gen = generator(samples)
    epoch = 0
    y_true = np.array([])
    y_pred = np.array([])

    while epoch < 10:
        x, labels = next(gen(batch_size))
        p_labels = model_predict()(x)
        y_true = np.append(y_true, np.argmax(labels, axis=1))
        y_pred = np.append(y_pred,np.argmax(p_labels, axis=1))
        print(epoch)
        epoch += 1

    precision, recall, fscore, support = score(y_true, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


if __name__ == "__main__":
    # model = load_model('model.h5')

    # model = layers(1, add_softmax=True)
    # model.load_weights('./my_model_weights.h5')
    # run_model(model)

    # save keras model to ckpt
    # saver = tf.train.Saver()
    # saver.save(K.get_session(), './keras_model.ckpt')
    # run(graph_predict())
    run(model_predict())
    # run(graph_predict())













