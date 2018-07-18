
import time
import tensorflow as tf
from typing import *
import pandas as pd
import random


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def read_data(max=13) -> pd.DataFrame:
    frames = []
    for i in random.sample(range(1, 14), max):
        uri = './dayTraining/dayClip{}/frameAnnotationsBOX.csv'.format(i)
        frame = pd.read_csv(uri, sep=';')
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return data[
        ['Filename', 'Upper left corner X', 'Upper left corner Y',
         'Lower right corner X', 'Lower right corner Y', 'Annotation tag']]


@timeit
def generate_protobuf(graph_name: str):
    tf.reset_default_graph()
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver = tf.train.import_meta_graph(graph_name)
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        tf.train.write_graph(graph_def, './', 'binary.pb', as_text=False)


@timeit
def load_graph(graph_file: str, use_xla=False) -> (tf.Graph, List[tf.Operation]):
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        # print(tf.train.get_checkpoint_state('ckpt').all_model_checkpoint_paths)
        print(graph_file + " operations number:{}".format(n_ops))
    return sess, ops


if __name__ == "__main__":
    # GRAPH = "keras_model.ckpt.meta"
    # generate_protobuf(GRAPH)
    """python ~/anaconda/pkgs/tensorflow-1.3.0-py35_0/lib/python3.5/site-packages/tensorflow/python/tools/freeze_graph.py \
    --input_graph=binary.pb \
    --input_checkpoint=keras_model.ckpt \
    --input_binary=true \
    --output_graph=frozen.pb \
    --output_node_names=logits/Softmax"""
    sess, ops = load_graph("./frozen.pb")
    print([op.name for op in ops])


    """python ~/anaconda/pkgs/tensorflow-1.3.0-py35_0/lib/python3.5/site-packages/tensorflow/python/tools/optimize_for_inference.py \
    --input=frozen.pb \
    --output=optimized.pb \
    --frozen_graph=True \
    --input_names=lambda_1_input \
    --output_names=logits/Softmax"""
    sess, ops = load_graph("./optimized.pb")
    print([op.name for op in ops])

    """python ~/anaconda/pkgs/tensorflow-1.3.0-py35_0/lib/python3.5/site-packages/tensorflow/python/tools/transform_graph.py \
    --in_graph=frozen.pb \
    --out_graph=eightbit.pb \
    --inputs=lambda_1_input:0 \
    --outputs=logits/Softmax \
    --transforms='
    add_default_attributes
    remove_nodes(op=Identity, op=CheckNumerics)
    fold_constants(ignore_errors=true)
    fold_batch_norms
    fold_old_batch_norms
    fuse_resize_and_conv
    quantize_weights
    quantize_nodes
    strip_unused_nodes
    sort_by_execution_order'"""




