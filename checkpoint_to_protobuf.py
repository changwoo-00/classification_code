import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import freeze_graph
import os

class ConvertCheckpointToProtobuf(object):
    def __init__(self, checkpoint_path, checkpoint_name):
        self.CHECKPOINT_PATH = checkpoint_path
        self.CHECKPOINT_NAME = checkpoint_name

    def _save_as_pb(self, sess, directory, filename, output_node_name):
        if not os.path.exists(directory):
            os.makedirs(directory)

        ckpt_filepath = os.path.join(directory, filename)

        # Save check point for graph frozen later
        pbtxt_filename = filename + '.pbtxt'
        pbtxt_filepath = os.path.join(directory, pbtxt_filename)
        pb_filepath = os.path.join(directory, filename + '.pb')
        # This will only save the graph but the variables will not be saved.
        # You have to freeze your model first.
        tf.train.write_graph(graph_or_graph_def=sess.graph_def, 
                             logdir=directory, 
                             name=pbtxt_filename, 
                             as_text=True)

        # Freeze graph
        #'InceptionResnetV2/Logits/Predictions'
        freeze_graph.freeze_graph(input_graph=pbtxt_filepath, 
                                  input_saver='', 
                                  input_binary=False, 
                                  input_checkpoint=ckpt_filepath, 
                                  output_node_names=output_node_name,
                                  restore_op_name='save/restore_all', 
                                  filename_tensor_name='save/Const:0', 
                                  output_graph=pb_filepath, 
                                  clear_devices=True, 
                                  initializer_nodes='')

    def release(self, save_path, output_node_name):
        with tf.Session() as sess:
            meta_path = os.path.join(self.CHECKPOINT_PATH, self.CHECKPOINT_NAME + '.meta')
            if not os.path.exists(meta_path):
                return

            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(self.CHECKPOINT_PATH))

            self._save_as_pb(sess, save_path, self.CHECKPOINT_NAME, output_node_name)