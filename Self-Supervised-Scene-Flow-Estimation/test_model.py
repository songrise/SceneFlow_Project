
import tensorflow as tf


ckpt = tf.train.get_checkpoint_state("log_train_pretrained/")
print(ckpt.model_checkpoint_path)

with tf.Sesson() as sess:
    saver = tf.train.importmeta_graph("log_train_pretrained/model.ckpt.meta")
    saver.resore(sess,)