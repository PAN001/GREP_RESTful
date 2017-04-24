import tensorflow as tf
from LSTM import *

sess=tf.Session()
lstm=LSTM(5,sess)
keys=[v.name for v in tf.get_collection(tf.GraphKeys.VARIABLES)]

init = tf.initialize_all_variables()
sess.run(init)

lstm.inference([])
keys=[v.name for v in tf.get_collection(tf.GraphKeys.VARIABLES)]