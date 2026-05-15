import tensorflow as tf
from tensorflow.keras import Input, Model

old_model = tf.keras.models.load_model('model.h5')

print(old_model.summary())
