import tensorflow as tf
from tensorflow.keras import Input, Model

old_model = tf.keras.models.load_model('model.h5')

inp = Input(shape=old_model.input_shape[1:])  # dùng đúng shape gốc (loại bỏ dim batch)
x = inp

for layer in old_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer) or isinstance(layer, tf.keras.layers.Dropout):
        continue
    new_layer = layer.__class__.from_config(layer.get_config())
    x = new_layer(x)
    if layer.get_weights():
        new_layer.set_weights(layer.get_weights())

new_model = Model(inputs=inp, outputs=x)
new_model.save("model-nodropout.h5")