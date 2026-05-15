import os
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
X_test = np.load("x_test.npy")          
Y_test = np.load("y_test.npy")         
Y_test = Y_test[:, 0]                  
Y_test = Y_test - 1                    
X_test = (X_test - np.mean(X_test)) / (np.std(X_test) + 1e-8)
Y_test = tf.keras.utils.to_categorical(Y_test, 6)
print(X_test[0])
print(Y_test[0])