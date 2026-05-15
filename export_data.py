import numpy as np
from sklearn.model_selection import train_test_split

dataRead = np.load("X.npy")
labelRead = np.load("Y.npy")


print(dataRead.shape)

labelRead = labelRead[:, 0].astype(np.int32)
print(labelRead.shape)
print(np.unique(labelRead))
x_train, x_test, y_train, y_test = train_test_split(
    dataRead, labelRead, test_size=0.2, random_state=42
)


np.savetxt("hls/X.txt", x_test[:20].reshape(-1), fmt="%.6f")
np.savetxt("hls/Y.txt", y_test[:20], fmt="%d")