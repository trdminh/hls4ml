import os
import numpy as np
import tensorflow as tf
# Quan trọng: Đảm bảo bạn đang ở trong môi trường conda vitis-ai-tensorflow2
from tensorflow_model_optimization.quantization.keras import vitis_quantize

def quantize_my_model():
    # 1. Load dữ liệu thật của bạn
    # Giả sử X có shape (N, 2, 138) và Y là nhãn (N,) hoặc (N, 6)
    X = np.load("X_train.npy").astype("float32")
    Y = np.load("Y_train.npy")

    if len(Y.shape) == 1 or Y.shape[1] == 1:
        Y = tf.keras.utils.to_categorical(Y, num_classes=6)

    if not os.path.exists("model.h5"):
        print("Lỗi: Không tìm thấy file model.h5")
        return

    model = tf.keras.models.load_model("model.h5")
    print("--- Original Model Summary ---")
    model.summary()

    model.compile(loss='categorical_crossentropy', metrics=["accuracy"])
    score = model.evaluate(X[:500], Y[:500], verbose=0) # Test thử trên 500 mẫu
    print(f"Original model accuracy: {score[1]:.4f}")

    quantizer = vitis_quantize.VitisQuantizer(model)

    quantized_model = quantizer.quantize_model(
        calib_dataset=X[0:1024],   
        weight_bit=8,
        activation_bit=8
    )

    quantized_model.save("model_quant.h5")
    print("Đã lưu model_quant.h5 thành công!")

    quantized_model.compile(loss='categorical_crossentropy', metrics=["accuracy"])
    q_score = quantized_model.evaluate(X[:500], Y[:500], verbose=0)
    print(f"Quantized model accuracy: {q_score[1]:.4f}")

if __name__ == "__main__":
    quantize_my_model()