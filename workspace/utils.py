import os
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

def dump_weights_to_txt_and_bin(model, save_dir="dump_results/dump_results_weights"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dense_idx = 0
    for layer in model.layers:
        params = layer.get_weights()
        
        if len(params) >= 2 and 'dense' in layer.name:
            weights = params[0]
            biases = params[1]
            try:
                w_pos = params[5] if len(params) >= 6 else 7.0
                print(f"Layer {layer.name}: Using Weight Pos = {w_pos}")
            except:
                w_pos = 7.0

            # QUAN TRỌNG: Nhân với 2^pos để chuyển sang số nguyên
            scale_factor = 2**w_pos
            weights_int = np.round(weights * scale_factor).astype(np.int32)
            biases_int = np.round(biases * scale_factor).astype(np.int32)

            base_name = f"quant_dense_{dense_idx}"
            
            # Lưu file .txt (Bây giờ sẽ có số nguyên: 15, -60, 110...)
            np.savetxt(os.path.join(save_dir, f"{base_name}_kernel.txt"), weights_int.flatten(), fmt='%d')
            np.savetxt(os.path.join(save_dir, f"{base_name}_bias.txt"), biases_int.flatten(), fmt='%d')
            
            # Lưu file .bin cho FPGA
            weights_int.astype(np.int8).tofile(os.path.join(save_dir, f"{base_name}_kernel.bin"))
            
            print(f"Saved {base_name} with actual integer values.")
            dense_idx += 1
        else:
            print(f"Skipping layer {layer.name} (No weights to dump)")
class utils():
    def __init__(self, path):
        self.path = path
        X = np.load("X.npy")          
        Y = np.load("Y.npy")         

        Y = Y[:, 0]                  
        Y = Y - 1                    
        X = X // 255
        self.num_classes = 6
        Y = tf.keras.utils.to_categorical(Y, self.num_classes)
        
        X_test = np.load("x_test.npy")          
        Y_test = np.load("y_test.npy")         
        Y_test = Y_test[:, 0]                  
        Y_test = Y_test - 1                    
        X_test = X_test // 255
        Y_test = tf.keras.utils.to_categorical(Y_test, self.num_classes)
        self.x_train = X
        self.y_train = Y
        self.x_test = X_test
        self.y_test = Y_test
        self.input_dim = 276
        self.kernel_size = 3
        self.dims = [128,256,6]
    def NN(self, train=False):
        if train:
            inputs = tf.keras.Input(shape=(self.input_dim,))
            x = tf.keras.layers.Dense(128, activation="relu")(inputs)
            x = tf.keras.layers.Dense(256, activation="relu")(x)
            outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

            model = tf.keras.Model(inputs, outputs)
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            model.fit(self.x_train, self.y_train, epochs=20, batch_size=32, validation_split=0.1)

            score = model.evaluate(self.x_test, self.y_test, verbose=0)
            print("Float accuracy:", score[1])
    
            model.save("model.h5")
        else:
            model = tf.keras.models.load_model("model.h5")

        weights = model.get_weights()
        self.weights = weights[::2]
        self.bias = weights[1::2]

        self.layer_data = []
        data = self.x_test[0].flatten()
        for i in range(3):
            print(np.shape(data), np.shape(weights[2*i]), np.shape(weights[2*i+1]))
            data = data@weights[2*i] + weights[2*i+1]
            if i != 2:
                data[data<0] = 0
            else:
                pass
            self.layer_data.append(data)
        self.res_mnist = data
        return self.weights, self.bias, self.layer_data
    
    def FullyVitisAI(self):
        bias = []
        weight = []
        self.out = []
        test = []

        scales = [256, 128, 128, 128] 
        self.log_scales = [8, 7, 7, 7]
        layers = [276, 128, 64, 32, 6] 

        dense_indices = [0, 2, 4, 6] 

        for i, idx in enumerate(dense_indices):
            w_path = f"dump_results/dump_results_weights/quant_dense_{idx}_kernel.txt"
            b_path = f"dump_results/dump_results_weights/quant_dense_{idx}_bias.txt"
            
            w_data = np.loadtxt(w_path)
            b_data = np.loadtxt(b_path)
            weight.append(w_data.reshape(layers[i], layers[i+1]))
            bias.append(b_data)

        self.weights_int = weight
        self.bias_int = bias

        err = 0

        for i in range(len(self.x_test)):
            data = self.x_test[i].astype(np.int16)
            for j in range(4): 
                z = data @ weight[j] + bias[j]
                test.append(z)
                data = z // scales[j]
                if j != 3: 
                    data = data * (data > 0)
                self.out.append(data)
            res = np.argmax(data)
            if res != np.argmax(self.y_test[i]):
                err += 1
        print(err)
        print("acc {}".format(1-err/len(self.x_test)))
        
        # Check if still in range of int16 (-32,768 to +32,767)

        mins= []
        maxs = []
        for i in range(len(self.x_test)):
            mins.append(np.min(self.out[i]))
            maxs.append(np.max(self.out[i]))

        print(np.min(mins), np.max(maxs))
        
        mins= []
        maxs = []
        for i in range(len(self.x_test)):
            mins.append(np.min(test[i]))
            maxs.append(np.max(test[i]))

        print(np.min(mins), np.max(maxs))

        return self.out, bias, weight, test
    
    def write_model_h(self):
        self.weights_int[2] = np.pad(self.weights_int[2], ((0,0),(0,32)))
        self.weights_int[3] = np.pad(self.weights_int[3], ((0,0),(0,58)))

        x = self.x_test.astype(np.int16)

        w1 = self.weights_int[0].astype(np.int16).flatten()
        w2 = self.weights_int[1].astype(np.int16).flatten()
        w3 = self.weights_int[2].astype(np.int16).flatten()
        w4 = self.weights_int[3].astype(np.int16).flatten()

        b1 = self.bias_int[0].astype(np.int16)
        b2 = self.bias_int[1].astype(np.int16)
        b3 = self.bias_int[2].astype(np.int16)
        b4 = self.bias_int[3].astype(np.int16)

        w = np.concatenate([w1, w2, w3, w4])
        b = np.concatenate([b1, b2, b3, b4])
        res = np.concatenate([
            self.out[0].flatten(),
            self.out[1].flatten(),
            self.out[2].flatten(),
            self.out[3].flatten()
        ]).astype(np.int16)
        
        n_weights1 = np.shape(w1)[0]
        n_weights2 = np.shape(w2)[0]
        n_weights3 = np.shape(w3)[0]
        n_weights4 = np.shape(w4)[0]
        n_weights = np.shape(w)[0]
        n_bias = np.shape(b)[0]
        
        with open(self.path+'/model.h','w') as file:
            file.writelines('/* HW AI HLS, autogenerated File, Tran Doan Minh, {} */ \n \n'.format(datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))
            
            file.writelines('#include <stdint.h> \n')
            file.writelines('\n')
            
            file.writelines('static const int16_t weights1[{}] = \n'.format(n_weights1))
            file.writelines(' {')
            
            for j in range(n_weights1):
                if j == n_weights1 - 1:
                    file.writelines('{}'.format(w1[j]))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(w1[j]))
                    else:
                        file.writelines('{},'.format(w1[j]))
            
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int16_t weights2[{}] = \n'.format(n_weights2))
            file.writelines(' {')
            
            for j in range(n_weights2):
                if j == n_weights2 - 1:
                    file.writelines('{}'.format(w2[j]))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(w2[j]))
                    else:
                        file.writelines('{},'.format(w2[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int16_t weights3[{}] = \n'.format(n_weights3))
            file.writelines(' {')
            
            for j in range(n_weights3):
                if j == n_weights3 - 1:
                    file.writelines('{}'.format(w3[j]))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(w3[j]))
                    else:
                        file.writelines('{},'.format(w3[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int16_t weights4[{}] = \n'.format(n_weights4))
            file.writelines(' {')
            
            for j in range(n_weights4):
                if j == n_weights4 - 1:
                    file.writelines('{}'.format(w4[j]))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(w4[j]))
                    else:
                        file.writelines('{},'.format(w4[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int16_t bias[{}] = \n'.format(n_bias))
            file.writelines('  {')
             
            for j in range(n_bias):
                if j == n_bias -1:
                    file.writelines('{}'.format(b[j]))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(b[j]))
                    else:
                        file.writelines('{},'.format(b[j]))     
            file.writelines('  };\n')
            file.writelines('\n')
            n_input = self.input_dim   

            file.writelines('static const int16_t im[{}] = \n'.format(n_input))
            file.writelines('  {')
            print(x[0])
            print(self.y_test[0])
            for j in range(n_input):
                val = int(x[0][j])   

                if j == n_input - 1:
                    file.writelines('{}'.format(val))
                else:
                    if j % 16 == 0:   
                        file.writelines('\n        {},'.format(val))
                    else:
                        file.writelines('{},'.format(val))

            file.writelines('  };\n')
            file.writelines('\n')
            n_res = len(res)

            file.writelines('static const int16_t res_layers[{}] = \n'.format(n_res))
            file.writelines('  {')

            for j in range(n_res):
                if j == n_res - 1:
                    file.writelines('{}'.format(int(res[j])))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(int(res[j])))
                    else:
                        file.writelines('{},'.format(int(res[j])))

            file.writelines('  };\n')
            file.writelines('\n')
            file.writelines('static const int16_t scales[{}] = \n'.format(len(self.log_scales)))
            file.writelines('  {')
             
            for j in range(4):
                if j == np.sum(np.shape(self.dims)[0]) -1:
                    file.writelines('{}'.format((self.log_scales[j])))
                else:
                    if j % self.input_dim == 0:
                        file.writelines('\n        {},'.format(self.log_scales[j]))
                    else:
                        file.writelines('{},'.format(self.log_scales[j]))
            file.writelines('};\n')
            file.writelines('\n')
    def pynq_dpu(self):
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        model = tf.keras.models.load_model("model.h5")
        quantizer = vitis_quantize.VitisQuantizer(model)
        quantized_model = quantizer.quantize_model(calib_dataset = self.x_test[1:1024], weight_bit=16, activation_bit=16)
        
        quantized_model.compile(loss='categorical_crossentropy', metrics=["accuracy"])
        score = quantized_model.evaluate(self.x_test, self.y_test,  verbose=0, batch_size=32)
        print("Quantized acc: ",score)
        quantized_model.save('model_quant.h5')
        
        os.system("vai_c_tensorflow2 \
            --model ./model_quant.h5 \
            --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
            --output_dir comp/ \
            --net_name model")
        
        quantizer.dump_model(
            quantized_model, 
            dataset=self.x_test[1:1024], 
            output_dir='./dump_results', 
            dump_float=True,

        )
        quantizer.dump_model(
            quantized_model, 
            dataset=self.x_test[1:1024], 
            output_dir='./dump_results', 
            dump_float=False,
        )
        
        dump_weights_to_txt_and_bin(quantized_model)
if __name__ == '__main__':

    utils_obj = utils('cpp/')

    w, b, data = utils_obj.NN(True)

    # utils_obj.rename()
    #out, b, w, test = utils_obj.FullyVitisAI()
    #utils_obj.write_model_h()
    #utils_obj.pynq_dpu()

