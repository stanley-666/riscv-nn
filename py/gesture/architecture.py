# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, BatchNormalization, Dense, Flatten
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Conv1D, GRU, LSTM, BatchNormalization, Activation, 
    GlobalAveragePooling1D, Dense, Flatten
)
import os
import configparser
import numpy as np
import time

from tensorflow.keras.layers import Reshape
from tensorflow.keras.losses import CategoricalCrossentropy

import tensorflow as tf

try:
    import torch  # optional
except ImportError:  # pragma: no cover
    torch = None

def focal_loss_advanced(y_true, y_pred, gamma=[0.2, 1.5, 1.5, 1.5], beta=4.0, alpha=2.0):
    beta = tf.constant(4.0, dtype=tf.float32)  # 使用浮点类型的常量
    y_true = tf.cast(y_true, tf.float32)  # 确保 y_true 是浮点数

    """
    高階 Focal Loss，針對每個類別使用不同的 alpha 權重，並根據樣本類型調整損失。
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) 

    gamma= tf.convert_to_tensor(gamma)

    pos_loss = (1 - y_pred) ** alpha * tf.math.log(y_pred)

    neg_loss = (1 - y_true) ** beta * (y_pred ** alpha) * tf.math.log(1 - y_pred)

    # 計算整體損失 (選擇正樣本或負樣本的對應損失)
    loss = -tf.reduce_sum(
        y_true * gamma * pos_loss + (1 - y_true) * gamma * neg_loss,
        axis=-1
    )

    return tf.reduce_mean(loss)

def focal_loss_piecewise(y_true, y_pred, beta=4.0, alpha=2.0):
    """
    依照圖二公式：
      L = - sum_k {
            (1 - ŷ_k)^α log(ŷ_k),                if y_k = 1
            (1 - y_k)^β * (ŷ_k)^α * log(1-ŷ_k), otherwise
          }
    """
    # 確保型態
    y_true = tf.cast(y_true, tf.float32)
    # 防止 log(0)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)

    # 正樣本分支： (1 - ŷ)^α * log(ŷ)
    pos_term = tf.pow(1. - y_pred, alpha) * tf.math.log(y_pred)
    # 負樣本分支： (1 - y)^β * (ŷ)^α * log(1 - ŷ)
    neg_term = tf.pow(1. - y_true, beta) * tf.pow(y_pred, alpha) * tf.math.log(1. - y_pred)

    # 根據 y_true 決定用哪一支
    # tf.equal(y_true, 1.) 會回傳 True/False 的 mask
    loss_per_class = tf.where(tf.equal(y_true, 1.0), pos_term, neg_term)

    # sum over classes, mean over batch
    loss = -tf.reduce_sum(loss_per_class, axis=-1)
    return tf.reduce_mean(loss)

def focal_loss_pos_only(y_true, y_pred, alpha=3.0):
    """
    只保留正樣本那一項，且不做任何 normalization：
      L = - Σ_k [ Y_k * (1 - ŷ_k)^α * log(ŷ_k) ]
    最後對 batch 做 mean。
    """
    # 1. 轉成 float 並 clip，避免 log(0)
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)

    # 2. 計算每個類別的正樣本 focal loss 項
    pos_term = tf.pow(1. - y_pred, alpha) * tf.math.log(y_pred)
    loss_per_class = y_true * pos_term

    # 3. sum over classes，再 - 號，最後 mean over batch
    loss = -tf.reduce_sum(loss_per_class, axis=-1)   # shape = (batch,)
    return tf.reduce_mean(loss)

class GestureModel(object):
    def __init__(self, channel, class_num, windows_size=128):
        self.model = None
        self.channel = channel
        self.windows_size = windows_size
        self.class_num = class_num


    def build_model(self):
        signal_input = Input(shape=(self.windows_size, self.channel), name="signal_input")
        out = Conv1D(32, 3, strides=1, activation=None, use_bias=False, padding='valid', name='conv1d_1')(signal_input)
        out = BatchNormalization(name='batch_normalization_1')(out)
        out = Activation('relu', name='relu_1')(out)
        # print('out1 :',out.shape)

        out = Conv1D(64, 3, strides=1, activation=None, use_bias=False, padding='valid', name='conv1d_2')(out)
        out = BatchNormalization(name='batch_normalization_2')(out)
        out = Activation('relu', name='relu_2')(out)
        # print('out2 :',out.shape)

        out = Conv1D(128, 3, strides=1, activation=None, use_bias=False, padding='valid', name='conv1d_3')(out)
        out = BatchNormalization(name='batch_normalization_3')(out)
        out = Activation('relu', name='relu_3')(out)
        # print('out3 :',out.shape)

        out = Conv1D(256, 3, strides=1, activation=None, use_bias=False, padding='valid', name='conv1d_4')(out)
        out = BatchNormalization(name='batch_normalization_4')(out)
        out = Activation('relu', name='relu_4')(out)
        # print('out4 :',out.shape)

        out = Conv1D(256, 1, strides=1, activation=None, use_bias=False, padding='valid', name='conv1d_5')(out)
        out = BatchNormalization(name='batch_normalization_5')(out)
        out = Activation('relu', name='relu_5')(out)
        print('out5 :',out.shape)


        # 計算 batch_size
        # batch_size = tf.shape(out)[0]  # 動態計算 batch_size

        # 使用 tf.zeros 動態生成初始隱藏狀態和細胞狀態
        # hidden_state = tf.zeros([batch_size, 128], dtype=tf.float32)
        # cell_state = tf.zeros([batch_size, 128], dtype=tf.float32)

        # 初始化 LSTM 層的初始狀態
        # initial_state = [hidden_state, cell_state]

        # 傳遞給 LSTM 層
        # out = LSTM(128, return_sequences=False, name="lstm_1")(out, initial_state=initial_state)

        # print(f"Shape after LSTM: {out.shape}")




        # Add Reshape before LSTM
        # out = Reshape((-1, 256))(out)  # 根据实际输出大小调整
        # out = LSTM(128, return_sequences=False)(out)  # 在 flatten 之前加 LSTM
        # print('out6 :',out.shape)

        out = Flatten(name='flatten_1')(out)    
        # print('out7 :',out.shape)

        # Final output layer with softmax activation for multi-class classification
        out = Dense(self.class_num, activation='softmax', name='softmax_1')(out)
        # print('out8 :',out.shape)

        self.model = keras.Model(
            inputs=signal_input,
            outputs=out
        )

        # Compile the model with the focal loss function
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            # loss=focal_loss_multi_class  # 使用新版本的 Focal Loss
            loss=focal_loss_advanced # 使用高階 Focal Loss
            # loss=focal_loss_piecewise
            # loss=focal_loss_pos_only
            # loss=custom_focal_loss
            # loss=CustomFocalLoss(alpha=2.0, beta=4.0, weights={0: 1.0, 1: 18.0, 2: 18.0, 3: 20.0})
            # optimizer=Adam(learning_rate=0.0001),
            # loss=focal_loss(gamma=2.0, alpha=1.0)
            # loss=focal_loss(gamma=2.0, alpha=0.25)
            # loss = mean_squared_error,
            # loss = CategoricalCrossentropy(from_logits=False)
            # loss=combined_loss  # 使用自訂的混合損失函數
        )

        self.model.summary()

        return self.model
    

# **加入程式1和程式2進行形狀檢查**
def check_tensor_shape(input_tensor):
    """
    替換可能導致問題的 NumPy 操作，檢查形狀和數據。
    """
    print(f"Input Tensor Shape (Static): {input_tensor.shape}")  # 靜態形狀
    dynamic_shape = tf.shape(input_tensor)  # 動態形狀
    print(f"Input Tensor Shape (Dynamic): {dynamic_shape}")
    tf_prod = tf.reduce_prod(dynamic_shape)  # 計算形狀乘積
    print(f"Product of Tensor Shape: {tf_prod.numpy()} (if in eager execution)")
    return dynamic_shape

def print_layer_weights_and_biases(model):
    """
    列印模型中每一層的權重和偏置，並顯示其數值範圍。
    """
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        if layer.weights:  # 如果該層有權重和偏置
            for weight in layer.weights:
                weight_values = weight.numpy()
                weight_min = weight_values.min()
                weight_max = weight_values.max()
                print(f"  Weight Name: {weight.name}")
                print(f"  Weight Shape: {weight.shape}")
                print(f"  Weight Values (Preview, First 20): {weight_values.flatten()[:20]}")
                print(f"  Weight Range: Min={weight_min}, Max={weight_max}\n")  # 顯示範圍
        elif torch is not None and isinstance(layer, torch.nn.BatchNorm1d):
            # 只有在安裝了 torch 且圖層為 torch BN 時才會進入
            print(f"Layer: {layer.name}")
            print("BatchNorm gamma (scale):", layer.weight.data.cpu().numpy())
            print("BatchNorm beta (bias):", layer.bias.data.cpu().numpy())
            print("BatchNorm running mean:", layer.running_mean.data.cpu().numpy())
            print("BatchNorm running variance:", layer.running_var.data.cpu().numpy())
        else:
            print("  No Weights or Biases in this layer.")
        print("\n")



if __name__ == "__main__":
    import numpy as np

    model_path = os.environ.get("GESTURE_MODEL_H5", "model.h5")
    sample_path = os.environ.get("GESTURE_SAMPLE_NPY", "prepared_inputs/1_1_2025-03-06-11-54-13_Frank_win0.npy")

    custom_objects = {name: obj for name, obj in globals().items() if callable(obj)}
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    gesture = np.array
    if sample_path and os.path.exists(sample_path):
        arr = np.load(sample_path)
        if arr.ndim == 2:
            arr = arr[None, ...]  # [1,C,W] 或 [1,W,C]
        if arr.shape[1] == 5 and arr.shape[2] != 5:
            arr = np.transpose(arr, (0, 2, 1))  # channels-first -> NHWC
        elif arr.shape[2] != 5:
            raise ValueError(f"Unexpected input shape {arr.shape}, expected 5 channels.")
        preds = model(arr, training=False).numpy()[0]
        formatted = ", ".join(f"{x:.6f}" for x in preds)
        print(f"output: [{formatted}]")
        print("argmax:", int(np.argmax(preds)))
    else:
        # 沒提供樣本路徑就跑原本的 warm-up for reference
        net = GestureModel(5, class_num=4, windows_size=50)
        @tf.function
        def infer(x):
            return model(x, training=False)
        dummy = tf.random.normal((1, net.windows_size, net.channel))
        _ = infer(dummy)
        t0 = time.time()
        _ = infer(dummy)
        t1 = time.time()
        print(f"Single-window inference latency (graph infer): {(t1 - t0)*1000:.2f} ms")
