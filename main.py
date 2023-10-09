#=====================
#注意: 理解させる気はない
#=====================
import numpy as np
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def init_layers(nn_arcitecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_arcitecture)
    params_values = {}

    for idx, layer in enumerate(nn_arcitecture):
        layer_idx = idx + 1
        layerinp_size = layer["input_dim"]
        layerout_size = layer["output_dim"]

        params_values["W" + str(layer_idx)] = np.random.randn(layerout_size, layerinp_size) * 0.1

        params_values["b" + str(layer_idx)] = np.random.randn(layerout_size, 1) * 0.1

    return  params_values
#対称性の色々でベクトルと行列を乱数で満たす。でも線形代数しね

def sigmoid(Z):
    return 1/(1+np.exp(-Z))
#便利。

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is relu:
        activation_func = relu

    elif activation is "sigmoid":
        activation_func = sigmoid

    else:
        raise Exception("Non-supported activation function")
    
    return activation_func(Z_curr), Z_curr

#前方伝搬知らないからコピペ

def full_forward_propagation(X, params_value, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_func_curr = layer["activation"]
        W_curr = params_value["W" + str(layer_idx)]
        b_curr = params_value["b" + str(layer_idx)]

        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_func_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory

#損失関数も知らないwww 俺絶対ニューラルネットワーク作るべきじゃないwww
#↓コピペ

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1/m*(np.dot(Y, np.log(Y_hat).T)) + np.dot(1 - Y, np.log(1 - Y_hat).T)

    return np.squeeze(cost)

def convert_prob_into_class(probs):
         probs_ = np.copy(probs)
         probs_[probs_ > 0.5] = 1
         probs_[probs_ <= 0.5] = 0
         return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat) #←この謎の関数が定義されてなかったから結局全部理解することになったわ。もっとちゃんとしろ。(?)
    return (Y_hat_ == Y).all(axis=0).mean()

#どうやら後方伝搬の方がむずいらしい。まぁ微積分学と線形代数の組み合わせなんかクソむずいに決まってるし、3B1Bの動画出たらみようかな...
#↓コピペ

#oct. 9。やっぱ休憩。
#could commit with ssh! brilliant!!
